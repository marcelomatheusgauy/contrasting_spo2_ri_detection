# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# AST: https://github.com/YuanGongND/ast
# --------------------------------------------------------
import csv, os, sys
import json
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DistributedSampler, WeightedRandomSampler
import torch.distributed as dist
import random
import math
import pandas as pd
import torchaudio


def extract_index_from_path(data_path_with_index):
    if (data_path_with_index.find('opus') != -1):
        _position_in_path = data_path_with_index.find('opus')+4
    elif (data_path_with_index.find('mp3') != -1):
        _position_in_path = data_path_with_index.find('mp3')+3
    else:
        _position_in_path = data_path_with_index.find('ogg')+3
    data_path = data_path_with_index[:_position_in_path]
    index = int(data_path_with_index[_position_in_path+1:])
    return data_path, index
    

class DistributedSamplerWrapper(DistributedSampler):
    def __init__(
            self, sampler, dataset,
            num_replicas=None,
            rank=None,
            shuffle: bool = True):
        super(DistributedSamplerWrapper, self).__init__(
            dataset, num_replicas, rank, shuffle)
        # source: @awaelchli https://github.com/PyTorchLightning/pytorch-lightning/issues/3238
        self.sampler = sampler

    def __iter__(self):
        if self.sampler.generator is None:
            self.sampler.generator = torch.Generator()
        self.sampler.generator.manual_seed(self.seed + self.epoch)
        indices = list(self.sampler)
        if self.epoch == 0:
            print(f"\n DistributedSamplerWrapper :  {indices[:10]} \n\n")
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)
        
class DistributedWeightedSampler(Sampler):
    #dataset_train, samples_weight,  num_replicas=num_tasks, rank=global_rank
    def __init__(self, dataset, weights, num_replicas=None, rank=None, replacement=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.weights = torch.from_numpy(weights)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # # get targets (you can alternatively pass them in __init__, if this op is expensive)
        # targets = self.dataset.targets
        # # select only the wanted targets for this subsample
        # targets = torch.tensor(targets)[indices]
        # assert len(targets) == self.num_samples
        # # randomly sample this subset, producing balanced classes
        # weights = self.calculate_weights(targets)
        weights = self.weights[indices]

        subsample_balanced_indicies = torch.multinomial(weights, self.num_samples, self.replacement)
        # now map these target indicies back to the original dataset index...
        dataset_indices = torch.tensor(indices)[subsample_balanced_indicies]
        return iter(dataset_indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None, use_fbank=False, fbank_dir=None, roll_mag_aug=False, load_video=False, mode='train'):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.use_fbank = use_fbank
        self.fbank_dir = fbank_dir

        self.data = data_json['data']
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        if 'multilabel' in self.audio_conf.keys():
            self.multilabel = self.audio_conf['multilabel']
        else:
            self.multilabel = False
        print(f'multilabel: {self.multilabel}')
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        print('Dataset: {}, mean {:.3f} and std {:.3f}'.format(self.dataset, self.norm_mean, self.norm_std))
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')
        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        self.roll_mag_aug=roll_mag_aug
        print(f'number of classes: {self.label_num}')
        print(f'size of dataset {self.__len__()}')


    def _roll_mag_aug(self, waveform):
        waveform=waveform.numpy()
        idx=np.random.randint(len(waveform))
        rolled_waveform=np.roll(waveform,idx)
        mag = np.random.beta(10, 10) + 0.5
        return torch.Tensor(rolled_waveform*mag)

    def _wav2fbank(self, filename, filename2=None):
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
            if self.roll_mag_aug:
                waveform = self._roll_mag_aug(waveform)
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if self.roll_mag_aug:
                waveform1 = self._roll_mag_aug(waveform1)
                waveform2 = self._roll_mag_aug(waveform2)

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()
        # 498 128, 998, 128
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        # 512
        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda


    def _fbank(self, filename, filename2=None):
        if filename2 == None:
            fn1 = os.path.join(self.fbank_dir, os.path.basename(filename).replace('.wav','.npy'))
            fbank = np.load(fn1)
            return torch.from_numpy(fbank), 0
        else:
            fn1 = os.path.join(self.fbank_dir, os.path.basename(filename).replace('.wav','.npy'))
            fn2 = os.path.join(self.fbank_dir, os.path.basename(filename2).replace('.wav','.npy'))
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)
            fbank = mix_lambda * np.load(fn1) + (1-mix_lambda) * np.load(fn2)  
            return torch.from_numpy(fbank), mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup: # for audio_exp, when using mixup, assume multilabel
            datum = self.data[index]
            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]

            # get the mixed fbank
            if not self.use_fbank:
                fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
            else:
                fbank, mix_lambda = self._fbank(datum['wav'], mix_datum['wav'])
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += 1.0-mix_lambda
            label_indices = torch.FloatTensor(label_indices)
        # if not do mixup
        else:
            datum = self.data[index]
            label_indices = np.zeros(self.label_num)
            if not self.use_fbank:
                fbank, mix_lambda = self._wav2fbank(datum['wav'])
            else:
                fbank, mix_lambda = self._fbank(datum['wav'])
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0

            if self.multilabel:
                label_indices = torch.FloatTensor(label_indices)
            else:
                # remark : for ft cross-ent
                label_indices = int(self.index_dict[label_str])
        # SpecAug for training (not for eval)
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = fbank.transpose(0,1).unsqueeze(0) # 1, 128, 1024 (...,freq,time)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank) # (..., freq, time)
        fbank = torch.transpose(fbank.squeeze(), 0, 1) # time, freq
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        if self.noise == True: # default is false, true for spc
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank.unsqueeze(0), label_indices, datum['wav']

    def __len__(self):
        return len(self.data)


class SpiraDataset(Dataset):
    def __init__(self, audio_conf, folder_path=None, dataset_csv=None, use_fbank=False, fbank_dir=None, roll_mag_aug=False, load_video=False, mode='train'):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_csv
        """
        self.folder_path = folder_path
        self.datapath = dataset_csv
        with open(dataset_csv, 'r') as fp:
            data_csv = pd.read_csv(fp)
        self.use_fbank = use_fbank
        self.fbank_dir = fbank_dir

        self.data = data_csv
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        if 'multilabel' in self.audio_conf.keys():
            self.multilabel = self.audio_conf['multilabel']
        else:
            self.multilabel = False
        print(f'multilabel: {self.multilabel}')
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        print('Dataset: {}, mean {:.3f} and std {:.3f}'.format(self.dataset, self.norm_mean, self.norm_std))
        self.noise = self.audio_conf.get('noise')
        self.normalize = self.audio_conf.get('normalize')
        if self.noise == True:
            print('now use noise augmentation')
        #self.index_dict = make_index_dict(label_csv)
        self.label_num = 2#len(self.index_dict)
        self.roll_mag_aug=roll_mag_aug
        print(f'number of classes: {self.label_num}')
        print(f'size of dataset {self.__len__()}')


        self.noise_file_paths = []
        noise_folder = '../SPIRA_Dataset_V2/Ruidos-Hospitalares_V1/Ruidos-Hospitalares/Ruidos/Hospitalares-validados/'
        for file in os.listdir(noise_folder):
            if file.find(".wav") != -1:
                data_path = noise_folder+file
                self.noise_file_paths.append(data_path)

    def _roll_mag_aug(self, waveform):
        waveform=waveform.numpy()
        idx=np.random.randint(len(waveform))
        rolled_waveform=np.roll(waveform,idx)
        mag = np.random.beta(10, 10) + 0.5
        return torch.Tensor(rolled_waveform*mag)

    def add_noise(self, data_elem, label, noise_file_paths):
        if self.noise == True: # default is false
            noise_max_amp = 0.19233719
            noise_min_amp = 0.033474047
            num_patient_noise_files = 1
            num_control_noise_files = 1
            if label == 1:
                for noise_index in range(num_patient_noise_files):
                    #select random file
                    random_noise_index = random.randint(0, len(noise_file_paths)-1)
                    noise_path = noise_file_paths[random_noise_index]
                    #load audio file
                    noise_file = torchaudio.load(noise_path, normalize = self.normalize)[0]
                    audio_len = data_elem.size(0)
                    noise_len = noise_file[0].size(0)
                    noise_start = random.randint(0,noise_len-(audio_len+1))
                    noise_file = noise_file[0][noise_start:noise_start+audio_len]
                    #should I play with the amplitude of the noise file
                    max_amp = random.uniform(noise_min_amp, noise_max_amp)
                    reduct_factor = max_amp/float(noise_file.max().numpy())
                    noise_file = noise_file*reduct_factor
                    data_elem = data_elem + noise_file
            else:
                for noise_index in range(num_control_noise_files):
                    #select random file
                    random_noise_index = random.randint(0, len(noise_file_paths)-1)
                    noise_path = noise_file_paths[random_noise_index]
                    #load audio file
                    noise_file = torchaudio.load(noise_path, normalize = self.normalize)[0]
                    audio_len = data_elem.size(0)
                    noise_len = noise_file[0].size(0)
                    noise_start = random.randint(0,noise_len-(audio_len+1))
                    noise_file = noise_file[0][noise_start:noise_start+audio_len]
                    #should I play with the amplitude of the noise file
                    max_amp = random.uniform(noise_min_amp, noise_max_amp)
                    reduct_factor = max_amp/float(noise_file.max().numpy())
                    noise_file = noise_file*reduct_factor
                    data_elem = data_elem + noise_file
        
        return data_elem


    def _wav2fbank(self, filename, dat_index, label, filename2=None, dat_index2=None, label_2 = None):
        audio_length = 4
        new_sample_rate = 16000
        if filename2 == None:
            sample_rate = torchaudio.info(filename).sample_rate
            waveform, sr = torchaudio.load(filename, frame_offset=dat_index*sample_rate, num_frames = audio_length*sample_rate, normalize = self.normalize)
            waveform = torchaudio.transforms.Resample(sr, new_sample_rate)(waveform)
            sr = new_sample_rate
            waveform = self.add_noise(waveform, label, self.noise_file_paths)
            
            waveform = waveform.mean(axis=0, keepdims=True)
            
            waveform = waveform - waveform.mean()
            if self.roll_mag_aug:
                waveform = self._roll_mag_aug(waveform)
        # mixup
        else:
            sample_rate = torchaudio.info(filename).sample_rate
            waveform1, sr = torchaudio.load(filename, frame_offset=dat_index*sample_rate, num_frames = audio_length*sample_rate, normalize = self.normalize)
            waveform1 = torchaudio.transforms.Resample(sr, new_sample_rate)(waveform1)
            waveform1 = self.add_noise(waveform1, label, self.noise_file_paths)
            
            waveform1 = waveform1.mean(axis=0, keepdims=True)
            
            waveform2, _ = torchaudio.load(filename2, frame_offset=dat_index2*sample_rate, num_frames = audio_length*sample_rate, normalize = self.normalize)
            waveform2 = torchaudio.transforms.Resample(sr, new_sample_rate)(waveform2)
            sr = new_sample_rate
            waveform2 = self.add_noise(waveform2, label_2, self.noise_file_paths)

            waveform2 = waveform2.mean(axis=0, keepdims=True)
            
            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if self.roll_mag_aug:
                waveform1 = self._roll_mag_aug(waveform1)
                waveform2 = self._roll_mag_aug(waveform2)

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()
        # 498 128, 998, 128
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        # 512
        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda


    def _fbank(self, filename, filename2=None):
        if filename2 == None:
            fn1 = os.path.join(self.fbank_dir, os.path.basename(filename).replace('.wav','.npy'))
            fbank = np.load(fn1)
            return torch.from_numpy(fbank), 0
        else:
            fn1 = os.path.join(self.fbank_dir, os.path.basename(filename).replace('.wav','.npy'))
            fn2 = os.path.join(self.fbank_dir, os.path.basename(filename2).replace('.wav','.npy'))
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)
            fbank = mix_lambda * np.load(fn1) + (1-mix_lambda) * np.load(fn2)  
            return torch.from_numpy(fbank), mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup: # for audio_exp, when using mixup, assume multilabel
            datum = self.data.iloc[index]
            dat_file_path, dat_index = extract_index_from_path(datum['arquivo'])
            dat_file_path = self.folder_path+dat_file_path
            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data.iloc[mix_sample_idx]
            mix_dat_file_path, mix_dat_index = extract_index_from_path(mix_datum['arquivo'])
            mix_dat_file_path = self.folder_path+mix_dat_file_path
            
            # get the mixed fbank
            if not self.use_fbank:
                fbank, mix_lambda = self._wav2fbank(dat_file_path, dat_index, datum['oxigenacao'], mix_dat_file_path, mix_dat_index, mix_datum['oxigenacao'])
            else:
                fbank, mix_lambda = self._fbank(datum['arquivo'], mix_datum['arquivo'])
            # initialize the label
            label_indices = np.zeros(2)
            if datum['oxigenacao'] > 0.92:
                label_indices[1] += mix_lambda
            else:
                label_indices[0] += mix_lambda
            if mix_datum['oxigenacao'] > 0.92:
                label_indices[1] += (1.0-mix_lambda)
            else:
                label_indices[0] += (1.0-mix_lambda)
            #label_indices[datum['oxigenacao']>0.92] += mix_lambda
            #label_indices[mix_datum['oxigenacao']>0.92] += (1.0-mix_lambda)
            label_indices = torch.FloatTensor(label_indices)
            #print(label_indices)
        # if not do mixup
        else:
            datum = self.data.iloc[index]
            dat_file_path, dat_index = extract_index_from_path(datum['arquivo'])
            dat_file_path = self.folder_path+dat_file_path
            label_indices = np.zeros(2)
            if datum['oxigenacao'] > 0.92:
                label_indices[1] = 1
            else:
                label_indices[0] = 1
            #label_indices[datum['oxigenacao']>0.92] = 1
            label_indices = torch.FloatTensor(label_indices)
            #print("label_indices:", label_indices)
            if not self.use_fbank:
                fbank, mix_lambda = self._wav2fbank(dat_file_path, dat_index, datum['oxigenacao'])
            else:
                fbank, mix_lambda = self._fbank(datum['arquivo'])

        # SpecAug for training (not for eval)
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = fbank.transpose(0,1).unsqueeze(0) # 1, 128, 1024 (...,freq,time)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank) # (..., freq, time)
        fbank = torch.transpose(fbank.squeeze(), 0, 1) # time, freq
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank.unsqueeze(0), label_indices, datum['arquivo']

    def __len__(self):
        return len(self.data)

