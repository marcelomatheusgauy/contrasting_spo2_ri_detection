import os
import sys
import numpy as np
import argparse
import h5py
import math
import random
import time
import logging
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import csv

import torch
import torchaudio
#torch.backends.cudnn.benchmark=True
#torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
 
sys.path.append("utils")
from utilities import get_filename
sys.path.append("pytorch/")
from models import *
import config



class Transfer_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14, self).__init__()
        audioset_classes_num = 527
        
        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)
        self.ff_mid = nn.Linear(2048, 10, bias=True)
        self.ff_final = nn.Linear(10,classes_num, bias=True)
        self.gelu = nn.GELU()

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']
        
        #embedding = self.gelu(self.ff_mid(embedding))
        #clipwise_output =  torch.log_softmax(self.ff_final(embedding), dim=-1)
        
        clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict
        
class Transfer_Cnn10(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn10, self).__init__()
        audioset_classes_num = 527
        
        self.base = Cnn10(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(512, classes_num, bias=True)
        self.ff_mid = nn.Linear(512, 10, bias=True)
        self.ff_final = nn.Linear(10,classes_num, bias=True)
        self.gelu = nn.GELU()
        
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        #embedding = self.gelu(self.ff_mid(embedding))
        #clipwise_output =  torch.log_softmax(self.ff_final(embedding), dim=-1)
        
        clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict
        
        
class Transfer_Cnn6(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn6, self).__init__()
        audioset_classes_num = 527
        
        self.base = Cnn6(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(512, classes_num, bias=True)
        self.ff_mid = nn.Linear(512, 10, bias=True)
        self.ff_final = nn.Linear(10,classes_num, bias=True)
        self.gelu = nn.GELU()
        
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']
        
        #embedding = self.gelu(self.ff_mid(embedding))
        #clipwise_output =  torch.log_softmax(self.ff_final(embedding), dim=-1)
        
        clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict
        
        
def extract_index_from_path(data_path_with_index):
    position_in_path = data_path_with_index.rfind('_')
    file_path = data_path_with_index[:position_in_path]
    index = int(data_path_with_index[(position_in_path+1):])
    return file_path, index
    
import pandas

folder = '../alunos_finger_Tinder/SPIRA_Dataset_V2/'

def create_data_paths_list(csv_file, folder = folder):
    csv = pandas.read_csv(csv_file)
    
    file_paths = []
    for file_path in csv['arquivo']:
        data_path = folder+file_path
        file_paths.append(data_path)

    respiratory_insufficiency_list = []
    for respiratory_insufficiency_val in csv['oxigenacao']:
        low_spO2 = respiratory_insufficiency_val<=0.92
        respiratory_insufficiency_list.append(low_spO2)
        
        
    #merge
    data_paths_list = []
    for index in range(len(file_paths)):
        data_path_with_index = file_paths[index]
        respiratory_insufficiency_val = respiratory_insufficiency_list[index]
        data_path_resp_insufficiency = (data_path_with_index, respiratory_insufficiency_val)
        data_paths_list.append(data_path_resp_insufficiency)
        
    random.shuffle(data_paths_list)
    
    return data_paths_list
        
    
#training_csv_file = '../audio_dados/CorpusAudios/train.csv'
#train_test_folder = '../../transformers_mfcc_mask/SPIRA_Dataset_V2/'
training_csv_file = '../alunos_finger_Tinder/SPIRA_Dataset_V2/metadata_train_index_7.csv'
data_paths_train = create_data_paths_list(training_csv_file, folder = folder)
#print(data_paths_train)

#validation_csv_file = '../audio_dados/CorpusAudios/validation.csv'
validation_csv_file = '../alunos_finger_Tinder/SPIRA_Dataset_V2/metadata_eval_index_7.csv'
data_paths_valid = create_data_paths_list(validation_csv_file, folder = folder)

    
#test_csv_file = '../audio_dados/CorpusAudios/test.csv'
test_csv_file = '../alunos_finger_Tinder/SPIRA_Dataset_V2/metadata_test_index_7.csv'
data_paths_test = create_data_paths_list(test_csv_file, folder = folder)


noise_max_amp = 0.19233719
noise_min_amp = 0.033474047
noise_max_amp_segunda_fase = 0.15957803
noise_min_amp_segunda_fase = 0.021775

noise_file_paths = []
noise_folder = '../../../SPIRA_Dataset_V2/Ruidos-Hospitalares_V1/Ruidos-Hospitalares/Ruidos/Hospitalares-validados/'

for file in os.listdir(noise_folder):
    if file.find(".wav") != -1:
        data_path = noise_folder+file
        noise_file_paths.append(data_path)

noise_file_paths_segunda_fase = []

noise_folder = '../../../spira_segunda_fase_coleta/audio_dados/Ruido_wav/'

for file in os.listdir(noise_folder):
    if file.find(".wav") != -1:
        data_path = noise_folder+file
        noise_file_paths_segunda_fase.append(data_path)


def add_noise(data_elem, noise_file_paths, num_noise_files, noise_max_amp, noise_min_amp):
    for noise_index in range(num_noise_files):
        #select random file
        random_noise_index = random.randint(0, len(noise_file_paths)-1)
        noise_path = noise_file_paths[random_noise_index]
        #load audio file
        noise_file = torchaudio.load(noise_path)[0]
        audio_len = data_elem.size(0)
        noise_len = noise_file[0].size(0)
        noise_start = random.randint(0,noise_len-(audio_len+1))
        noise_file = noise_file[0][noise_start:noise_start+audio_len]
        #should I play with the amplitude of the noise file
        max_amp = random.uniform(noise_min_amp, noise_max_amp)
        if noise_file.max().numpy() == 0:
            reduct_factor = 0
        else:
            reduct_factor = max_amp/float(noise_file.max().numpy())
        #reduct_factor = 1
        noise_file = noise_file*reduct_factor
        data_elem = data_elem + noise_file
    return data_elem



#build function to process data in batches
def process_batches(data_paths, number_coeffs, min_frequency, max_frequency, batch_size, pretrain, path_index):
    
    #parameters below maybe should be defined elsewhere
    #set audio length in seconds - this is max length of audios
    audio_length = 4
    device = 'cuda'
    new_sample_rate = 16000
    
    
    
    ################################
    
    data_batch = []
    
    #in case we are doing supervised training we also need to store whether the file comes from healthy/unhealthy - is always computed but only used for supervised training
    data_ri_target_list = []
    
    while len(data_batch) < batch_size and path_index < len(data_paths):
        data_path_resp_insufficiency = data_paths[path_index]
        data_path_with_index = data_path_resp_insufficiency[0]
        respiratory_insufficiency_val = data_path_resp_insufficiency[1]

        data_path, index = extract_index_from_path(data_path_with_index)
        sample_rate = torchaudio.info(data_path).sample_rate
        
        data_elem, sample_rate = torchaudio.load(data_path, frame_offset=index*sample_rate, num_frames = audio_length*sample_rate)
        #downsampling to fit gpu memory
        data_elem = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(data_elem)
        sample_rate = new_sample_rate
        data_elem = data_elem[0]
        
        if True:
            #add the noise the corresponding number of times
            if data_path.find("PTT") != -1:
                data_elem = add_noise(data_elem, noise_file_paths, 0, noise_max_amp, noise_min_amp)
            else:
                data_elem = add_noise(data_elem, noise_file_paths, 0, noise_max_amp, noise_min_amp)
        if True:
            data_elem = add_noise(data_elem, noise_file_paths_segunda_fase, 0, noise_max_amp_segunda_fase, noise_min_amp_segunda_fase)

        data_batch.append(data_elem)
        
        #for supervised training we store data about the file
        if respiratory_insufficiency_val:
            data_ri_target_list.append(1)
        else:
            data_ri_target_list.append(0)
        #######################
        
        path_index +=1
        
    #convert list to torch tensor (pads different audio lengths to same size)
    data_batch = nn.utils.rnn.pad_sequence(data_batch, batch_first=True)
    
    data_batch = data_batch.to(device)
    
    #for supervised training
    data_ri_target_list = torch.LongTensor(data_ri_target_list)
    data_ri_target_list = data_ri_target_list.to(device)
    ###########################
    

    return data_batch, data_ri_target_list, path_index
    
#function to train model
def run_epoch(model, loss_compute, data_paths, avg_loss=0, pretrain='pretrain', training=True, batch_size=16, extract_coeffs='both', min_frequency = 0.0, max_frequency=None, number_coeffs=128, mask_proportion=0., mask_consecutive_frames=7, mask_frequency_proportion=0., random_noise_proportion=0.0):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    train_acc_avg = 0
    f1_score_avg = 0
    
    number_elements = len(data_paths)
    #number_steps = int(math.ceil(number_elements/batch_size))
    
    outputs=[]
    targets=[]
    
    #path index is the index of the audio file in the filenames list
    path_index = 0
    #step index stores the amount of steps taken by the algorithm so far
    step_index = 0
    while path_index < number_elements:
        step_index +=1
        #load the data and mask it
        data_batch, data_ri_target_list, path_index = process_batches(data_paths, number_coeffs, min_frequency, max_frequency, batch_size, pretrain, path_index)
        b_size = data_batch.shape[0]
        #pass data through transformer
        #print(data_batch.shape)
        output_dict = model.forward(data_batch)
        #compute loss
        #print('out', out.shape)
        #print('data_batch', data_batch.shape)
        if pretrain == 'pretrain':
            #print('data_batch')
            loss, train_acc = loss_compute(output_dict, data_batch, training)
        elif pretrain == 'ri':
            loss, train_acc, f1_score, output, target = loss_compute(output_dict, data_ri_target_list, training)
        
        outputs.append(output)
        targets.append(target)

        total_loss += loss
        avg_loss = avg_loss*0.99 + loss*0.01
        train_acc_avg = (train_acc_avg*(step_index-1)+train_acc)/(step_index)
        f1_score_avg = (f1_score_avg*(step_index-1)+f1_score)/(step_index)
        total_tokens += b_size
        tokens += b_size
        
        #if path_index > 10:
        #    break
        
        if step_index % 5 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f Train_acc: %f F1_score: %f" %
                    (step_index, avg_loss, tokens / elapsed, train_acc_avg, f1_score_avg))
            start = time.time()
            tokens = 0

    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)
    true_f1_score = metrics.f1_score(targets, outputs, labels=[0,1], average='macro')
    print('Final F1_score=', true_f1_score)
    true_acc_score = metrics.accuracy_score(targets, outputs)
    print('Final Accuracy=', true_acc_score)
    
    return total_loss / (total_tokens), avg_loss, train_acc_avg, f1_score_avg, true_f1_score, true_acc_score
    
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        #return self.factor * \ (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
        return 0.0001
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
            
class LossCompute:
    "A simple loss compute and train function."
    def __init__(self, model, opt=None, pretrain='pretrain'):
        self.model = model
        self.opt = opt
        self.pretrain = pretrain
        
    def __call__(self, output_dict, y, training):
        train_acc = 0
        f1_score=0
        if self.pretrain == 'pretrain':
            L1_loss = nn.L1Loss()
            loss = L1_loss(output_dict['clipwise_output'], y)
        else:#respiratory insufficiency
            cross_entropy_loss = nn.CrossEntropyLoss()
            loss = cross_entropy_loss(output_dict['clipwise_output'], y)
            _, predicted = torch.max(output_dict['clipwise_output'], 1)
            train_acc = torch.sum(predicted==y)/y.shape[0]
            preds = predicted.detach().cpu().clone()
            y_true = y.detach().cpu().clone()
            f1_score = metrics.f1_score(y_true, preds, labels=[0,1], average='macro')
            
        if training == True:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.optimizer.zero_grad()
        return loss.data.item(), train_acc, f1_score, preds, y_true
        
        
args = {}
args['sample_rate']= 32000
args['window_size']= 1024
args['hop_size']=320
args['mel_bins']=64
args['fmin']=0
args['fmax']=32000
args['model_type']="Transfer_Cnn6"
args['pretrained_checkpoint_path']="Cnn6.pth"
args['freeze_base']=False
args['cuda']=True




# Arguments & parameters
sample_rate = args['sample_rate']
window_size = args['window_size']
hop_size = args['hop_size']
mel_bins = args['mel_bins']
fmin = args['fmin']
fmax = args['fmax']
model_type = args['model_type']
pretrained_checkpoint_path = args['pretrained_checkpoint_path']
freeze_base = args['freeze_base']
device = 'cuda' if (args['cuda'] and torch.cuda.is_available()) else 'cpu'
classes_num = 2#config.classes_num
pretrain = True if pretrained_checkpoint_path else False

# Model
Model = eval(model_type)
model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base)


# Load pretrained model
if pretrain:
    logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
    model.load_from_pretrain(pretrained_checkpoint_path)


if 'cuda' in device:
    model.to(device)

print('Load pretrained model successfully!')

pretrain = 'ri'
d_model = 512
model_opt = NoamOpt(d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

avg_loss=0
best_val_acc = 0
best_val_f1_score = 0
model_path = 'model_mel_'+args['model_type']+'.ckpt'
min_frequency = 0.0
max_frequency = None

for epoch in range(10):
    model.train()
    loss, avg_loss, _, _, _, _ = run_epoch(model, 
                  LossCompute(model, model_opt, pretrain),
                  data_paths_train, pretrain=pretrain, training=True, avg_loss=avg_loss,
                  min_frequency=min_frequency, max_frequency=max_frequency, number_coeffs=64)
    model.eval()
    with torch.no_grad():
        loss, _, val_acc, val_f1_score, true_val_f1_score, true_val_acc_score = run_epoch(model, 
                    LossCompute(model, None, pretrain=pretrain),
                    data_paths_valid, pretrain=pretrain, training=False,
                    min_frequency=min_frequency, max_frequency=max_frequency, number_coeffs=64)
        print(val_acc)
        print(val_f1_score)
    if epoch%5 == 0:
        print('Epoch=', epoch)
    #if best_val_acc < true_val_acc_score:
    #    best_val_acc = true_val_acc_score
    if best_val_f1_score < true_val_f1_score:
        best_val_f1_score = true_val_f1_score
        print('Saving model')
        torch.save({
            'model_state_dict': model.state_dict()
            }, model_path)
        #torch.save(model, model_path)



model_path = 'model_mel_'+args['model_type']+'.ckpt'
Model = eval(model_type)
model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base)

# Load trained model
logging.info('Load pretrained model from {}'.format(model_path))
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

if 'cuda' in device:
    model.to(device)
V=64
pretrain = 'ri'
model.eval()
with torch.no_grad():
    print(run_epoch(model,
                        LossCompute(model, None, pretrain=pretrain),
                        data_paths_valid, pretrain=pretrain, training=False, number_coeffs=64))
    print(run_epoch(model, 
                        LossCompute(model, None, pretrain=pretrain),
                        data_paths_test, pretrain=pretrain, training=False, number_coeffs=64))
