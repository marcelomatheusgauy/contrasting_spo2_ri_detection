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
from sklearn import metrics
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
        #print(embedding.shape)
        #print(self.fc_transfer(embedding).shape)
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
        #print(embedding.shape)
        #print(self.fc_transfer(embedding).shape)
        clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict

def extract_index_from_path(data_path_with_index):
    position_in_path = data_path_with_index.rfind('_')
    file_path = data_path_with_index[:position_in_path]
    index = int(data_path_with_index[(position_in_path+1):])
    return file_path, index
    
import pandas

def create_data_paths_list(csv_file, folder):
    csv = pandas.read_csv(csv_file)
    
    file_paths = []
    for file_path in csv['file_path']:
        data_path_with_index = folder+file_path
        file_paths.append(data_path_with_index)

    respiratory_insufficiency_list = []
    for respiratory_insufficiency_val in csv['insuficiencia_respiratoria']:
        respiratory_insufficiency_list.append(respiratory_insufficiency_val)

    first_phase_list = []
    for first_phase_val in csv['hospitalName']:
        if first_phase_val == 'first_phase':
            first_phase_list.append(True)
        else:
            first_phase_list.append(False)
            
    #merge
    data_paths_list = []
    for index in range(len(file_paths)):
        data_path_with_index = file_paths[index]
        respiratory_insufficiency_val = respiratory_insufficiency_list[index]
        first_phase_val = first_phase_list[index]
        data_path_resp_insufficiency = (data_path_with_index, respiratory_insufficiency_val, first_phase_val)
        data_paths_list.append(data_path_resp_insufficiency)
        
    random.shuffle(data_paths_list)
    
    return data_paths_list



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
def process_batches(data_paths, noise_file_paths, noise_file_paths_segunda_fase, number_coeffs, min_frequency, max_frequency, batch_size, pretrain, path_index):
    
    #parameters below maybe should be defined elsewhere
    #set audio length in seconds - this is max length of audios
    audio_length = 4
    device = 'cuda'
    new_sample_rate = 16000
    
    noise_max_amp = 0.19233719
    noise_min_amp = 0.033474047
    noise_max_amp_segunda_fase = 0.15957803
    noise_min_amp_segunda_fase = 0.021775
    
    
    ################################
    
    data_batch = []
    
    #in case we are doing supervised training we also need to store whether the file comes from healthy/unhealthy - is always computed but only used for supervised training
    data_ri_target_list = []
    
    while len(data_batch) < batch_size and path_index < len(data_paths):
        data_path_resp_insufficiency = data_paths[path_index]
        data_path_with_index = data_path_resp_insufficiency[0]
        respiratory_insufficiency_val = data_path_resp_insufficiency[1]
        first_phase_val = data_path_resp_insufficiency[2]

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
        if first_phase_val:
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
def run_epoch(model, loss_compute, data_paths, noise_file_paths, noise_file_paths_segunda_fase, output_file,  avg_loss=0, pretrain='pretrain', training=True, batch_size=16, extract_coeffs='both', min_frequency = 0.0, max_frequency=None, number_coeffs=128, mask_proportion=0., mask_consecutive_frames=7, mask_frequency_proportion=0., random_noise_proportion=0.0):
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
    outputs_prob=[]
    targets=[]
    
    #path index is the index of the audio file in the filenames list
    path_index = 0
    #step index stores the amount of steps taken by the algorithm so far
    step_index = 0
    while path_index < number_elements:
        step_index +=1
        #load the data and mask it
        data_batch, data_ri_target_list, path_index = process_batches(data_paths, noise_file_paths, noise_file_paths_segunda_fase, number_coeffs, min_frequency, max_frequency, batch_size, pretrain, path_index)
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
        else:
            loss, train_acc, f1_score, output, target = loss_compute(output_dict, data_ri_target_list, training)
        
        outputs.append(output)
        outputs_prob.append(output_dict['clipwise_output'][:,1].detach().cpu().clone())
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
                    (step_index, avg_loss, tokens / elapsed, train_acc_avg, f1_score_avg), file=output_file)
            start = time.time()
            tokens = 0

    outputs = np.concatenate(outputs)
    outputs_prob = np.concatenate(outputs_prob)
    targets = np.concatenate(targets)
    true_f1_score = metrics.f1_score(targets, outputs, labels=[0,1], average='macro')
    print('Final F1_score=', true_f1_score, file=output_file)
    true_acc_score = metrics.accuracy_score(targets, outputs)
    print('Final Accuracy=', true_acc_score, file=output_file)

    confusion_matrix = metrics.confusion_matrix(targets, outputs)

    #print(np.shape(targets))
    #print(np.shape(outputs_prob))
    fpr, tpr, thresholds = metrics.roc_curve(targets, outputs_prob)
    roc_auc = metrics.auc(fpr, tpr)

    return total_loss / (total_tokens), avg_loss, train_acc_avg, f1_score_avg, true_f1_score, true_acc_score, confusion_matrix, fpr, tpr, roc_auc, outputs, targets


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
