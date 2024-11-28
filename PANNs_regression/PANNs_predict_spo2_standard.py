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
import sklearn
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

from scipy.stats import pearsonr
from torchmetrics import R2Score, ConcordanceCorrCoef


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
        self.ff_mid = nn.Linear(2048, 10, bias=True)
        #self.fc_transfer = nn.Linear(2048, classes_num, bias=True)
        self.fc_transfer = nn.Linear(10, classes_num, bias=True)
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

        #clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        embedding = self.gelu(self.ff_mid(embedding))
        output = self.fc_transfer(embedding)
        output_dict['regressor'] = output
 
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
        self.ff_mid = nn.Linear(512, 10, bias=True)
        #self.fc_transfer = nn.Linear(512, classes_num, bias=True)
        self.fc_transfer = nn.Linear(10, classes_num, bias=True)
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
        #print(embedding.shape)
        #print(self.fc_transfer(embedding).shape)
        #clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)

        embedding = self.gelu(self.ff_mid(embedding))
        output = self.fc_transfer(embedding)
        output_dict['regressor'] = output
 
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
        self.ff_mid = nn.Linear(512, 10, bias=True)
        #self.fc_transfer = nn.Linear(512, classes_num, bias=True)
        self.fc_transfer = nn.Linear(10, classes_num, bias=True)
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
        #print(embedding.shape)
        #print(self.fc_transfer(embedding).shape)
        #clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)

        embedding = self.gelu(self.ff_mid(embedding))
        output = self.fc_transfer(embedding)
        output_dict['regressor'] = output
 
        return output_dict
        
        
def extract_index_from_path(data_path_with_index):
    position_in_path = data_path_with_index.rfind('_')
    file_path = data_path_with_index[:position_in_path]
    index = int(data_path_with_index[(position_in_path+1):])
    return file_path, index
    
    
import pandas

folder = '../alunos_finger_Tinder/SPIRA_Dataset_V2/'

def create_data_paths_list(csv_file, folder=folder):
    csv = pandas.read_csv(csv_file)

    file_paths = []
    for file_path in csv['arquivo']:
        data_path_with_index = folder+file_path
        file_paths.append(data_path_with_index)
    
    oxigen_saturation_list = []
    for oxigen_saturation_val in csv['oxigenacao']:
        oxigen_saturation_list.append(oxigen_saturation_val)

    #merge
    data_paths_list = []
    for index in range(len(file_paths)):
        data_path_with_index = file_paths[index]
        oxigen_saturation_val = oxigen_saturation_list[index]
        data_path_oxygen_saturation = (data_path_with_index, oxigen_saturation_val)
        data_paths_list.append(data_path_oxygen_saturation)
        
    random.shuffle(data_paths_list)
    
    return data_paths_list
        
training_csv_file = '../alunos_finger_Tinder/SPIRA_Dataset_V2/metadata_train_index_10.csv'
data_paths_train = create_data_paths_list(training_csv_file)
#print(data_paths_train)

eval_csv_file = '../alunos_finger_Tinder/SPIRA_Dataset_V2/metadata_eval_index_10.csv'
data_paths_valid = create_data_paths_list(eval_csv_file)

test_csv_file = '../alunos_finger_Tinder/SPIRA_Dataset_V2/metadata_test_index_10.csv'
data_paths_test = create_data_paths_list(test_csv_file)


#build function to process data in batches
def process_batches(data_paths, number_coeffs, min_frequency, max_frequency, batch_size, pretrain, path_index):
    
    #parameters below maybe should be defined elsewhere
    #set audio length in seconds - this is max length of audios
    audio_length = 4
    device = 'cuda'
    new_sample_rate = 16000
    
    
    
    ################################
    
    data_batch = []
    
    data_target_list = []

    while len(data_batch) < batch_size and path_index < len(data_paths):
        data_path_oxygen_saturation = data_paths[path_index]
        data_path_with_index = data_path_oxygen_saturation[0]
        oxygen_saturation_val = data_path_oxygen_saturation[1]
        
        data_path, index = extract_index_from_path(data_path_with_index)
        sample_rate = torchaudio.info(data_path).sample_rate
        
        data_elem, sample_rate = torchaudio.load(data_path, frame_offset=index*sample_rate, num_frames = audio_length*sample_rate)
        #downsampling to fit gpu memory
        data_elem = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(data_elem)
        sample_rate = new_sample_rate
        data_elem = data_elem[0]
        
        data_batch.append(data_elem)
        
        #for supervised training we store whether the file comes from patient/healthy group
        data_target_list.append(oxygen_saturation_val)
        #######################
        
        path_index +=1
            
    #convert list to torch tensor (pads different audio lengths to same size)
    data_batch = nn.utils.rnn.pad_sequence(data_batch, batch_first=True)
    
    data_batch = data_batch.to(device)
    
    #for supervised training
    data_target_list = torch.FloatTensor(data_target_list)
    data_target_list = data_target_list.to(device)
    ###########################
    

    return data_batch, data_target_list, path_index
    
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
        data_batch, data_target_list, path_index = process_batches(data_paths, number_coeffs, min_frequency, max_frequency, batch_size, pretrain, path_index)
        b_size = data_batch.shape[0]
        #pass data through transformer
        #print(data_batch.shape)
        out = model.forward(data_batch)
        #compute loss
        #print('out', out.shape)
        #print('data_batch', data_batch.shape)
        if pretrain=='pretrain':
            #print('data_batch')
            loss = loss_compute(out, data_batch, training)
        else:
            loss, output, target = loss_compute(out, data_target_list, training)
        
        outputs.append(output)
        targets.append(target)

        total_loss += loss
        avg_loss = avg_loss*0.99 + loss*0.01
        total_tokens += b_size
        tokens += b_size
        true_avg_loss = total_loss/step_index
        
        #if path_index > 10:
        #    break
        
        if step_index % 5 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f True Avg Loss: %f" %
                    (step_index, avg_loss, tokens / elapsed, true_avg_loss))
            start = time.time()
            tokens = 0

    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)
    
    abs_diff = np.abs(outputs-targets)
    diff_mean = np.mean(abs_diff)
    diff_std = np.std(abs_diff)
    
    corr_matrix = np.corrcoef(targets[:,0], outputs[:,0])
    corr = corr_matrix[0,1]
    R_sq = corr**2
    #Calculo Pearson https://www.geeksforgeeks.org/python-pearson-correlation-test-between-two-variables/
    pearson, _ = pearsonr(outputs[:,0], targets[:,0])

    print("Loss:", true_avg_loss)
    print("R2:", R_sq)
    print("Pearson:", pearson)

    if training == False:
        for res in ["Saida: {}  Alvo: {}".format(x,y) for x,y in zip(outputs,targets)] :
            print(res)

    print("Diff mean:", diff_mean)
    print("Diff std:", diff_std)

    return total_loss / (total_tokens), avg_loss, true_avg_loss, diff_mean, diff_std, R_sq, pearson
    
    
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
        #return self.factor * \
        #    (self.model_size ** (-0.5) *
        #    min(step ** (-0.5), step * self.warmup ** (-1.5)))
        return 1e-4
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
            
            
            
class LossCompute:
    "A simple loss compute and train function."
    def __init__(self, model, opt=None, pretrain='pretrain', loss_function = 'MAE'):
        self.model = model
        self.opt = opt
        self.pretrain = pretrain
        self.loss_function = loss_function
        
    def __call__(self, output_dict, y, training):
        train_acc = 0
        if self.pretrain == 'pretrain':
            L1_loss = nn.L1Loss()
            loss = L1_loss(output_dict['clipwise_output'], y)
        else:
            if self.loss_function == 'MAE':#mae loss
                loss_fct = nn.L1Loss()
            else:#MSE loss
                loss_fct = nn.MSELoss()
            #check the indices for cross entropy loss
            y = y.unsqueeze(1)
            loss = loss_fct(output_dict['regressor'], y)
            preds = output_dict['regressor'].detach().cpu().clone()
            y_true = y.detach().cpu().clone()

        if training == True:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.optimizer.zero_grad()
        return loss.data.item(), preds, y_true
        
        
args = {}
args['sample_rate']= 32000
args['window_size']= 1024
args['hop_size']=320
args['mel_bins']=64
args['fmin']=0
args['fmax']=32000
args['model_type']="Transfer_Cnn6"
args['pretrained_checkpoint_path']="../PANNS_classification/Cnn6.pth"
args['freeze_base']=False
args['cuda']=True



pretrain = 'ri'
loss_function = 'MSE'
d_model = 2048

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
classes_num = 1#config.classes_num
pretrain = True if pretrained_checkpoint_path else False


# Model
Model = eval(model_type)
model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base)
model_opt = NoamOpt(d_model, 1, 10,
        torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# Load pretrained model
if pretrain:
    logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
    model.load_from_pretrain(pretrained_checkpoint_path)


if 'cuda' in device:
    model.to(device)

print('Load pretrained model successfully!')

    
avg_loss=0
model_path = 'model_test_mel.ckpt'
min_frequency = 0.0
best_loss = 1e10
max_frequency = None

for epoch in range(10):
    print("Epoch= ", epoch)
    model.train()
    loss, avg_loss, _, _, _, _, _ = run_epoch(model, 
              LossCompute(model, model_opt, pretrain, loss_function),
              data_paths_train, pretrain=pretrain, training=True, avg_loss=avg_loss,
              min_frequency=min_frequency, max_frequency=max_frequency, number_coeffs=64)
    model.eval()
    with torch.no_grad():
        loss, avg_loss, true_avg_loss, diff_mean, diff_std, r2, pearson = run_epoch(model, 
                    LossCompute(model, None, pretrain, loss_function),
                    data_paths_valid, pretrain=pretrain, training=False,
                    min_frequency=min_frequency, max_frequency=max_frequency, number_coeffs=64)
        print('true_avg_loss=', true_avg_loss)
        print('diff_mean=', diff_mean)
        print('diff_std=', diff_std)
        print('r2=', r2)
        print('pearson=', pearson)
        print('loss=', loss)
        
    if best_loss > true_avg_loss:
        best_loss = true_avg_loss
        print('Saving model')
        torch.save({
            'model_state_dict': model.state_dict()
            }, model_path)
            
    print("Best loss= ", best_loss, ", Actual loss: ", true_avg_loss) 

model_path = 'model_test_mel.ckpt'
Model = eval(model_type)
model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base)

# Load trained model
logging.info('Load pretrained model from {}'.format(model_path))
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

if 'cuda' in device:
    model.to(device)
V=64
model.eval()
with torch.no_grad():
    loss, avg_loss, true_avg_loss, diff_mean, diff_std, r2, pearson = run_epoch(model, 
                    LossCompute(model, None, pretrain, loss_function),
                    data_paths_test, pretrain=pretrain, training=False,
                    min_frequency=min_frequency, max_frequency=max_frequency, number_coeffs=64)
                    
    print('true_avg_loss=', true_avg_loss)
    print('diff_mean=', diff_mean)
    print('diff_std=', diff_std)
    print('r2=', r2)
    print('pearson=', pearson)
    print('loss=', loss)
