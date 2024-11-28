import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math, copy, time
from torch.autograd import Variable
import random
import os
import pandas


#load training functions
from run_finetuning import run_finetuning_function

#parameters
num_repetitions = 10
pretrain = 'ri'
model_type = 'Transfer_Cnn6'
model_load_path = "Cnn6.pth"
#model_save_path='model_test_pretrain_all_mfcc_'+str(num_patient_noise_files) + '_' + str(num_control_noise_files) + '.ckpt'

#run finetuning function
for index in range(num_repetitions):
    output_filename = 'experiments_after_review/ri_CNN6_fase_1' + '_' + str(index) + '.txt'
    model_save_path = 'saved_models/model_pretrained_CNN6_' + '_' + str(index) + '.ckpt'
    run_finetuning_function(output_filename, model_type, model_load_path, model_save_path, pretrain)


