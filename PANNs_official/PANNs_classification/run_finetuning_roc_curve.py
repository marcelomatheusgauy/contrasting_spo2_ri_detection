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



#load transformers and training functions
from train_utils_roc import run_epoch, LossCompute, NoamOpt, Transfer_Cnn6, Transfer_Cnn10, Transfer_Cnn14, create_data_paths_list

def run_finetuning_function(output_filename, model_type, model_load_path, model_save_path, pretrain):

    output_file = open(output_filename, 'w')

    #first phase 
    #folder = '../../'
    #second phase
    folder = '../../../spira_segunda_fase_coleta/'
    #training_csv_file = '../audio_dados/CorpusAudios/train.csv'
    #first phase location
    #train_test_folder = '../../../transformers_mfcc_mask/SPIRA_Dataset_V2/'
    #second phase location
    train_test_folder = '../../../spira_segunda_fase_coleta/audio_dados/CorpusAudios/'
    training_csv_file = train_test_folder+'train.csv'
    data_paths_train = create_data_paths_list(training_csv_file, folder=folder)
    #print(data_paths_train)

    #validation_csv_file = '../audio_dados/CorpusAudios/validation.csv'
    validation_csv_file = train_test_folder+'validation.csv'
    data_paths_valid = create_data_paths_list(validation_csv_file, folder=folder)

    
    #test_csv_file = '../audio_dados/CorpusAudios/test.csv'
    test_csv_file = train_test_folder+'test.csv'
    data_paths_test = create_data_paths_list(test_csv_file, folder=folder)




    noise_file_paths = []
    noise_folder = '../../../transformers_mfcc_mask/SPIRA_Dataset_V2/Ruidos-Hospitalares_V1/Ruidos-Hospitalares/Ruidos/Hospitalares-validados/'

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


    ##############################################################################################

    ##############################################################################################

    #run training
    args = {}
    args['sample_rate']= 32000
    args['window_size']= 1024
    args['hop_size']=320
    args['mel_bins']=64
    args['fmin']=0
    args['fmax']=32000
    args['model_type']=model_type#"Transfer_Cnn10"
    args['pretrained_checkpoint_path']=model_load_path#"Cnn10_mAP=0.380.pth"
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

    print('Load pretrained model successfully!', file=output_file)



    d_model = 512
    model_opt = NoamOpt(d_model, 1, 400,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    avg_loss=0
    best_val_acc = 0
    best_val_f1_score = 0
    model_path = model_save_path
    min_frequency = 0.0
    max_frequency = None

    for epoch in range(100):
        model.train()
        loss, avg_loss, _, _, _, _, _, _, _, _, _, _ = run_epoch(model, 
                  LossCompute(model, model_opt, pretrain),
                  data_paths_train, noise_file_paths, noise_file_paths_segunda_fase, pretrain=pretrain, training=True, output_file=output_file, avg_loss=avg_loss,
                  min_frequency=min_frequency, max_frequency=max_frequency, number_coeffs=64)
        model.eval()
        with torch.no_grad():
            loss, _, val_acc, val_f1_score, true_val_f1_score, true_val_acc_score, _, _, _, _, _, _ = run_epoch(model, 
                    LossCompute(model, None, pretrain=pretrain),
                    data_paths_valid, noise_file_paths, noise_file_paths_segunda_fase, pretrain=pretrain, training=False, output_file=output_file,
                    min_frequency=min_frequency, max_frequency=max_frequency, number_coeffs=64)
            print('true_val_acc_score=', true_val_acc_score)
            print('true_val_f1_score=', true_val_f1_score)
        #if epoch%10 == 0:
            print('Epoch=', epoch)
        if best_val_acc < true_val_acc_score:
            best_val_acc = true_val_acc_score
            #if best_val_f1_score < true_val_f1_score:
            #best_val_f1_score = true_val_f1_score
            print('Saving model')
            torch.save({
                'model_state_dict': model.state_dict()
                }, model_path)
            #torch.save(model, model_path)



    #model_path = 'model_test_mel.ckpt'

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
        #print(run_epoch(model,
        #                    LossCompute(model, None, pretrain=pretrain),
        #                    data_paths_train, pretrain=pretrain, training=False, number_coeffs=64))
        loss, _, valid_acc, valid_f1_score, true_valid_f1_score, true_valid_acc_score, _, _, _, _, valid_outputs, valid_targets = run_epoch(model,
                        LossCompute(model, None, pretrain=pretrain),
                        data_paths_valid, noise_file_paths, noise_file_paths_segunda_fase, pretrain=pretrain, training=False, output_file=output_file, number_coeffs=64)
        print('true_valid_f1_score=', true_valid_f1_score, file=output_file)
        print('true_valid_acc_score=', true_valid_acc_score, file=output_file)    
        loss, _, test_acc, test_f1_score, true_test_f1_score, true_test_acc_score, test_confusion_matrix, test_fpr, test_tpr, test_roc_auc, test_outputs, test_targets = run_epoch(model, 
                        LossCompute(model, None, pretrain=pretrain),
                        data_paths_test, noise_file_paths, noise_file_paths_segunda_fase, pretrain=pretrain, training=False, output_file=output_file, number_coeffs=64)
        print('true_test_f1_score=', true_test_f1_score, file=output_file)
        print('true_test_acc_score=', true_test_acc_score, file=output_file)
        tn, fp, fn, tp = test_confusion_matrix.ravel()
        print("test_tp=", tp, file=output_file)
        print("test_tn=", tn, file=output_file)
        print("test_fn=", fn, file=output_file)
        print("test_fp=", fp, file=output_file)
        print("test_fpr=", test_fpr, file=output_file)
        print("test_tpr=", test_tpr, file=output_file)
        print("test_roc_auc=", test_roc_auc, file=output_file)
        #loss, _, clean_acc, clean_f1_score, true_clean_f1_score, true_clean_acc_score, clean_outputs, clean_targets = run_epoch(model, 
        #                    LossCompute(model, None, pretrain=pretrain),
        #                    data_paths_clean, pretrain=pretrain, training=False, number_coeffs=64)
        #print('true_clean_f1_score=', true_clean_f1_score)
        #print('true_clean_acc_score=', true_clean_acc_score)
        #loss, _, _, first_phase_acc, first_phase_f1_score, first_phase_acc_score, _, first_phase_outputs, first_phase_targets = run_epoch(model, 
        #            LossCompute(model, None, pretrain=pretrain),
        #            data_paths_first_phase_test, noise_file_paths, noise_file_paths_segunda_fase, training=False, pretrain=pretrain, number_coeffs=64)
        #print('first_phase_f1_score=', first_phase_f1_score, file=output_file)
        #print('first_phase_acc_score=', first_phase_acc_score, file=output_file)

