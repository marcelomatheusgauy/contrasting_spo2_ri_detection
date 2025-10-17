This code is a fork from https://github.com/facebookresearch/AudioMAE. We only make the relevant changes to enable the models to be finetuned on the classification and regression tasks we consider.

# Folder structure

It is probably unnecessary to make changes to files present in any of the subfolders, with the exception of experiments/configs/ which contains the configuration files. The audiomae folder contains the original audiomae code. The folders classification_ri, spo2_classification, spo2_regression contain slight adaptations to a few utility functions of audiomae to enable it to run on the SPIRA dataset. classification_ri deals with the respiratory insufficiency detection task, spo2_classification deals with the high/low SpO2 threshold classification task and spo2_regression is used for the spo2 regression task that was not included in the paper. 

# How to run the models

The file finetune_ri_classification.py is the one which should be run for finetuning Audio-MAE for respiratory insufficiency detection. 
The file spo2_classification_finetune.py is the one which should be run for finetuning Audio-MAE for the high/low SpO2 threshold classification task.
The file spo2_regression_finetune.py is the one which should be run for finetuning Audio-MAE for SpO2 regression.

In order to run one of the files a command of the form CUDA_VISIBLE_DEVICES=0 python3 (one of the filenames above).py --config_path=(here add the path to the configuration file). The configuration files can be found in the respective folders (described above) at experiments/configs. The config files contain all necessary parameters for Audio-MAE. It will probably be necessary to change the respective data_paths from the configuration file. Other parameters are unlikely to require changes, unless one wants to search for hyperparameter configurations of Audio-MAE.

You will need to generate csvs in the right format to run the code (or make changes to the respective utility files).  The simplest way is to make a csv, which contains each audio file, a  file_path entry and an insuficiencia_respiratoria entry (corresponding to whether a patient suffers from respiratory insufficiency - so that no issues occur with the current version of the code this has to be either 0 or 1 - do not use True/False).
