This code is a fork from https://github.com/facebookresearch/AudioMAE. We only make the relevant changes to enable the models to be finetuned on the classification and regression tasks we consider.

The file finetune_ri_classification.py is the one which should be run for finetuning Audio-MAE for respiratory insufficiency detection. 
The file spo2_classification_finetune.py is the one which should be run for finetuning Audio-MAE for the SpO2 high/low binary classification task.
The file spo2_regression_finetune.py is the one which should be run for finetuning Audio-MAE for SpO2 regression.

In order to run one of the files a command of the form CUDA_VISIBLE_DEVICES=0 python3 (one of the filenames above).py --config_path=(here add the path to the configuration file - 
this can be found on the respective folders at experiments/configs and may need to have some paths corrected)

You will need to generate csvs in the right format to run the code (or adapt some of the utility files).  E.g make a csv with file_path and insuficiencia_respiratoria (whether patient suffers 
from respiratory insufficiency - has to be 0 or 1) entries for each audio file you have.
