This code is a fork from https://github.com/qiuqiangkong/audioset_tagging_cnn. We only make the relevant changes to enable the models to be finetuned on the classification and regression tasks we consider.

# Folder structure

The PANNs_classification folder contains all files used for the binary classfication tasks. These are respiratory insufficiency detection and high/low SpO2 threshold classification (with a threshold usually set at 92%). The PANNs_regression folder contains files for performing an SpO2 regression task. The regression task results were not included in the paper.

# How to run the models
For respiratory insufficiency detection one should run the files named run_finetuning_main_xxx.py. The parameters involve the number of times one wishes to repeat the experiment, the model type (Transfer_CNN6 for CNN6, Transfer_CNN10 for CNN10 and Transfer_CNN14 for CNN14) and the pretrained model's path (change to the respective path and make sure that model type and pretrained model are consistent - e.g. pretrained CNN6 with model type Transfer_CNN6). It will also be necessary to change the underlying data paths in run_finetuning to the correct paths with the right dataset.

The run_finetuning_roc_curve file is only to generate an experiment with the necessary information to build a ROC curve. If you do not need to build a ROC curve the original main can be used.

For high/low SpO2 classification one should run CNN6_classification.py. Despite the name, it contains the code for CNN6, CNN10 and CNN14.  It is necessary to change the internal parameters to the corresponding model and pretrained model path required. It may also be necessary to replace the noise_file_path to the correct path, as well as the data_paths for training, validation and testing.

For SpO2 regression just run the file PANNs_predict_spo2_standard.py. It may be necessary to replace the noise_file_path to the correct path, as well as the data_paths for training, validation and testing. It is necessary to change the internal parameters to the corresponding model and pretrained model path required. 
