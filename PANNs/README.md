This code is a fork from https://github.com/qiuqiangkong/audioset_tagging_cnn. We only make the relevant changes to enable the models to be finetuned on the classification and regression tasks we consider.

Files under PANNs_classification are for the binary classfication tasks. Files under PANNs_regression are for the regression task.

For respiratory insufficiency detection one should run the files named run_finetuning_main_xxx.py provided the parameters inside and in the called files are changed to the ones you desire. The roc_curve file is only to generate
an experiment with the necessary information to build a ROC curve. If you do not need to build a ROC curve the original main can be used.
For high/low SpO2 classification one should run CNN6_classification.py. Despite the name, it contains the code for CNN6, CNN10 and CNN14. 
For SpO2 regression just run the file PANNs_predict_spo2_standard.py.

The pathways to data files and pretrained models will need to be changed as will whatever parameters you desire to change. This holds for all the codes in this folder
