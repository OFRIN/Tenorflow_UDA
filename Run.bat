
## 1. preprocess dataset
python Generate_Dataset.py --labels 250 --augment_copy 100
python Generate_Dataset.py --labels 4000 --augment_copy 100

## 2. run fully supervised learning
python Train_Supervised.py --use_gpu 0 --labels all
python Train_Supervised.py --use_gpu 0 --labels 4000

## 2. run unsupervised data augmentations
python Train_Semi_Supervised.py --use_gpu 1 --labels 4000
python Train_Semi_Supervised.py --use_gpu 0 --labels 4000 --tsa linear_schedule --softmax-temp 0.5 --confidence-mask 0.6
python Train_Semi_Supervised.py --use_gpu 0 --labels 250 --softmax-temp 0.5 --confidence-mask 0.6

