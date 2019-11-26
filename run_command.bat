
## 1. fully supervised learning
python train/Train_Supervised.py --use_gpu 0 --labels all
python train/Train_Supervised.py --use_gpu 0 --labels 4000

## 2. UDA (labels = 4000)
python train/Train_Semi_Supervised.py --use_gpu 0 --labels 4000

