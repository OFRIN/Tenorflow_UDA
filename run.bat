python Train.py --use_gpu 0 --labels 4000 --su_ratios 1
python Train.py --use_gpu 1 --labels 4000 --su_ratios 5
python Train_with_cosine.py --use_gpu 2 --labels 4000 --su_ratios 5
python Train_without_UDA.py --use_gpu 3 --labels 4000

python Train.py --use_gpu 1 --labels 4000
python Train.py --use_gpu 2 --labels 2000
python Train.py --use_gpu 3 --labels 1000
python Train.py --use_gpu 3 --labels 500
python Train.py --use_gpu 3 --labels 250
