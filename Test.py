import cv2
import numpy as np

from core.randaugment.augment import *

images = np.load('./dataset/cifar10@4000/unlabeled_1.npy', allow_pickle = True)

augment = RandAugment()

for u, ua in images:
    _ui = cv2.resize(u, (112, 112))
    _uai = cv2.resize(ua, (112, 112))
    
    cv2.imshow('show', _ui)
    cv2.imshow('show with augment', _uai)
    cv2.waitKey(0)

