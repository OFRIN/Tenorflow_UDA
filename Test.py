import cv2
import numpy as np

from core.randaugment.augment import *

uimages = np.load('./dataset/cifar10@4000/unlabeled.npy', allow_pickle = True)
uaimages = np.load('./dataset/cifar10@4000/unlabeled_1.npy', allow_pickle = True)

augment = RandAugment()

for _ui, _uai in zip(uimages, uaimages):
    _uni = cv2.resize(augment(_ui), (112, 112))

    _ui = cv2.resize(_ui, (112, 112))
    _uai = cv2.resize(_uai, (112, 112))
    
    cv2.imshow('show', _ui)
    cv2.imshow('uni', _uni)
    cv2.imshow('show with augment', _uai)
    cv2.waitKey(0)

