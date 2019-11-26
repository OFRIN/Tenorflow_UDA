import sys
sys.path.insert(1, './')

import cv2
import numpy as np

from core.Define import *
from core.WideResNet import *
from core.randaugment.augment import *

from utils.tensorflow_utils import *

augment = RandAugment()
dataset = np.load('./dataset/train.npy', allow_pickle = True)

for data in dataset:
    image, label = data
    rand_image = augment(image.copy())

    image = cv2.resize(image, (112, 112))

    rand_image = (rand_image * augment.std) + augment.mean
    rand_image = (rand_image * 255.).astype(np.uint8)
    rand_image = cv2.resize(rand_image, (112, 112))

    cv2.imshow('show', image)
    cv2.imshow('randaugment', rand_image)
    cv2.waitKey(0)
