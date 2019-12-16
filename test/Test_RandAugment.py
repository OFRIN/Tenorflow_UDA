import sys
sys.path.insert(1, '../')

import cv2
import numpy as np

from core.Define import *
from core.WideResNet import *
from core.randaugment.augment import *

from utils.Utils import *
from utils.Tensorflow_Utils import *

augment = RandAugment()
labeled_data, unlabeled_image_data, test_dataset = get_dataset('../dataset/', 4000)

for data in labeled_data:
    image, label = data
    rand_image = augment(image.copy())

    image = cv2.resize(image, (112, 112))
    rand_image = cv2.resize(rand_image, (112, 112))

    cv2.imshow('show', image)
    cv2.imshow('randaugment', rand_image)
    cv2.waitKey(0)
