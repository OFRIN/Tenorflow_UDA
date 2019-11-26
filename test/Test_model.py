import sys
sys.path.insert(1, './')

import cv2
import numpy as np

from core.Define import *
from core.WideResNet import *

from utils.tensorflow_utils import *

input_var = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
is_training = tf.placeholder(tf.bool)

logits, predictions = WideResNet(input_var, is_training, filters = 32, repeat = 4)

vars = tf.trainable_variables()
model_summary(vars, './wider_resnet_28_large.txt')
