import os
import sys
import time
import argparse

import numpy as np

from core.randaugment.augment import *

from utils.Utils import *
from utils.StopWatch import *

def parse_args():
    parser = argparse.ArgumentParser(description='Unsupervised Data Augmentations', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--labels', dest='labels', help='labels', default='all', type=str)
    parser.add_argument('--augment_copy', dest='augment_copy', help='augment_copy', default=100, type=int)
    return parser.parse_args()

args = vars(parse_args())

dataset_dir = './dataset/cifar10@{}/'.format(args['labels'])
if not os.path.isdir(dataset_dir):
    os.makedirs(dataset_dir)

# load batch files.
labeled_data, unlabeled_data, test_dataset = get_dataset('./cifar10/', int(args['labels']))
unlabeled_data = np.asarray(unlabeled_data, dtype = np.uint8)

# labeled dataset
images = []
labels = []

for image, label in labeled_data:
    images.append(image)
    labels.append(label)

np.save(dataset_dir + 'labeled.npy', {
    'images' : np.asarray(images, dtype = np.uint8),
    'labels' : np.asarray(labels, dtype = np.uint8),
})

# unlabeled dataset
watch = StopWatch()
augment = RandAugment()

for i in range(args['augment_copy']):
    watch.tik()

    images = [augment(image) for image in unlabeled_data]
    np.save(dataset_dir + 'unlabeled_{}.npy'.format(i + 1), images)
    
    print('# {} - {}sec'.format(i + 1, watch.tok() / 1000))

