# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import time
import random
import threading

import numpy as np

import AutoAugment.AutoAugment as augment

from WideResNet import *
from Define import *
from Utils import *
from UDA_Loss import *

class Teacher(threading.Thread):
    ready = False
    min_data_size = 0
    max_data_size = 5

    total_indexs = []
    total_data_list = []
    
    ratio = 0

    labeled_data_list = []
    labeled_indexs = []

    unlabeled_data_list = []
    unlabeled_indexs = []

    batch_data_list = []
    batch_data_length = 0

    end = False
    debug = False
    name = ''
    
    def __init__(self, labeled_data_list, unlabeled_data_list, ratio, min_data_size = 1, max_data_size = 5, name = 'Thread', debug = False):
        self.name = name
        self.debug = debug
        
        self.min_data_size = min_data_size
        self.max_data_size = max_data_size

        self.labeled_data_list = np.asarray(labeled_data_list)
        self.labeled_indexs = np.arange(len(self.labeled_data_list)).tolist()

        self.unlabeled_data_list = np.asarray(unlabeled_data_list)
        self.unlabeled_indexs = np.arange(len(self.unlabeled_data_list)).tolist()
        
        self.ratio = ratio
        self.end = False
        threading.Thread.__init__(self)
        
    def get_batch_data(self):
        batch_data = self.batch_data_list[0]
        
        del self.batch_data_list[0]
        self.batch_data_length -= 1

        if self.batch_data_length < self.min_data_size:
            self.ready = False
        
        return batch_data
    
    def run(self):
        while not self.end:
            while self.batch_data_length >= self.max_data_size:
                continue
            
            batch_s_image_list = []
            batch_s_label_list = []
            batch_u_image_list = []
            batch_u_augment_image_list = []

            batch_labeled_indexs = random.sample(self.labeled_indexs, BATCH_SIZE)
            batch_unlabeled_indexs = random.sample(self.unlabeled_indexs, BATCH_SIZE * self.ratio)
            
            for s_data in self.labeled_data_list[batch_labeled_indexs]:
                image, label = s_data
                image = augment.AutoAugment(image, normalize = False)

                batch_s_image_list.append(image)
                batch_s_label_list.append(label)
            
            for u_data in self.unlabeled_data_list[batch_unlabeled_indexs]:
                batch_u_image_list.append(u_data)
                batch_u_augment_image_list.append(augment.AutoAugment(u_data, normalize = False))
            
            batch_s_image_list = np.asarray(batch_s_image_list, dtype = np.float32)
            batch_s_label_list = np.asarray(batch_s_label_list, dtype = np.float32)
            batch_u_image_list = np.asarray(batch_u_image_list, dtype = np.float32)
            batch_u_augment_image_list = np.asarray(batch_u_augment_image_list, dtype = np.float32)        
            
            self.batch_data_list.append([
                batch_s_image_list, 
                batch_s_label_list,
                batch_u_image_list,
                batch_u_augment_image_list
            ])
            self.batch_data_length += 1

            # print(self.batch_data_length, self.min_data_size)

            if self.batch_data_length >= self.min_data_size:
                self.ready = True
            else:
                self.ready = False

