# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import copy
import time
import random

import numpy as np

from threading import Thread

from core.Define import *
from core.randaugment.augment import *

from utils.Utils import *
from utils.StopWatch import *

class Teacher(Thread):
    
    def __init__(self, labeled_data_list, unlabeled_data_list, sup_batch_size, unsup_batch_size, main_queue):
        Thread.__init__(self)

        self.train = True
        self.watch = StopWatch()
        self.main_queue = main_queue
        
        self.sup_batch_size = sup_batch_size
        self.unsup_batch_size = unsup_batch_size

        self.labeled_data_list = copy.deepcopy(labeled_data_list)
        self.unlabeled_data_list = copy.deepcopy(unlabeled_data_list)

        self.augment = RandAugment()
        
    def run(self):
        while self.train:
            while self.main_queue.full() and self.train:
                time.sleep(0.1)
                continue

            # self.watch.tik()
            
            batch_x_image_data = []
            batch_x_label_data = []
            batch_u_image_data = []
            batch_ua_image_data = []

            np.random.shuffle(self.labeled_data_list)
            np.random.shuffle(self.unlabeled_data_list)
            
            for data in self.labeled_data_list[:self.sup_batch_size]:
                image, label = data
                # image = self.augment(image.copy())

                batch_x_image_data.append(image.copy())
                batch_x_label_data.append(label)
            
            for u_image in self.unlabeled_data_list[:self.unsup_batch_size]:
                ua_image = self.augment(u_image.copy())
                batch_u_image_data.append(u_image)
                batch_ua_image_data.append(ua_image)

                # batch_ua_image_data.append(self.augment(u_image.copy()))
            
            batch_x_image_data = np.asarray(batch_x_image_data, dtype = np.float32)
            batch_x_label_data = np.asarray(batch_x_label_data, dtype = np.float32)
            batch_u_image_data = np.asarray(batch_u_image_data, dtype = np.float32)
            batch_ua_image_data = np.asarray(batch_ua_image_data, dtype = np.float32)
            
            self.main_queue.put([
                batch_x_image_data, 
                batch_x_label_data, 
                batch_u_image_data, 
                batch_ua_image_data
            ])
