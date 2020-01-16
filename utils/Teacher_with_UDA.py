# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import copy
import time
import random

import numpy as np

from threading import Thread

from core.randaugment.augment import *

from utils.Utils import *
from utils.StopWatch import *

def get_data_list(npy_paths):
    # st = time.time()

    data_list = []
    for npy_path in npy_paths:
        data_list += [data for data in np.load(npy_path, allow_pickle = True)]
    
    # end = int(time.time() - st)
    # print('get_data_list : {}sec'.format(end)) # 0sec
    
    return data_list

class Teacher(Thread):
    
    def __init__(self, labeled_data_list, unlabeled_paths, option, main_queue):
        Thread.__init__(self)

        self.train = True
        self.watch = StopWatch()
        self.main_queue = main_queue
        
        self.sup_batch_size = option['sup_batch_size']
        self.unsup_batch_size = option['unsup_batch_size']
        
        self.option = option
        self.unlabeled_paths = unlabeled_paths

        self.labeled_data_list = labeled_data_list
        self.unlabeled_data_list = get_data_list(random.sample(self.unlabeled_paths, self.option['load_npy_count']))
        self.unlabeled_data_list = random.sample(self.unlabeled_data_list, self.option['unsup_samples'] * self.option['unsup_batch_size'])

        self.labeled_index = 0
        self.unlabeled_index = 0
        
        self.labeled_iteration = len(self.labeled_data_list) // self.sup_batch_size
        self.unlabeled_iteration = len(self.unlabeled_data_list) // self.unsup_batch_size

        np.random.shuffle(self.labeled_data_list)
        np.random.shuffle(self.unlabeled_data_list)
        
        self.augment = RandAugment()
    
    def run(self):
        while self.train:
            while self.main_queue.full() and self.train:
                time.sleep(0.1)
                continue
            
            batch_x_image_data = []
            batch_x_label_data = []
            batch_u_image_data = []
            batch_ua_image_data = []

            for data in self.labeled_data_list[self.labeled_index * self.sup_batch_size : (self.labeled_index + 1) * self.sup_batch_size]:
                image, label = data
                # image = self.augment(image.copy())

                batch_x_image_data.append(image.copy())
                batch_x_label_data.append(label)
            
            for u_image, ua_image in self.unlabeled_data_list[self.unlabeled_index * self.unsup_batch_size : (self.unlabeled_index + 1) * self.unsup_batch_size]:
                batch_u_image_data.append(u_image)
                batch_ua_image_data.append(ua_image)
                
            batch_x_image_data = np.asarray(batch_x_image_data, dtype = np.float32)
            batch_x_label_data = np.asarray(batch_x_label_data, dtype = np.float32)
            batch_u_image_data = np.asarray(batch_u_image_data, dtype = np.float32)
            batch_ua_image_data = np.asarray(batch_ua_image_data, dtype = np.float32)

            self.labeled_index += 1
            self.unlabeled_index += 1

            if self.labeled_index == self.labeled_iteration:
                self.labeled_index = 0
                np.random.shuffle(self.labeled_data_list)

            if self.unlabeled_index == self.unlabeled_iteration:
                self.unlabeled_data_list = get_data_list(random.sample(self.unlabeled_paths, self.option['load_npy_count']))
                self.unlabeled_iteration = len(self.unlabeled_data_list) // self.unsup_batch_size

                self.unlabeled_index = 0
                np.random.shuffle(self.unlabeled_data_list)
            
            self.main_queue.put([
                batch_x_image_data, 
                batch_x_label_data, 
                batch_u_image_data, 
                batch_ua_image_data
            ])
