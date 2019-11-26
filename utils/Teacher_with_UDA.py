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

            self.watch.tik()
            
            batch_sup_image_list = []
            batch_sup_label_list = []
            batch_unsup_image_list = []
            batch_unsup_image_with_augment_list = []

            np.random.shuffle(self.labeled_data_list)
            np.random.shuffle(self.unlabeled_data_list)

            for data in self.labeled_data_list[:self.sup_batch_size]:
                image, label = data
                # image = self.augment(image.copy())

                batch_sup_image_list.append(image.copy())
                batch_sup_label_list.append(label)
            
            for image in self.unlabeled_data_list[:self.unsup_batch_size]:
                batch_unsup_image_list.append(image.copy())
                batch_unsup_image_with_augment_list.append(self.augment(image.copy()))
            
            batch_sup_image_list = np.asarray(batch_sup_image_list, dtype = np.float32)
            batch_sup_label_list = np.asarray(batch_sup_label_list, dtype = np.float32)
            batch_unsup_image_list = np.asarray(batch_unsup_image_list, dtype = np.float32)
            batch_unsup_image_with_augment_list = np.asarray(batch_unsup_image_with_augment_list, dtype = np.float32)
            
            # normalize
            batch_sup_image_list /= 255.
            batch_sup_image_list = (batch_sup_image_list - self.augment.mean) / self.augment.std

            batch_unsup_image_list /= 255.
            batch_unsup_image_list = (batch_unsup_image_list - self.augment.mean) / self.augment.std

            # print(batch_sup_image_list.min(), batch_sup_image_list.max())
            # print(batch_unsup_image_list.min(), batch_unsup_image_list.max())
            # print(batch_unsup_image_with_augment_list.min(), batch_unsup_image_with_augment_list.max())
            
            self.main_queue.put([batch_sup_image_list, batch_sup_label_list, batch_unsup_image_list, batch_unsup_image_with_augment_list])

            # print('[{}] - {}ms'.format(self.main_queue.qsize(), self.watch.tok()))

