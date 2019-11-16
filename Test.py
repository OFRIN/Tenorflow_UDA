# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import time
import argparse

import numpy as np
import tensorflow as tf

from MixMatch import *
from WideResNet import *

from Define import *
from Utils import *
from DataAugmentation import *

def parse_args():
    parser = argparse.ArgumentParser(description='A Neural Algorithm of Artistic Style', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', default='0', type=str)
    parser.add_argument('--labels', dest='labels', help='labels', default=4000, type=int)
    parser.add_argument('--learning_rate', dest='learning_rate', help='learning_rate', default=INIT_LEARNING_RATE, type=float)
    args = parser.parse_args()
    return args

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu

print('# Use GPU : {}'.format(args.use_gpu))
print('# Labels : {}'.format(args.labels))
print('# Learning_Rate : {}'.format(args.learning_rate))

# 1. dataset
labels = args.labels
train_data_dic = np.load('./dataset/train_{}.npy'.format(labels), allow_pickle = True)

# 1.1 get labeled, unlabeled dataset
labeled_data_list = train_data_dic.item().get('labeled')
unlabeled_data_list = train_data_dic.item().get('unlabeled')

shape = [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL]

x_var = tf.placeholder(tf.float32, [None] + shape, name = 'image/labeled')
x_label_var = tf.placeholder(tf.float32, [None, 10], name = 'label')

logits_op, predictions_op = WideResNet(x_var, True, reuse = False)

loss_x_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits_op, labels = x_label_var)
loss_x_op = tf.reduce_mean(loss_x_op)

correct_op = tf.equal(tf.argmax(predictions_op, axis = -1), tf.argmax(x_label_var, axis = -1))
accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './model/MixMatch_4000.ckpt')

for i in range(10):
    np.random.shuffle(labeled_data_list)

    print(i * BATCH_SIZE, (i + 1) * BATCH_SIZE)
    print(len(labeled_data_list))

    batch_image_data = []
    batch_label_data = []

    for data in labeled_data_list[:BATCH_SIZE]:
        image, label = data
        batch_image_data.append(image)
        batch_label_data.append(label)

    loss, accuracy = sess.run([loss_x_op, accuracy_op], feed_dict = {x_var : batch_image_data, x_label_var : batch_label_data})
    print(loss, accuracy)
    input()
    
    # _index = np.argmax(class_prob)
    # _prob = class_prob[_index]

    # print(_index, CLASS_NAMES[_index], _prob * 100)

    # show_image = cv2.resize(image, (224, 224))
    # cv2.imshow('show', show_image)
    # cv2.waitKey(0)
