# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import sys
sys.path.insert(1, './')

import os
import cv2
import time
import random
import argparse

import numpy as np
import tensorflow as tf

from queue import Queue
from core.WideResNet import *

from core.Define import *
from utils.Utils import *
from utils.Teacher_with_UDA import *
from utils.tensorflow_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='MixMatch', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', default='0', type=str)
    parser.add_argument('--labels', dest='labels', help='labels', default='all', type=str)
    parser.add_argument('--lr_decay', dest='lr_decay', help='decay', default='none', type = str)
    return parser.parse_args()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu

    lr_decay = args.lr_decay # ['none', 'cosine']

    model_dir = './experiments/model/WRN-28-2_1.5M_{}_with_UDA/'.format(args.labels)
    ckpt_format = model_dir + '{}.ckpt'
    log_txt_path = model_dir + 'log.txt'
    summary_txt_path = model_dir + 'model_summary.txt'
    
    tensorboard_path = './experiments/tensorboard/WRN-28-2_1.5M_{}_with_UDA'.format(args.labels)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    
    open(log_txt_path, 'w').close()

    log_print('# Use GPU : {}'.format(args.use_gpu), log_txt_path)
    log_print('# Labels : {}'.format(args.labels), log_txt_path)
    log_print('# batch size : {}'.format(BATCH_SIZE), log_txt_path)
    log_print('# max_iteration : {}'.format(MAX_ITERATION), log_txt_path)

    # 1. dataset
    labels = args.labels
    train_data_dic = np.load('./dataset/train_{}.npy'.format(labels), allow_pickle = True)

    # 1.1 get labeled, unlabeled dataset
    labeled_data_list = np.asarray(train_data_dic.item().get('labeled'))
    unlabeled_data_list = np.asarray(train_data_dic.item().get('unlabeled'))

    labeled_indexs = np.arange(len(labeled_data_list)).tolist()
    unlabeled_indexs = np.arange(len(unlabeled_data_list)).tolist()

    test_data_list = np.load('./dataset/test.npy', allow_pickle = True)
    test_iteration = len(test_data_list) // BATCH_SIZE

    # 2. model

    # 2.1. init placeholder
    shape = [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL]

    sup_image_var = tf.placeholder(tf.float32, [None] + shape, name = 'image/labeled')
    sup_label_var = tf.placeholder(tf.float32, [None, CLASSES])
    
    unsup_image_var = tf.placeholder(tf.float32, [None] + shape, name = 'image/unlabeled')
    unsup_image_with_augment_var = tf.placeholder(tf.float32, [None] + shape, name = 'image/unlabeled_with_augment')

    is_training = tf.placeholder(tf.bool)

    # 2.2. supervised model
    sup_logits_op, sup_predictions_op = WideResNet(sup_image_var, is_training)

    # 2.3. unsupervised model
    p_logits_op = tf.concat([WideResNet(u, is_training)[0] for u in tf.split(unsup_image_var, UNSUP_RATIO)], axis = 0)
    q_logits_op = tf.concat([WideResNet(u, is_training)[0] for u in tf.split(unsup_image_with_augment_var, UNSUP_RATIO)], axis = 0)

    # 2.4. calculate supervised/unsupervised loss
    sup_loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits = sup_logits_op, labels = sup_label_var)
    sup_loss_op = tf.reduce_mean(sup_loss_op)

    unsup_loss_op = KL_Divergence_with_logits(tf.stop_gradient(p_logits_op), q_logits_op)
    unsup_loss_op = tf.reduce_mean(unsup_loss_op)

    # 2.5. l2 regularization loss
    train_vars = tf.get_collection('trainable_variables', 'WideResNet')
    l2_vars = [var for var in train_vars if 'kernel' in var.name or 'weights' in var.name]
    l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in l2_vars]) * WEIGHT_DECAY
    
    # 2.6. total loss
    loss_op = sup_loss_op + unsup_loss_op + l2_reg_loss_op

    correct_op = tf.equal(tf.argmax(sup_predictions_op, axis = -1), tf.argmax(sup_label_var, axis = -1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100

    model_summary(train_vars, summary_txt_path)

    # 3. optimizer & tensorboard
    learning_rate_var = tf.placeholder(tf.float32)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = tf.train.MomentumOptimizer(learning_rate_var, momentum = 0.9, use_nesterov = True).minimize(loss_op)

    train_summary_dic = {
        'Loss/Total_Loss' : loss_op,
        'Loss/Supervised_Loss' : sup_loss_op,
        'Loss/Unsupervised_Loss' : unsup_loss_op,
        'Loss/L2_Regularization_Loss' : l2_reg_loss_op,                                                                                                                                                                                                                                                     
        'Accuracy/Train' : accuracy_op,
        'Learning_rate' : learning_rate_var,
    }

    train_summary_list = []
    for name in train_summary_dic.keys():
        value = train_summary_dic[name]
        train_summary_list.append(tf.summary.scalar(name, value))
    train_summary_op = tf.summary.merge(train_summary_list)

    valid_accuracy_var = tf.placeholder(tf.float32)
    valid_accuracy_op = tf.summary.scalar('Accuracy/Validation', valid_accuracy_var)

    # 4. train loop
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(tensorboard_path)

    # 5. thread
    best_valid_accuracy = 0.0
    
    train_threads = []
    main_queue = Queue(100 * NUM_THREADS)

    for i in range(NUM_THREADS):
        log_print('# create thread : {}'.format(i), log_txt_path)

        train_thread = Teacher(labeled_data_list, unlabeled_data_list, BATCH_SIZE, BATCH_SIZE * UNSUP_RATIO, main_queue)
        train_thread.start()
        
        train_threads.append(train_thread)

    train_ops = [train_op, loss_op, sup_loss_op, unsup_loss_op, l2_reg_loss_op, accuracy_op, train_summary_op]
    
    loss_list = []
    sup_loss_list = []
    unsup_loss_list = []
    l2_reg_loss_list = []
    accuracy_list = []
    train_time = time.time()
    
    augment = RandAugment()
    learning_rate_list = cosine_learning_schedule(0, WARMUP_LEARNING_RATE, MIN_LEARNING_RATE, WARMUP_ITERATION, MAX_ITERATION, MIN_LEARNING_RATE)

    for iter in range(1, MAX_ITERATION + 1):
        # get batch data with Thread
        batch_sup_image_list, batch_sup_label_list, batch_unsup_image_list, batch_unsup_image_with_augment_list = main_queue.get()

        _feed_dict = {
            sup_image_var : batch_sup_image_list, 
            sup_label_var : batch_sup_label_list, 
            unsup_image_var : batch_unsup_image_list, 
            unsup_image_with_augment_var : batch_unsup_image_with_augment_list, 
            is_training : True,
            learning_rate_var : learning_rate_list[iter - 1]
        }
        
        _, loss, sup_loss, unsup_loss, l2_reg_loss, accuracy, summary = sess.run(train_ops, feed_dict = _feed_dict)
        train_writer.add_summary(summary, iter)

        loss_list.append(loss)
        sup_loss_list.append(sup_loss)
        unsup_loss_list.append(unsup_loss)
        l2_reg_loss_list.append(l2_reg_loss)
        accuracy_list.append(accuracy)
        
        if iter % LOG_ITERATION == 0:
            loss = np.mean(loss_list)
            sup_loss = np.mean(sup_loss_list)
            unsup_loss = np.mean(unsup_loss_list)
            l2_reg_loss = np.mean(l2_reg_loss_list)
            accuracy = np.mean(accuracy_list)
            train_time = int(time.time() - train_time)
            
            log_print('[i] iter = {}, loss = {:.4f}, sup_loss = {:.4f}, unsup_loss = {:.4f}, l2_reg_loss = {:.4f}, accuracy = {:.2f}, train_time = {}sec'.format(iter, loss, sup_loss, unsup_loss, l2_reg_loss, accuracy, train_time), log_txt_path)
            
            loss_list = []
            sup_loss_list = []
            unsup_loss_list = []
            l2_reg_loss_list = []
            accuracy_list = []
            train_time = time.time()

        if iter % SAVE_ITERATION == 0:
            valid_time = time.time()
            valid_accuracy_list = []
            
            for i in range(test_iteration):
                batch_data_list = test_data_list[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

                batch_image_data = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL), dtype = np.float32)
                batch_label_data = np.zeros((BATCH_SIZE, CLASSES), dtype = np.float32)
                
                for i, (image, label) in enumerate(batch_data_list):
                    batch_image_data[i] = image.astype(np.float32)
                    batch_label_data[i] = label.astype(np.float32)

                batch_image_data /= 255.
                batch_image_data = (batch_image_data - augment.mean) / augment.std
                
                _feed_dict = {
                    sup_image_var : batch_image_data,
                    sup_label_var : batch_label_data,
                    is_training : False
                }

                accuracy = sess.run(accuracy_op, feed_dict = _feed_dict)
                valid_accuracy_list.append(accuracy)

            valid_time = int(time.time() - valid_time)
            valid_accuracy = np.mean(valid_accuracy_list)

            summary = sess.run(valid_accuracy_op, feed_dict = {valid_accuracy_var : valid_accuracy})
            train_writer.add_summary(summary, iter)

            if best_valid_accuracy <= valid_accuracy:
                best_valid_accuracy = valid_accuracy
                saver.save(sess, ckpt_format.format(iter))            

            log_print('[i] iter = {}, valid_accuracy = {:.2f}, best_valid_accuracy = {:.2f}, valid_time = {}sec'.format(iter, valid_accuracy, best_valid_accuracy, valid_time), log_txt_path)
    
    for th in train_threads:
        th.train = False
        th.join()

if __name__ == '__main__':
    args = parse_args()
    main(args)
