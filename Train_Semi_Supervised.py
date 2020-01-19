# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import sys
import os
import cv2
import time
import random
import argparse

import numpy as np
import tensorflow as tf

from queue import Queue

from core.UDA import *
from core.WideResNet import *

from utils.Utils import *
from utils.Teacher_with_UDA import *
from utils.Tensorflow_Utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='UDA', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # gpu properties
    parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', default='0', type=str)

    # cifar10 dataset
    parser.add_argument('--labels', dest='labels', help='labels', default='all', type=str)

    # training properties
    parser.add_argument('--model_name', dest='model_name', help='model_name', default='WideResNet', type=str)

    parser.add_argument('--batch_size', dest='batch_size', help='batch_size', default=64, type=int)
    parser.add_argument('--num_threads', dest='num_threads', help='num_threads', default=4, type=int)

    parser.add_argument('--log_iteration', dest='log_iteration', help='log_iteration', default=100, type=int)
    parser.add_argument('--save_iteration', dest='save_iteration', help='save_iteration', default=10000, type=int)

    parser.add_argument('--max_iteration', dest='max_iteration', help='max_iteration', default=400000, type=int)
    parser.add_argument('--warmup_iteration', dest='warmup_iteration', help='warmup_iteration', default=20000, type=int)

    parser.add_argument('--warmup_learning_rate', dest='warmup_learning_rate', help='warmup_learning_rate', default=0.03, type=float)
    parser.add_argument('--min_learning_rate', dest='min_learning_rate', help='min_learning_rate', default=0.004, type=float)

    # uda properties
    parser.add_argument('--weight_decay', dest='weight_decay', help='weight_decay', default=0.0005, type=int)

    parser.add_argument('--unsup_ratio', dest='unsup_ratio', default=5, type=int)
    parser.add_argument('--tsa', dest='tsa', default='', type=str)

    parser.add_argument('--softmax_temp', dest='softmax_temp', default=-1, type=float)
    parser.add_argument('--confidence_mask', dest='confidence_mask', default=-1, type=float)

    return parser.parse_args()

##########################################################################################################
# preprocessing
##########################################################################################################
# 1.1.  
args = vars(parse_args())

folder_name = '{}'.format(args['model_name'])
folder_name += '_cifar10@{}'.format(args['labels'])
folder_name += '_unsup_ratio@{}'.format(args['unsup_ratio'])

if args['tsa'] != '': folder_name += 'tsa@{}'.format(args['tsa'])
if args['softmax_temp'] != '': folder_name += 'softmax_temp@{:.1f}'.format(args['softmax_temp'])
if args['confidence_mask'] != '': folder_name += 'confidence_mask@{:.1f}'.format(args['confidence_mask'])

model_dir = './experiments/model/{}/'.format(folder_name)
ckpt_format = model_dir + '{}.ckpt'
log_txt_path = model_dir + 'log.txt'
summary_txt_path = model_dir + 'model_summary.txt'

tensorboard_path = './experiments/tensorboard/{}'.format(folder_name)

# 1.2.
os.environ["CUDA_VISIBLE_DEVICES"] = args['use_gpu']

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

open(log_txt_path, 'w').close()

# 1.3.
data_dir = './dataset/cifar10@{}/'.format(args['labels'])

labeled_path = data_dir + 'labeled.npy'
unlabeled_paths = glob.glob(data_dir + 'unlabel*.npy')

dataset = np.load(labeled_path, allow_pickle = True)
labeled_data_list = [[image, label] for image, label in zip(dataset.item().get('images'), dataset.item().get('labels'))]

log_print('# labeled dataset : {}'.format(len(labeled_data_list)), log_txt_path)
log_print('# unlabeled paths', log_txt_path)
for path in unlabeled_paths:
    log_print('-> {}'.format(path), log_txt_path)

_, test_data_list = get_dataset_fully_supervised('./cifar10/', only_test = True)
test_iteration = len(test_data_list) // args['batch_size']

##########################################################################################################
# preprocessing
##########################################################################################################
# 2.1. define placeholders.
x_image_var = tf.placeholder(tf.float32, [None] + [32, 32, 3])
x_label_var = tf.placeholder(tf.float32, [None, 10])

u_image_var = tf.placeholder(tf.float32, [None] + [32, 32, 3])
ua_image_var = tf.placeholder(tf.float32, [None] + [32, 32, 3])

is_training = tf.placeholder(tf.bool)
global_step = tf.placeholder(tf.int32)

# 2.2. supervised model
x_logits_op, x_predictions_op = WideResNet(x_image_var, is_training)

# 2.3. unsupervised model
p_logits_op = tf.concat([WideResNet(u, is_training)[0] for u in tf.split(u_image_var, args['unsup_ratio'])], axis = 0)
q_logits_op = tf.concat([WideResNet(u, is_training)[0] for u in tf.split(ua_image_var, args['unsup_ratio'])], axis = 0)

# 2.4. calculate supervised/unsupervised loss
sup_loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits = x_logits_op, labels = x_label_var)

# mode_list = ['exp_schedule', 'log_schedule', 'linear_schedule']
if args['tsa'] != '':
    log_print('[i] TSA : {}'.format(args['tsa']), log_txt_path)
    alpha_t, nt = TSA_schedule(global_step, MAX_ITERATION, args['tsa'], CLASSES)

    correct_prob = tf.reduce_sum(x_label_var * x_predictions_op, axis = -1)
    sup_mask = 1. - tf.cast(tf.greater(correct_prob, nt), tf.float32)
    sup_mask = tf.stop_gradient(sup_mask)

    sup_loss_op = sup_mask * sup_loss_op
    sup_loss_op = tf.reduce_sum(sup_loss_op) / tf.maximum(tf.reduce_sum(sup_mask), 1.)
else:
    sup_loss_op = tf.reduce_mean(sup_loss_op)

# with softmax temperature
if args['softmax_temp'] != -1:
    log_print('[i] softmax temperature : {}'.format(args['softmax_temp']), log_txt_path)
    p_logits_temp_op = p_logits_op / args['softmax_temp']
else:
    p_logits_temp_op = p_logits_op

unsup_loss_op = KL_Divergence_with_logits(tf.stop_gradient(p_logits_temp_op), q_logits_op)

# with confidence mask
if args['confidence_mask'] != -1:
    log_print('[i] confidence mask : {}'.format(args['confidence_mask']), log_txt_path)

    unsup_prob = tf.nn.softmax(p_logits_op, axis = -1)
    largest_prob = tf.reduce_max(unsup_prob, axis = -1)

    unsup_mask = tf.cast(tf.greater(largest_prob, args['confidence_mask']), tf.float32)
    unsup_loss_op = tf.stop_gradient(unsup_mask) * unsup_loss_op

unsup_loss_op = args['unsup_ratio'] * tf.reduce_mean(unsup_loss_op)

# 2.5. l2 regularization loss
train_vars = tf.trainable_variables()
l2_vars = train_vars # [var for var in train_vars if 'kernel' in var.name or 'weights' in var.name]
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in l2_vars]) * args['weight_decay']

# 2.6. total loss
loss_op = sup_loss_op + unsup_loss_op + l2_reg_loss_op

correct_op = tf.equal(tf.argmax(x_predictions_op, axis = -1), tf.argmax(x_label_var, axis = -1))
accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100

model_summary(train_vars, summary_txt_path)

# 3. optimizer & tensorboard

# increase learning rate linearly.
warmup_lr = tf.to_float(global_step) / tf.to_float(args['warmup_iteration']) * args['warmup_learning_rate']

# decrease learning rate using cosine decay.
decay_lr = tf.train.cosine_decay(
    args['warmup_learning_rate'],
    global_step = global_step - args['warmup_iteration'],
    decay_steps = args['max_iteration'] - args['warmup_iteration'],
    alpha = args['min_learning_rate']
)

learning_rate = tf.where(global_step < args['warmup_iteration'], warmup_lr, decay_lr)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9, use_nesterov = True).minimize(loss_op)

train_summary_dic = {
    'Loss/Total_Loss' : loss_op,
    'Loss/Supervised_Loss' : sup_loss_op,
    'Loss/Unsupervised_Loss' : unsup_loss_op,
    'Loss/L2_Regularization_Loss' : l2_reg_loss_op,     

    'Accuracy/Train' : accuracy_op,
    
    'HyperParams/Learning_rate' : learning_rate,
}

if args['tsa'] != '':
    train_summary_dic['HyperParams/TSA_{}'.format(args['tsa'])] = nt

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

option = {
    'sup_batch_size' : args['batch_size'],
    'unsup_batch_size' : args['batch_size'] * args['unsup_ratio'],
    
    'load_npy_count' : 5,
    'unsup_samples' : 50,
}

train_threads = []
main_queue = Queue(25 * args['num_threads'])

for i in range(args['num_threads']):
    log_print('# create thread : {}'.format(i), log_txt_path)

    train_thread = Teacher(labeled_data_list, unlabeled_paths, option, main_queue)
    train_thread.start()
    
    train_threads.append(train_thread)

    if i == 0:
        log_print('# supervised batch size : {}'.format(option['sup_batch_size']), log_txt_path)
        log_print('# unsupervised batch size : {}'.format(option['unsup_batch_size']), log_txt_path)

train_ops = [train_op, loss_op, sup_loss_op, unsup_loss_op, l2_reg_loss_op, accuracy_op, train_summary_op]

loss_list = []
sup_loss_list = []
unsup_loss_list = []
l2_reg_loss_list = []
accuracy_list = []
train_time = time.time()

for iter in range(1, args['max_iteration'] + 1):
    batch_x_image_data, batch_x_label_data, batch_u_image_data, batch_ua_image_data = main_queue.get()

    _feed_dict = {
        x_image_var : batch_x_image_data, 
        x_label_var : batch_x_label_data, 
        u_image_var : batch_u_image_data, 
        ua_image_var : batch_ua_image_data, 
        is_training : True,
        global_step : iter,
    }
    
    _, loss, sup_loss, unsup_loss, l2_reg_loss, accuracy, summary = sess.run(train_ops, feed_dict = _feed_dict)
    train_writer.add_summary(summary, iter)

    loss_list.append(loss)
    sup_loss_list.append(sup_loss)
    unsup_loss_list.append(unsup_loss)
    l2_reg_loss_list.append(l2_reg_loss)
    accuracy_list.append(accuracy)
    
    if iter % args['log_iteration'] == 0:
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

    if iter % args['save_iteration'] == 0:
        valid_time = time.time()
        valid_accuracy_list = []
        
        for i in range(args['save_iteration']):
            batch_data_list = test_data_list[i * args['batch_size'] : (i + 1) * args['batch_size']]

            batch_image_data = np.zeros((args['batch_size'], 32, 32, 3), dtype = np.float32)
            batch_label_data = np.zeros((args['batch_size'], 10), dtype = np.float32)
            
            for i, (image, label) in enumerate(batch_data_list):
                batch_image_data[i] = image.astype(np.float32)
                batch_label_data[i] = label.astype(np.float32)

            _feed_dict = {
                x_image_var : batch_image_data,
                x_label_var : batch_label_data,
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