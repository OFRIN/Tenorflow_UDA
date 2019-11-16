# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import time
import argparse

import numpy as np
import tensorflow as tf

import AutoAugment.AutoAugment as augment

from WideResNet import *
from Define import *
from Teacher import *
from Utils import *

from UDA_Loss import *

def parse_args():
    parser = argparse.ArgumentParser(description='A Neural Algorithm of Artistic Style', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', default='0', type=str)
    parser.add_argument('--labels', dest='labels', help='labels', default=4000, type=int)
    parser.add_argument('--learning_rate', dest='learning_rate', help='learning_rate', default=INIT_LEARNING_RATE, type=float)
    parser.add_argument('--su_ratios', dest='su_ratios', help='su_ratios', default=SU_RATIOS, type=int)
    args = parser.parse_args()
    return args

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu

su_ratios = args.su_ratios
log_txt_path = './log_with_cosine@labels={}@SU_ratios={}.txt'.format(args.labels, su_ratios)

log_print('# Use GPU : {}'.format(args.use_gpu), log_txt_path)
log_print('# Labels : {}'.format(args.labels), log_txt_path)
log_print('# Learning_Rate : {}'.format(args.learning_rate), log_txt_path)

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

# 2.1 placeholder
shape = [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL]

s_var = tf.placeholder(tf.float32, [None] + shape, name = 'image/labeled')
s_label_var = tf.placeholder(tf.float32, [None, 10], name = 'label/labeled')

u_var = tf.placeholder(tf.float32, [None] + shape, name = 'image/unlabeled')
u_augment_var = tf.placeholder(tf.float32, [None] + shape, name = 'image/unlabeled_with_augment')

is_training = tf.placeholder(tf.bool)

# 2.2 build Wide-ResNet-28
s_logits_op, _ = WideResNet(s_var, is_training, reuse = False)

p_logits_ops = [WideResNet(u, is_training, reuse = True)[0] for u in tf.split(u_var, su_ratios)]
q_logits_ops = [WideResNet(u, is_training, reuse = True)[0] for u in tf.split(u_augment_var, su_ratios)]

p_logits_op = tf.concat(p_logits_ops, axis = 0)
q_logits_op = tf.concat(q_logits_ops, axis = 0)

# 2.3 calculate supervised loss / unsupervised loss
s_loss_op = Cross_Entropy_with_logits(s_logits_op, s_label_var)
s_loss_op = tf.reduce_mean(s_loss_op)

u_loss_op = KL_Divergence_with_logits(p_logits_op, q_logits_op)
u_loss_op = tf.reduce_mean(u_loss_op)

loss_op = s_loss_op + u_loss_op

# + l2 regularizer
train_vars = tf.get_collection('trainable_variables', 'Wider-ResNet-28')
l2_vars = [var for var in train_vars if 'kernel' in var.name or 'weights' in var.name]

l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in l2_vars]) * WEIGHT_DECAY
loss_op += l2_reg_loss_op

# 2.4 add EMA
ema = tf.train.ExponentialMovingAverage(decay = EMA_DECAY)
ema_op = ema.apply(train_vars)

_, predictions_op = WideResNet(s_var, is_training, reuse = True, getter = get_getter(ema))

# 2.5 accuracy
correct_op = tf.equal(tf.argmax(predictions_op, axis = -1), tf.argmax(s_label_var, axis = -1))
accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100

# 3. optimizer
learning_rate_var = tf.placeholder(tf.float32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.MomentumOptimizer(learning_rate_var, momentum = 0.9, use_nesterov = True).minimize(loss_op)
    train_op = tf.group(train_op, ema_op)

# 4. tensorboard
train_summary_dic = {
    'Loss/Total_Loss' : loss_op,
    'Loss/Supervised_Loss' : s_loss_op,
    'Loss/UnSupervised_Loss' : u_loss_op,
    'Loss/L2_Regularization_Loss' : l2_reg_loss_op,                                                                                                                                                                                                                                                     
    'Accuracy/Train' : accuracy_op,
    'Learning_rate' : learning_rate_var,
}

train_summary_list = []
for name in train_summary_dic.keys():
    value = train_summary_dic[name]
    train_summary_list.append(tf.summary.scalar(name, value))
train_summary_op = tf.summary.merge(train_summary_list)

test_accuracy_var = tf.placeholder(tf.float32)
test_accuracy_op = tf.summary.scalar('Accuracy/Test', test_accuracy_var)

# 4. train loop
sess = tf.Session()
sess.run(tf.global_variables_initializer())

learning_rate = args.learning_rate
log_print('[i] max_iteration : {}'.format(MAX_ITERATION), log_txt_path)

train_writer = tf.summary.FileWriter('./logs/train_with_UDA_and_cosine@labels={}@SU_ratios={}'.format(labels, su_ratios))
train_ops = [train_op, loss_op, s_loss_op, u_loss_op, l2_reg_loss_op, accuracy_op, train_summary_op]

saver = tf.train.Saver()

train_time = time.time()
loss_list = []
s_loss_list = []
u_loss_list = []
l2_reg_loss_list = []
accuracy_list = []

warmup_learning_rate = MAX_ITERATION / WARMUP_ITERATION * learning_rate
learning_rate_list = cosine_learning_schedule(learning_rate, warmup_learning_rate, 0.004, WARMUP_ITERATION, MAX_ITERATION)

for iter in range(1, MAX_ITERATION + 1):
    learning_rate = learning_rate_list[iter - 1]

    batch_s_image_list = []
    batch_s_label_list = []
    batch_u_image_list = []
    batch_u_augment_image_list = []

    batch_labeled_indexs = random.sample(labeled_indexs, BATCH_SIZE)
    batch_unlabeled_indexs = random.sample(unlabeled_indexs, BATCH_SIZE * su_ratios)
    
    for s_data in labeled_data_list[batch_labeled_indexs]:
        image, label = s_data
        image = augment.AutoAugment(image, normalize = False)

        batch_s_image_list.append(image)
        batch_s_label_list.append(label)
    
    for u_data in unlabeled_data_list[batch_unlabeled_indexs]:
        batch_u_image_list.append(u_data)
        batch_u_augment_image_list.append(augment.AutoAugment(u_data, normalize = False))
    
    batch_s_image_list = np.asarray(batch_s_image_list, dtype = np.float32)
    batch_s_label_list = np.asarray(batch_s_label_list, dtype = np.float32)
    batch_u_image_list = np.asarray(batch_u_image_list, dtype = np.float32)
    batch_u_augment_image_list = np.asarray(batch_u_augment_image_list, dtype = np.float32) 
    batch_data = [batch_s_image_list, batch_s_label_list, batch_u_image_list, batch_u_augment_image_list]

    _feed_dict = {
        s_var : batch_data[0], 
        s_label_var : batch_data[1], 
        u_var : batch_data[2],
        u_augment_var : batch_data[3],
        is_training : True,
        learning_rate_var : learning_rate
    }
    
    _, loss, s_loss, u_loss, l2_reg_loss, accuracy, summary = sess.run(train_ops, feed_dict = _feed_dict)
    train_writer.add_summary(summary, iter)

    loss_list.append(loss)
    s_loss_list.append(s_loss)
    u_loss_list.append(u_loss)
    l2_reg_loss_list.append(l2_reg_loss)
    accuracy_list.append(accuracy)

    if iter % 100 == 0:
        train_time = int(time.time() - train_time)
        loss = np.mean(loss_list)
        s_loss = np.mean(s_loss_list)
        u_loss = np.mean(u_loss_list)
        l2_reg_loss = np.mean(l2_reg_loss_list)
        accuracy = np.mean(accuracy_list)

        log_print('[i] iter = {}, loss = {:.4f}, supervised_loss = {:.4f}, unsupervised_loss = {:.4f}, l2_reg_loss = {:.4f}, accuracy = {:.2f}%, train_time = {}sec'.format(iter, loss, s_loss, u_loss, l2_reg_loss, accuracy, train_time), log_txt_path)
        
        train_time = time.time()
        loss_list = []
        s_loss_list = []
        u_loss_list = []
        l2_reg_loss_list = []
        accuracy_list = []

    if iter % 2000 == 0:
        test_time = time.time()
        test_accuracy_list = []

        for i in range(test_iteration):
            batch_data_list = test_data_list[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

            batch_image_data = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL), dtype = np.float32)
            batch_label_data = np.zeros((BATCH_SIZE, CLASSES), dtype = np.float32)
            
            for i, (image, label) in enumerate(batch_data_list):
                batch_image_data[i] = image.astype(np.float32)
                batch_label_data[i] = label.astype(np.float32)
            
            _feed_dict = {
                s_var : batch_image_data,
                s_label_var : batch_label_data,
                is_training : False
            }

            accuracy = sess.run(accuracy_op, feed_dict = _feed_dict)
            test_accuracy_list.append(accuracy)

        test_time = int(time.time() - test_time)
        test_accuracy = np.mean(test_accuracy_list)

        summary = sess.run(test_accuracy_op, feed_dict = {test_accuracy_var : test_accuracy})
        train_writer.add_summary(summary, iter)

        log_print('[i] iter = {}, test_accuracy = {:.2f}, test_time = {}sec'.format(iter, test_accuracy, test_time), log_txt_path)

saver.save(sess, './model/UDA_with_cosine@labels={}@SU_ratios={}.ckpt'.format(labels, su_ratios))
