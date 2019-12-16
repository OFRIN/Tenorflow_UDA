
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

train_steps = 40000
warmup_steps = 2000
learning_rate = 0.03
min_lr_ratio = 0.004

global_step = tf.placeholder(tf.int32)
warmup_lr = tf.to_float(global_step) / tf.to_float(warmup_steps) * learning_rate

# decay the learning rate using the cosine schedule
decay_lr = tf.train.cosine_decay(
    learning_rate,
    global_step = global_step-warmup_steps,
    decay_steps = train_steps-warmup_steps,
    alpha = min_lr_ratio)

learning_rate = tf.where(global_step < warmup_steps, warmup_lr, decay_lr)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

step_list = []
lr_list = []

for step in range(train_steps):
    lr = sess.run(learning_rate, feed_dict = {global_step : step})
    
    if step % 1000 == 0: 
        print(step, lr)
    # input()

    step_list.append(step)
    lr_list.append(lr)

# 0.0, 0.03
print(min(lr_list), max(lr_list))

plt.plot(step_list, lr_list)
plt.show()

