import cv2
import numpy as np
import matplotlib.pyplot as plt

from Utils import *
from Define import *

learning_rate = 0.03
min_learning_rate = 0.004

warmup_steps = 20000
global_step = 100000

warmup_learning_rate = global_step / warmup_steps * learning_rate
# print(learning_rate, warmup_learning_rate, min_learning_rate)

# learning_rate_list = []
# warmup_lr = (warmup_learning_rate - learning_rate) / warmup_steps

# for i in range(1, warmup_steps + 1):
#     learning_rate_list.append(learning_rate + i * warmup_lr)

# for i in range(1, global_step - warmup_steps + 1):
#     lr = 0.5 * warmup_learning_rate * (1 + np.cos(i * np.pi / (global_step - warmup_steps))) 
#     learning_rate_list.append(lr)

learning_rate_list = cosine_learning_schedule(learning_rate, warmup_learning_rate, min_learning_rate, warmup_steps, global_step)


warmup_learning_rate = MAX_ITERATION / WARMUP_ITERATION * learning_rate
learning_rate_list = cosine_learning_schedule(learning_rate, warmup_learning_rate, 0.004, WARMUP_ITERATION, MAX_ITERATION)

plt.plot(np.arange(len(learning_rate_list)), learning_rate_list)
plt.ylabel('learning_rate')
plt.xlabel('step')
plt.show()