
import sys
sys.path.insert(1, '../')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.Utils import *

train_steps = 40000
warmup_steps = 2000
learning_rate = 0.03
min_lr_ratio = 0.004

step_list = np.arange(train_steps)
lr_list = cosine_learning_schedule(0.0, learning_rate, min_lr_ratio, warmup_steps, train_steps, alpha = 0.0)
alpha_lr_list = cosine_learning_schedule(0.0, learning_rate, min_lr_ratio, warmup_steps, train_steps, alpha = min_lr_ratio)

print(min(alpha_lr_list), max(alpha_lr_list))

plt.plot(step_list, lr_list)
plt.plot(step_list, alpha_lr_list)
plt.show()

