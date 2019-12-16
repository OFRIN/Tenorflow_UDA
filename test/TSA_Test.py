
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# mode_list = ['exp_schedule', 'log_schedule', 'linear_schedule']
def TSA_schedule(global_step, max_iteration, mode, K):
    t = global_step
    T = max_iteration

    if mode == 'log_schedule':
        alpha_t = 1 - np.exp(-t/T * 5)
    elif mode == 'linear_schedule':
        alpha_t = t/T
    elif mode == 'exp_schedule':
        alpha_t = np.exp((t/T - 1) * 5)
    else:
        assert False, "[!] TSA_schedule : {}".format(mode)
    
    return alpha_t, alpha_t * (1 - 1/K) + 1/K

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

