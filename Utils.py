
import numpy as np

from Define import *

def log_print(string, log_path = './log.txt'):
    print(string)
    
    f = open(log_path, 'a+')
    f.write(string + '\n')
    f.close()

def cosine_learning_schedule(st_lr, warmup_lr, end_lr, warmup_iteration, max_iteration):
    learning_rate_list = []
    decay_iteration = max_iteration - warmup_iteration
    t_warmup_lr = (warmup_lr - st_lr) / warmup_iteration

    for t in range(1, warmup_iteration + 1):
        learning_rate_list.append(st_lr + t * t_warmup_lr)

    for t in range(1, decay_iteration + 1):
        learning_rate = 0.5 * warmup_lr * (1 + np.cos(t * np.pi / decay_iteration))
        learning_rate_list.append(max(learning_rate, end_lr))

    return learning_rate_list