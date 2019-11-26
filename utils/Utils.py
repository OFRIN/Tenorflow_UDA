
import numpy as np

from core.Define import *

def log_print(string, log_path = './log.txt'):
    print(string)
    
    f = open(log_path, 'a+')
    f.write(string + '\n')
    f.close()
    
def cosine_learning_schedule(st_lr, warmup_lr, end_lr, warmup_iteration, max_iteration, alpha):
    learning_rate_list = []
    decay_iteration = max_iteration - warmup_iteration
    t_warmup_lr = (warmup_lr - st_lr) / warmup_iteration

    for t in range(1, warmup_iteration + 1):
        learning_rate_list.append(st_lr + t * t_warmup_lr)
    
    for t in range(1, decay_iteration + 1):
        cosine_decay = 0.5 * (1 + np.cos(np.pi * t / decay_iteration))
        cosine_decay = (1 - alpha) * cosine_decay + alpha
        learning_rate_list.append(cosine_decay * warmup_lr)

    return learning_rate_list