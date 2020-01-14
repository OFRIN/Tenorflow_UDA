
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

mode = 'log_schedule'
max_iteration = 400000

x_list = (1 + np.arange(max_iteration)) / max_iteration
at_list = [TSA_schedule(step, max_iteration, mode, 10)[0] for step in range(1, max_iteration + 1)]
nt_list = [TSA_schedule(step, max_iteration, mode, 10)[1] for step in range(1, max_iteration + 1)]

plt.plot(x_list, at_list)
plt.plot(x_list, nt_list)
plt.show()

