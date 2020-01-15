import tensorflow as tf

def KL_Divergence_with_logits(p_logits, q_logits):
    p = tf.nn.softmax(p_logits, axis = -1)
    
    log_p = tf.nn.log_softmax(p_logits, axis = -1)
    log_q = tf.nn.log_softmax(q_logits, axis = -1)

    kl = tf.reduce_sum(p * (log_p - log_q), axis = -1)
    return kl

# mode_list = ['exp_schedule', 'log_schedule', 'linear_schedule']
def TSA_schedule(global_step, max_iteration, mode, K):
    t = tf.cast(global_step, tf.float32)
    T = tf.cast(max_iteration, tf.float32)

    if mode == 'log_schedule':
        alpha_t = 1 - tf.exp(-t/T * 5)
    elif mode == 'linear_schedule':
        alpha_t = t/T
    elif mode == 'exp_schedule':
        alpha_t = tf.exp((t/T - 1) * 5)
    else:
        assert False, "[!] TSA_schedule : {}".format(mode)

    return alpha_t, alpha_t * (1 - 1/K) + 1/K

if __name__ == '__main__':
    p_logits = [
        [0.5, 0.1515, 12],
        [0.5, 0.1515, 12]
    ]
    q_logits = [
        [0.5, 0.1515, 12],
        [0.5, 10, 14]
    ]
    
    loss = KL_Divergence_with_logits(p_logits, q_logits)
    print(loss)

    sess = tf.Session()
    print(sess.run(loss))

