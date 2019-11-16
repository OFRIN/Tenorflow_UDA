import tensorflow as tf

def KL_Divergence_with_logits(p_logits, q_logits):
    p = tf.nn.softmax(p_logits, axis = -1)

    log_p = tf.nn.log_softmax(p_logits, axis = -1)
    log_q = tf.nn.log_softmax(q_logits, axis = -1)

    kl = tf.reduce_sum(p * (log_p - log_q), axis = -1)
    return kl

def Cross_Entropy_with_logits(logits, labels):
    class_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = logits)
    return class_loss

