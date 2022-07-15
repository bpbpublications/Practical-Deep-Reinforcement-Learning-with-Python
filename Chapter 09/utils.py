import tensorflow as tf


def tf_2d_gather(params, idx):
    idx = tf.stack([tf.range(tf.shape(idx)[0]), idx[:, 0]], axis = -1)
    out = tf.gather_nd(params, idx)
    out = tf.expand_dims(out, axis = 1)
    return out
