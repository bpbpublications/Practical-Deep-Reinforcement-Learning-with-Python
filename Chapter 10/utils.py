import tensorflow as tf
import numpy as np


def tf_2d_gather(params, idx):
    idx = tf.stack([tf.range(tf.shape(idx)[0]), idx[:, 0]], axis = -1)
    out = tf.gather_nd(params, idx)
    out = tf.expand_dims(out, axis = 1)
    return out


def image_rgb_to_grayscale(img, crop_up = 0, crop_down = 0):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    if crop_down > 1:
        gs_img = np.dot(img[crop_up:-crop_down, :, :3], rgb_weights)
    else:
        gs_img = np.dot(img[crop_up:, :, :3], rgb_weights)
    return gs_img
