import tensorflow as tf
from keras.layers import MaxPool2D

A = tf.constant(
    [[
        [[1], [2], [-1], [1]],
        [[0], [1], [-2], [-1]],
        [[3], [0], [5], [0]],
        [[0], [1], [4], [-3]]
    ]])

max_pool = MaxPool2D(pool_size = 2)

out = max_pool(A)

print(out.numpy())
