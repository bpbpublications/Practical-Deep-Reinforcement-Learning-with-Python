import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv1D

A = tf.constant([
    [[1], [0], [2], [0], [3], [0]]
], dtype = tf.float32)

conv2d = Conv1D(
    filters = 2,
    kernel_size = 3,
    use_bias = False
)

# Initializing Convolution Layer
conv2d.build(input_shape = A.shape)

# Setting Convolution Kernels
w = np.array([[[1, 0]],
              [[0, 2]],
              [[-1, 0]]])
conv2d.set_weights([w])

# Executing Convolution
B = conv2d(A)

print(B.numpy())
