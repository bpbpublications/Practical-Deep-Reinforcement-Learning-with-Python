import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D

A = tf.constant(
    [[
        [[1], [2], [0], [1]],
        [[-1], [0], [3], [2]],
        [[1], [3], [0], [1]],
        [[2], [-2], [1], [0]]
    ]]
    ,
    dtype = tf.float32
)

# A.shape = (1, 4, 4, 1)

conv2d = Conv2D(
    filters = 1,
    kernel_size = (2, 2),
    use_bias = False
)

# Initializing Convolution Layer
conv2d.build(input_shape = A.shape)

# Setting 2x2 Convolution Kernel
w = np.array([[[[1]], [[-1]]],
              [[[-1]], [[1]]]])
conv2d.set_weights([w])

# Executing Convolution
B = conv2d(A)

print(B.numpy())
