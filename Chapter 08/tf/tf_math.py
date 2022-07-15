import tensorflow as tf

x = tf.ones((1, 2), dtype = tf.float32)  # (1, 1)

y = tf.range(0, 2, dtype = tf.float32)  # (0, 1)

# implicit addition
z = x + y  # (1, 2)

# explicit addition
w = tf.add(z, y)  # (1, 3)

# implicit multiplication
k = w * -1  # (-1, -3)

# absolute value
a = tf.abs(k)  # (1, 3)

# implicit division
b = a / 2  # (0.5, 1.5)

# Rounding to nearest integer lower than
c = tf.floor(b)  # (0, 1)

# Rounding to nearest integer greater than
d = tf.math.ceil(b)  # (1, 2)

# Computes element-wise equality
eq = tf.equal(c, d)  # (False, False)

# Mean tensor value
avg = tf.reduce_mean(d)  # 1.5

# Max tensor value
mx = tf.math.reduce_max(d)  # 2

# Min tensor value
mn = tf.math.reduce_max(d)  # 1

# Sum of all tensor values
sm = tf.math.reduce_sum(d)  # 3
