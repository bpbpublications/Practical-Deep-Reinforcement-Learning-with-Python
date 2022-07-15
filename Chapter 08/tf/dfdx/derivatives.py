import tensorflow as tf
from ch8.tf.dfdx.function import get_function

args = [2, 4, 3, 5]

with tf.GradientTape() as tape:
    y, params = get_function(*args)

dy_dx1 = tape.gradient(y, params['x1'])

print(dy_dx1.numpy())
