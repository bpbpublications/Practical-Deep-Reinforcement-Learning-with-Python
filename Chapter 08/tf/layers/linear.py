import tensorflow as tf
from tensorflow.keras.layers import Dense

x = tf.constant([[1.0, 2.0, 3.0]])

linear = Dense(units = 2)

# We have to `build` layer to initialize it
linear.build(input_shape = x.shape)

# set weights
linear.set_weights([
    tf.Variable([[0, 1], [2, 0], [5, 2]]),  # weights
    tf.Variable([1, 1])  # bias
])

y = linear(x)

print(y.numpy())
