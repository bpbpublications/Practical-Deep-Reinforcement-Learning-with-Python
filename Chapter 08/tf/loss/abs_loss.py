import tensorflow as tf

a = tf.constant([1, 2], dtype = tf.float32)
b = tf.constant([1, 5], dtype = tf.float32)

abs_loss = tf.keras.losses.MeanAbsoluteError()
abs_error = abs_loss(a, b)

print(f'abs: {abs_error.numpy()}')
