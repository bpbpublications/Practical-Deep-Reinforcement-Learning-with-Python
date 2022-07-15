import tensorflow as tf
from tensorflow.keras.layers import Flatten

x = tf.random.uniform((1, 16, 10))

flat = Flatten()

y = flat(x)

print(f'initial shape: {x.shape}')
print(f'flatten shape: {y.shape}')
