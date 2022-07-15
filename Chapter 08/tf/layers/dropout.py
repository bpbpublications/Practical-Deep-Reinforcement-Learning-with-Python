import tensorflow as tf
from keras.layers import Dropout

tf.random.set_seed(0)

t = tf.range(1, 6, dtype = tf.float32)
print(f'Initial Tensor: {t.numpy()}')

dropout = Dropout(rate = .5)

# Training Mode
r = dropout(t, training = True)
print(f'Dropout Train: {r.numpy()}')

# Evaluation Mode
r = dropout(t)
print(f'Dropout Eval: {r.numpy()}')
