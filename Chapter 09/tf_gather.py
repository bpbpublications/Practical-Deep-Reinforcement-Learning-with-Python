import tensorflow as tf
from ch9.utils import tf_2d_gather

params = tf.constant([[1, 2], [3, 4]])
idx = tf.constant([[0], [1]])

out = tf_2d_gather(params, idx)

print(out.numpy())
# [[1]
#  [4]]
