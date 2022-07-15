import tensorflow as tf
from tensorflow.keras.layers import Dense


class TfCritic(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.linear1 = Dense(128, activation = 'relu')
        self.linear2 = Dense(256, activation = 'relu')
        self.linear3 = Dense(1, activation = 'relu')

    def call(self, state):
        x = tf.convert_to_tensor(state)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
