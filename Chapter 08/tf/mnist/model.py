import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D


class MnistClassifier(Model):

    def __init__(self):
        super().__init__()

        # 1st convolution layer
        self.conv1 = Conv2D(
            filters = 32,
            kernel_size = 5,
            activation = tf.nn.relu
        )

        # max pool layer
        self.pool = MaxPool2D(pool_size = 2)

        # 2nd convolution layer
        self.conv2 = Conv2D(
            filters = 64,
            kernel_size = 5,
            activation = tf.nn.relu
        )

        # Flatten Layer
        self.flatten = Flatten()

        # 1st linear layer
        self.lin1 = Dense(
            units = 256,
            activation = tf.nn.relu
        )

        # Dropout Layer
        self.drop = Dropout(rate = .5)

        # 2nd Linear Layer
        self.lin2 = Dense(
            units = 10,
            activation = tf.nn.softmax
        )

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.drop(x)
        return self.lin2(x)
