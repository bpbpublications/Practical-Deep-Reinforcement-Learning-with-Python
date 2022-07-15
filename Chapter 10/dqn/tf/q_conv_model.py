import tensorflow as tf
from keras.layers import Dense, Conv2D
from tensorflow.python.keras.layers import MaxPool2D, Flatten


class TfQConvNet(tf.keras.Model):
    """Actor (Policy) Model."""

    def __init__(self, action_size):
        """Initialize parameters and build model.
        """
        super(TfQConvNet, self).__init__()
        self.conv1 = Conv2D(
            filters = 8,
            kernel_size = 7,
            activation = tf.nn.relu
        )

        # max pool layer
        self.pool = MaxPool2D(pool_size = 3)

        # 2nd convolution layer
        self.conv2 = Conv2D(
            filters = 16,
            kernel_size = 5,
            activation = tf.nn.relu
        )

        # Flatten Layer
        self.flat = Flatten()

        self.fc1 = Dense(units = 256, activation = tf.nn.relu)
        self.fc2 = Dense(units = 32, activation = tf.nn.relu)
        self.fc3 = Dense(units = action_size)

    def call(self, x, **kwargs):
        """Build a network that maps state -> action values."""
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = self.flat(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
