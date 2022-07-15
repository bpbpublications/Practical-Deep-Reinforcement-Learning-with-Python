import tensorflow as tf
from keras.layers import Dense


class TfQNet(tf.keras.Model):
    """Q-Network. TensorFlow Implementation"""

    def __init__(self, state_size, action_size, fc1_units = 64, fc2_units = 64):
        """
            state_size: Dimension of state space
            action_size: Dimension of action space
            fc1_units: First hidden layer size
            fc2_units: Second hidden layer size
        """
        super(TfQNet, self).__init__()
        self.fc1 = Dense(units = fc1_units, activation = tf.nn.relu)
        self.fc2 = Dense(units = fc2_units, activation = tf.nn.relu)
        self.fc3 = Dense(units = action_size)

    def call(self, state, **kwargs):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.fc3(x)
