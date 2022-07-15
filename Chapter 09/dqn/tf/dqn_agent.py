import numpy as np
import tensorflow as tf
from ch9.dqn.base_dqn_agent import BaseDqnAgent
from ch9.dqn.tf.q_model import TfQNet
from ch9.utils import tf_2d_gather


class DqnTfAgent(BaseDqnAgent):
    """Interacts with and learns from the environment."""

    def __init__(self, *args, **kwargs):
        super(DqnTfAgent, self).__init__(*args, **kwargs)

    def init_q_net(self, state_size, action_size, learning_rate):
        self.q_net = TfQNet(state_size, action_size)
        self.q_net.build(input_shape = (1, state_size))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError()

    def greedy_act(self, state):
        state = tf.constant(state, dtype = tf.float64)
        state = tf.expand_dims(state, axis = 0)
        action_values = self.q_net(state)
        return np.argmax(action_values.numpy())

    def learn(self):
        samples = self.memory.batch()
        s, a, r, s_next, dones = samples
        variables = self.q_net.variables

        # V(s') = max(Q(s',a))
        v_s_next = tf.expand_dims(tf.reduce_max(self.q_net(s_next), axis = 1), axis = 1)

        with tf.GradientTape() as tape:
            # Q(s,a)
            q_sa_pure = self.q_net(s)
            q_sa = tf_2d_gather(q_sa_pure, a)

            # TD = r + g * V(s') - Q(s,a)
            td = r + (self.gamma * v_s_next * (1 - dones)) - q_sa

            # Compute loss: TD -> 0
            error = self.loss(td, tf.zeros(td.shape))

        gradient = tape.gradient(error, variables)
        self.optimizer.apply_gradients(zip(gradient, variables))

    def save(self, path):
        self.q_net.save(path)

    def load(self, path):
        self.q_net = tf.keras.models.load_model(path)
