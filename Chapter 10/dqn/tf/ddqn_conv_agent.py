import numpy as np
import random
import tensorflow as tf
from ch10.dqn.replay_buffer import ReplayBuffer
from ch10.dqn.tf.q_conv_model import TfQConvNet
from ch10.utils import tf_2d_gather


class DdqnConvTfAgent:
    """Interacts with and learns from the environment."""

    def __init__(
            self,
            frames,
            action_size,
            degp_epsilon = 1,
            degp_decay_rate = .9,
            degp_min_epsilon = .1,
            update_period = 4,  # target network update period
            train_batch_size = 64,  # minibatch size
            replay_buffer_size = 100_000,
            tau = 1e-3,
            gamma = 0.99,
            learning_rate = 5e-4
    ):
        """
        """
        self.action_size = action_size

        self.degp_epsilon = self.degp_initial_epsilon = degp_epsilon
        self.degp_decay_rate = degp_decay_rate
        self.degp_min_epsilon = degp_min_epsilon

        # Q-Network
        self.net_main = TfQConvNet(action_size)
        self.net_target = TfQConvNet(action_size)

        self.net_main.build(input_shape = (train_batch_size, 130, 160, frames))
        self.net_target.build(input_shape = (train_batch_size, 130, 160, frames))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError()

        # Replay memory
        self.memory = ReplayBuffer(action_size, replay_buffer_size, train_batch_size)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.training_steps_count = 0
        self.update_period = update_period
        self.train_batch_size = train_batch_size
        self.replay_buffer_size = replay_buffer_size
        self.tau = tau  # network blending parameter
        self.gamma = gamma

    def before_episode(self):
        """Adjusting Decayed Epsilon Greedy Policy Parameters before new episode"""

        # Decaying epsilon
        self.degp_epsilon *= self.degp_decay_rate
        self.degp_epsilon = max(self.degp_epsilon, self.degp_min_epsilon)

    def step(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        self.training_steps_count += 1

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.train_batch_size:
            self.learn()

    def act(self, state, mode = 'train'):
        state = tf.transpose(state, perm = [1, 2, 0])
        if mode == 'train':
            # Epsilon Greedy Policy
            if random.random() > self.degp_epsilon:
                state = tf.constant(state, dtype = tf.float64)
                state = tf.expand_dims(state, axis = 0)
                action_values = self.net_main(state)
                return np.argmax(action_values.numpy())
            else:
                return random.choice(np.arange(self.action_size))
        else:
            # Greedy Policy
            return random.choice(np.arange(self.action_size))

    def learn(self):

        samples = self.memory.sample()
        s, a, r, s_next, dones = samples

        s_next = tf.transpose(s_next, perm = [0, 2, 3, 1])
        s = tf.transpose(s, perm = [0, 2, 3, 1])

        # V'(s') = max(Q(s',a))
        v_s_next = tf.expand_dims(tf.reduce_max(self.net_target(s_next), axis = 1), axis = 1)

        variables = self.net_main.variables

        with tf.GradientTape() as tape:
            # q_sa
            q_sa_pure = self.net_main(s)
            # q_sa_pure.shape = [train_batch_size, action_size]
            q_sa = tf_2d_gather(q_sa_pure, a)
            # q_sa.shape = [train_batch_size, 1]

            # td = r + gamma * V(s') - Q(s,a)
            td = r + (self.gamma * v_s_next * (1 - dones)) - q_sa

            # Compute loss
            # TD -> 0
            error = self.loss(td, tf.zeros(td.shape))

        gradient = tape.gradient(error, variables)
        self.optimizer.apply_gradients(zip(gradient, variables))

        if self.training_steps_count % self.update_period == 0:
            # update target network
            self.soft_update()

    def soft_update(self):
        """
        Soft update model parameters.
        net_main -> net_target
        """
        target_params = self.net_target.variables
        net_params = self.net_main.variables
        for i, (t_param, n_param) in enumerate(zip(target_params, net_params)):
            blending_params = self.tau * n_param.numpy() + (1.0 - self.tau) * t_param.numpy()
            target_params[i] = tf.Variable(blending_params)

        self.net_target.set_weights(target_params)

    def save(self, path):
        self.net_main.save(path)

    def load(self, path):
        self.net_main = tf.keras.models.load_model(path)
