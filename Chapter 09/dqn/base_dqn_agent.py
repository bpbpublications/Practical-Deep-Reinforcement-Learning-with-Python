import random
import numpy as np
from ch9.dqn.replay_buffer import ReplayBuffer


class BaseDqnAgent:

    def __init__(
            self,
            state_size,
            action_size,
            degp_epsilon = 1,
            degp_decay_rate = .9,
            degp_min_epsilon = .1,
            train_batch_size = 64,
            replay_buffer_size = 100_000,
            gamma = 0.99,
            learning_rate = 5e-4,
            learn_period = 1
    ):
        """
        Deep Q-Network Base Implementation.
        Params:
        state_size - the size of state space
        action_size - the size of action space
        degp_epsilon - decayed epsilon greedy policy initial epsilon value
        degp_decay_rate - decay rate of epsilon greedy policy
        degp_min_epsilon - minimal epsilon value
        learn_period - how often q_net is trained
        """
        self.state_size = state_size
        self.action_size = action_size

        self.degp_epsilon = self.degp_initial_epsilon = degp_epsilon
        self.degp_decay_rate = degp_decay_rate
        self.degp_min_epsilon = degp_min_epsilon

        # Q-Network initialized in init_q_net method
        self.q_net = None
        self.optimizer = None
        self.loss = None
        self.learn_period = learn_period
        self.init_q_net(state_size, action_size, learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(replay_buffer_size, train_batch_size)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.training_steps_count = 0
        self.train_batch_size = train_batch_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma

    def init_q_net(self, state_size, action_size, learning_rate):
        ...

    def before_episode(self):
        """Adjusting Decayed Epsilon Greedy Policy Parameters before new episode"""

        self.degp_epsilon *= self.degp_decay_rate
        self.degp_epsilon = max(self.degp_epsilon, self.degp_min_epsilon)

    def step(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        self.training_steps_count += 1

        if self.training_steps_count % self.learn_period == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.train_batch_size:
                self.learn()

    def act(self, state, mode = 'train'):
        """
        Returns the action
        mode = train|test
        """
        r = random.random()
        random_action = mode == 'train' and r < self.degp_epsilon

        if random_action:
            # Random Policy
            action = random.choice(np.arange(self.action_size))
        else:
            action = self.greedy_act(state)

        return action

    def greedy_act(self, state):
        return None

    def learn(self):
        ...

    def save(self, path):
        ...

    def load(self, path):
        ...
