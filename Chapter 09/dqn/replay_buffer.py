import numpy as np
import random
from collections import namedtuple, deque


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        """
            buffer_size: maximum buffer size
            batch_size: size of training batch
        """
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def batch(self, batch_size = None):
        """Randomly sample a batch of experiences from memory."""
        if batch_size is None:
            batch_size = self.batch_size

        experiences = random.sample(self.memory, k = batch_size)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
