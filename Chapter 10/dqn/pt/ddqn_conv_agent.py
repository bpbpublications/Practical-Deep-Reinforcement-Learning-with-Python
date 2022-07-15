import numpy as np
import random
import torch
import torch.optim as optim
from ch10.dqn.pt.q_conv_model import PtQConvNet
from ch10.dqn.replay_buffer import ReplayBuffer


class DdqnConvPtAgent:
    """Interacts with and learns from the environment."""

    def __init__(
            self,
            frames,
            action_size,
            degp_epsilon = 1,
            degp_decay_rate = .9,
            degp_min_epsilon = .1,
            update_period = 5,  # target network update period
            train_batch_size = 64,  # minibatch size
            replay_buffer_size = 100_000,
            tau = 1e-3,
            gamma = 0.99,
            learning_rate = 5e-4
    ):
        """
        """
        self.frames = frames
        self.action_size = action_size

        self.degp_epsilon = self.degp_initial_epsilon = degp_epsilon
        self.degp_decay_rate = degp_decay_rate
        self.degp_min_epsilon = degp_min_epsilon

        # Q-Network
        self.net_main = PtQConvNet(frames, action_size)
        self.net_target = PtQConvNet(frames, action_size)

        self.optimizer = optim.Adam(self.net_main.parameters(), lr = learning_rate)
        self.loss = torch.nn.L1Loss()

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
        r = random.random()
        random_action = mode == 'train' and r < self.degp_epsilon

        if random_action:
            # Random Policy
            action = random.choice(np.arange(self.action_size))
        else:
            # Greedy Policy
            state = torch.from_numpy(np.flip(state, axis = 0).copy())\
                .float().unsqueeze(0)
            # state.shape = [1, 8]

            self.net_main.eval()
            with torch.no_grad():
                action_values = self.net_main(state)
                # action_values.shape = [1, 4]

            self.net_main.train()
            action = np.argmax(action_values.cpu().data.numpy())

        # action = [1]
        return action

    def learn(self):

        samples = self.memory.sample()

        s, a, r, s_next, dones = samples

        # to PyTorch Tensors
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).long()
        r = torch.from_numpy(r).float()
        s_next = torch.from_numpy(s_next).float()
        dones = torch.from_numpy(dones).float()

        # V'(s') = max(Q(s',a))
        v_s_next = self.net_target(s_next).detach().max(1)[0].unsqueeze(1)

        # q_sa
        q_s = self.net_main(s)
        q_sa = q_s.gather(dim = 1, index = a)

        # td = r + gamma * V'(s') - Q(s,a)
        td = r + (self.gamma * v_s_next * (1 - dones)) - q_sa

        # Compute loss
        # TD -> 0
        loss = self.loss(td, torch.zeros(td.shape))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.training_steps_count % self.update_period == 0:
            # update target network
            self.soft_update()

    def soft_update(self):
        """
        Soft update model parameters.
        net_main -> net_target
        """
        target_params = self.net_target.parameters()
        net_params = self.net_main.parameters()
        for t_param, n_param in zip(target_params, net_params):
            blending_params = self.tau * n_param.data + (1.0 - self.tau) * t_param.data
            t_param.data.copy_(blending_params)

    def save(self, path):
        torch.save(self.net_main.state_dict(), path)

    def load(self, path):
        self.net_main.load_state_dict(torch.load(path))
