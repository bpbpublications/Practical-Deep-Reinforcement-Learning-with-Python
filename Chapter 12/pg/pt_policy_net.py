import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PtPolicyNet(nn.Module):

    def __init__(self, state_size, action_space):
        super(PtPolicyNet, self).__init__()
        self.linear1 = nn.Linear(state_size, 32)
        self.linear2 = nn.Linear(32, action_space)
        self.opt = torch.optim.Adam(self.parameters())
        self.gamma = .99

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x), dim = 1)
        return x

    def act_sample(self, state):
        # probabilities for each action
        probs = self(torch.tensor(state).unsqueeze(0).float())
        # distribution
        dist = Categorical(probs)
        # sampling random action from distribution 'dist'
        action = dist.sample().item()
        return action

    def update(self, buffer):
        """
        Updates Neural Network using Policy Gradient method
        """

        # unpacks episode history
        rewards, actions, states = buffer.unzip()

        # Counting discounted rewards
        sum_reward = 0
        discnt_rewards = []
        rewards.reverse()
        for r in rewards:
            sum_reward = r + self.gamma * sum_reward
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()
        discnt_rewards = torch.tensor(discnt_rewards).float()

        # preprocess states and actions
        states = torch.tensor(states).float()
        actions = torch.tensor(actions)

        # Calculating Gradient
        probs = self(states)
        sampler = Categorical(probs)

        # log(Pi(at|st))
        log_probs = sampler.log_prob(actions)

        # E[ log(Pi(at|st)) ]
        E = torch.sum(log_probs * discnt_rewards)

        # Since our goal is to maximize E and optimizers are made for
        # minimization, we are changing the sign of E value
        loss = - E

        # Executing gradient shift: theta = theta + a x Gradient
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return discnt_rewards
