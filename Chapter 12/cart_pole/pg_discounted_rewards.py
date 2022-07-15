# import dependencies
import random
import numpy as np
import matplotlib.pyplot as plt
import gym
from ch11.utils import Buffer
from ch12.pg.tf_policy_net import TfPolicyNet
from ch12.pg.pt_policy_net import PtPolicyNet

# create environment
env = gym.make("CartPole-v1")

# making script reproducible
seed = 0
random.seed(seed)
np.random.seed(seed)
env.seed(seed)

# Instantiate the agent

# TensorFlow Implementation
policy = TfPolicyNet(env.action_space.n)

# PyTorch Implementation
# policy = PtPolicyNet(env.observation_space.shape[0], env.action_space.n)

episodes = 1_000
total_scores = []
avg_scores = []
buffer = Buffer()

for e in range(1, episodes + 1):
    buffer.clear()
    # reset environment
    state = env.reset()
    epoch_rewards = 0
    while True:
        action = policy.act_sample(state)
        # use that action in the environment
        new_state, reward, done, info = env.step(action)
        epoch_rewards += reward
        # store state, action and reward
        buffer.add(reward, action, state)

        state = new_state
        if done:
            total_scores.append(epoch_rewards)
            break

    disc_rewards = policy.update(buffer)

    if e % 100 == 0:
        plt.plot(disc_rewards)
        plt.title(f'Policy Gradient. Discounted Rewards.\nEpisode: {e}')
        plt.ylabel('Discounted reward sum (Gt)')
        plt.xlabel('State Number (St)')
        plt.show()

# close environment
env.close()
