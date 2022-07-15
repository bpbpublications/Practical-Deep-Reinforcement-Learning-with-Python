import gym
import numpy as np
import matplotlib.pyplot as plt
from ch12.a2c.tf.a2c_agent_tf import TfA2CAgent
from ch12.pg.tf_policy_net import TfPolicyNet
from ch12.a2c.pt.a2c_agent_pt import PtA2CAgent
from ch12.pg.pt_policy_net import PtPolicyNet

# different implementations for PG and A2C buffers
import ch12.utils as a2c_utils
import ch11.utils as pg_utils

env = gym.make("CartPole-v0").unwrapped

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

episodes = 200
total_rewards = []

# TensorFlow Implementation
pg_agent = TfPolicyNet(action_size)
a2c_agent = TfA2CAgent(state_size, action_size)

# PyTorch Implementation
# pg_agent = PtPolicyNet(state_size, action_size)
# a2c_agent = PtA2CAgent(state_size, action_size)

a2c_buffer = a2c_utils.Buffer()
pg_buffer = pg_utils.Buffer()

a2c_average = []
pg_average = []

# A2C performance
for episode in range(1, episodes + 1):
    epoch_rewards = 0
    state = env.reset()
    a2c_buffer.clear()

    while True:
        action = a2c_agent.action_sample(state)
        next_state, reward, done, _ = env.step(action)
        epoch_rewards += reward

        a2c_buffer.add(reward, state, action, done)
        state = next_state

        if done:
            total_rewards.append(epoch_rewards)
            break

    a2c_agent.update(next_state, a2c_buffer)
    a2c_average.append(np.mean(total_rewards[-min(100, episode):]))

    if episode % 10 == 0:
        print(f'A2C Average. Episode {episode}: {a2c_average[-1]}')

# PG performance
for episode in range(1, episodes + 1):
    epoch_rewards = 0
    state = env.reset()
    pg_buffer.clear()

    while True:
        action = pg_agent.act_sample(state)
        next_state, reward, done, _ = env.step(action)
        epoch_rewards += reward

        pg_buffer.add(reward, action, state)
        state = next_state

        if done:
            total_rewards.append(epoch_rewards)
            break

    pg_agent.update(pg_buffer)
    pg_average.append(np.mean(total_rewards[-min(100, episode):]))

    if episode % 10 == 0:
        print(f'PG Average. Episode {episode}: {pg_average[-1]}')

env.close()

# Displaying results
plt.plot(a2c_average, label = 'A2C')
plt.plot(pg_average, label = 'Policy Gradient')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Average Rewards')
plt.title('CartPole. A2C vs PG')
plt.show()
