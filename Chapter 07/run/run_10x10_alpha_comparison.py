import random
import gym
import numpy as np
from ch7.discrete_q_learning_agent import DiscreteQLearningAgent
from ch7.utils import create_grid, plot_rolling
import matplotlib.pyplot as plt

# Initializing Environment
env = gym.make('MountainCar-v0')

# Setting Random Seed
seed = 0
env.seed(seed)
random.seed(seed)
np.random.seed(seed)

s_space = env.observation_space
labels = ['Position', 'Velocity']
bins = (10, 10)

# Q-Learning Hyperparamters
train_episodes = 10_000
test_episodes = 1_000

alpha_parameters = [.02, .2, .5, .9]
# Test Results
test_results = {}

for alpha in alpha_parameters:
    grid = create_grid(s_space.low, s_space.high, bins)
    # Agent
    agent = DiscreteQLearningAgent(env, grid, q_alpha = alpha)

    # Training
    train_rewards, d_changes = agent.run(env, episodes = train_episodes)
    plot_rolling(
        train_rewards,
        title = f'Alpha {alpha}: Train Total Rewards'
    )
    plot_rolling(
        d_changes,
        title = f'Alpha {alpha}: Train Direction Changes'
    )

    # Testing
    test_rewards, _ = agent.run(env, episodes = test_episodes, mode = 'test')
    test_results[alpha] = test_rewards

env.close()

# Comparing Test Results
fig, ax = plt.subplots()
ax.boxplot(test_results.values())
ax.set_xticklabels(test_results.keys())
ax.set_title('Test Results')
plt.show()

for k, data in test_results.items():
    print(f'{k}. Avg: {round(sum(data) / len(data), 2)}')
