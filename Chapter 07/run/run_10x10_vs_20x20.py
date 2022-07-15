import random
import gym
import numpy as np
from ch7.discrete_q_learning_agent import DiscreteQLearningAgent
from ch7.utils import create_grid, plot_rolling
import matplotlib.pyplot as plt

# Initializing Environment
env = gym.make('MountainCar-v0')

# Setting Random Seed
seed = 11
env.seed(seed)
random.seed(seed)
np.random.seed(seed)

s_space = env.observation_space
labels = ['Position', 'Velocity']

# Q-Learning Hyperparamters
train_episodes = 10_000
test_episodes = 1_000

# Discretization Types
bin_map = {
    'Coarse 10x10': (10, 10),
    'Fine 20x20':   (20, 20)
}
# Test Results
test_results = {}

# Evaluating Discretization Type
for discretization_name, bins in bin_map.items():
    grid = create_grid(s_space.low, s_space.high, bins)
    # Agent
    agent = DiscreteQLearningAgent(env, grid)

    # Training
    train_rewards, d_changes = agent.run(env, episodes = train_episodes)
    plot_rolling(
        train_rewards,
        title = f'{discretization_name}: Train Total Rewards')

    # Testing
    test_rewards, _ = agent.run(env, episodes = test_episodes, mode = 'test')
    test_results[discretization_name] = test_rewards

env.close()

# Comparing Test Results
fig, ax = plt.subplots()
ax.boxplot(test_results.values())
ax.set_xticklabels(test_results.keys())
ax.set_title('Test Results')
plt.show()

for k, data in test_results.items():
    print(f'{k}. Avg: {round(sum(data) / len(data), 2)}')
