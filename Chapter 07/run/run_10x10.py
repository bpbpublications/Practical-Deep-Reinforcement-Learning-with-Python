import random
import gym
import numpy as np
from ch7.discrete_q_learning_agent import DiscreteQLearningAgent
from ch7.utils import create_grid, plot_rolling, plot_q_map, plot_route

# Initializing Environment
env = gym.make('MountainCar-v0')

# Setting Random Seed
seed = 0
env.seed(seed)
random.seed(seed)
np.random.seed(seed)

# Discretization
s_space = env.observation_space
bins = (10, 10)
labels = ['Position', 'Velocity']
grid = create_grid(s_space.low, s_space.high, bins)

# Q-Learning Hyperparamters
train_episodes = 10_000
alpha = .02

# Testing Parameters
test_episodes = 500
live_episodes = 10


# Agent
agent = DiscreteQLearningAgent(env, grid, q_alpha = alpha)

# Training
train_rewards, d_changes = agent.run(env, episodes = train_episodes)

# Training Results
plot_rolling(train_rewards, title = 'Train Total Rewards')
plot_rolling(d_changes, title = 'Train Direction Changes')

# Q(s,a) visualization
plot_q_map(agent.q, labels, title = 'Q Map')

# Testing
test_rewards, _ = agent.run(env, episodes = test_episodes, mode = 'test')
# Testing Results
avg_reward = np.mean(test_rewards)
print(f"Exploitation Average Reward: {avg_reward}")
plot_rolling(test_rewards, title = 'Test Total Rewards')

# Running Live
for e in range(1, live_episodes + 1):
    state = env.reset()
    score = 0
    state_history = []
    while True:
        action = agent.act(state, mode = 'test')
        state_history.append(agent.last_state)
        env.render()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            plot_route(
                bins, state_history, labels,
                title = f'Live Episode: {e}'
            )
            break
    print('Final score:', score)

env.close()
