import random
import gym
import numpy as np
from ch7.utils import discretize, create_grid, plot_route

env = gym.make('MountainCar-v0')
seed = 0
env.seed(seed)
random.seed(seed)
np.random.seed(seed)

s_space = env.observation_space

bins = (10, 10)
grid = create_grid(s_space.low, s_space.high, bins)

state_history = []

d_state = discretize(env.reset(), grid)
state_history.append(d_state)
while True:
    env.render()
    random_action = env.action_space.sample()
    state, reward, done, debug = env.step(random_action)
    d_state = discretize(state, grid)
    state_history.append(d_state)
    print(d_state)
    if done:
        break
env.close()

plot_route(bins, state_history, ['Position', 'Velocity'])
