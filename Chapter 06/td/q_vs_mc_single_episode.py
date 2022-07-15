import random
import numpy as np
from ch6.gym_maze.maze_env import MazeEnv
from ch6.maze.core import maze_states
from ch6.maze.map import map2
from ch6.td.policy import epsilon_greedy_policy
from ch6.td.td import q_learning

actions = ['U', 'R', 'D', 'L']
x_coord, y_coord, blocks, goal = map2()

seed = 1
random.seed(seed)
env = MazeEnv(x_coord, y_coord, blocks, goal)
env.seed(seed)
states = maze_states(x_coord, y_coord)
initial_state = 'A1'

# Q learning EXPLORATION EPISODE
epsilon = .2
gamma = 0.9
alpha = 0.8

Q = np.array(np.zeros([len(states), len(actions)]))
env.reset()
env.state = initial_state
state = initial_state
env.render()
while True:

    action_idx = epsilon_greedy_policy(epsilon, Q, states.index(state))
    action = actions[action_idx]
    next_state, reward, done, debug = env.step(action)
    Q = q_learning(
        Q,
        states.index(state),
        states.index(next_state),
        action_idx,
        reward,
        gamma,
        alpha
    )
    state = next_state

    if done:
        env.render()
        break

print(f'Q-learning total actions: {env._total_actions}')

# MONTE CARLO EXPLORATION EPISODE
env.reset()
env.state = initial_state
state = initial_state
while True:

    action_idx = epsilon_greedy_policy(epsilon, Q, states.index(state))
    action = actions[action_idx]
    next_state, reward, done, debug = env.step(action)
    state = next_state

    if done:
        env.render()
        break

print(f'Monte Carlo total actions: {env._total_actions}')
