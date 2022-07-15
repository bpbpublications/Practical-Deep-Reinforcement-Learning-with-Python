import random
import numpy as np
from ch6.gym_maze.maze_env import MazeEnv
from ch6.maze.core import maze_states
from ch6.maze.map import map1
from ch6.td.policy import epsilon_greedy_policy
from ch6.td.td import q_learning
from ch6.td.utils import plot_q_map

actions = ['U', 'R', 'D', 'L']
x_coord, y_coord, blocks, goal = map1()

seed = 10
random.seed(seed)
env = MazeEnv(x_coord, y_coord, blocks, goal)
env.seed(seed)

states = maze_states(x_coord, y_coord)

epsilon = .2
gamma = 0.9
alpha = 0.8

Q = np.array(np.zeros([len(states), len(actions)]))

state = env.reset()

i = 1
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
        plot_q_map(x_coord, y_coord, blocks, Q, goal,
                   title = f'After Final Action')
        break
    else:
        if i % 10 == 0 or i < 3:
            plot_q_map(x_coord, y_coord, blocks, Q, goal,
                       title = f'After Action: {i}')
    i += 1
