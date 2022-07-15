import random
import numpy as np
import matplotlib.pyplot as plt
from ch6.gym_maze.maze_env import MazeEnv
from ch6.maze.core import maze_states
from ch6.maze.map import map1
from ch6.td.policy import epsilon_greedy_policy
from ch9.td.td import q_learning
from ch6.td.utils import plot_q_map

actions = ['U', 'R', 'D', 'L']
x_coord, y_coord, blocks, goal = map1()

seed = 1
random.seed(seed)
env = MazeEnv(x_coord, y_coord, blocks, goal)
env.seed(seed)

states = maze_states(x_coord, y_coord)
Q = np.array(np.zeros([len(states), len(actions)]))

# Training
epsilon = .2
gamma = 0.9
alpha = 0.8
episodes = 1000

td_list = []
for e in range(episodes):
    state = env.reset()
    i = 0
    while True:
        i += 1

        action_idx = epsilon_greedy_policy(epsilon, Q, states.index(state))
        action = actions[action_idx]

        next_state, reward, done, debug = env.step(action)

        Q, td = q_learning(
            Q,
            states.index(state),
            states.index(next_state),
            action_idx,
            reward,
            gamma,
            alpha
        )

        td_list.append(td)

        state = next_state

        if done:
            break

env.close()

plt.plot(td_list)
plt.xlabel('Steps')
plt.ylabel('TD value')
plt.title('Temporal Difference for Maze Problem')
plt.show()
