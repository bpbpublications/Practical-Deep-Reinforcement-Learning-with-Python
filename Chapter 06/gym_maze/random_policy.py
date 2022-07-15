import random
from ch6.gym_maze.maze_env import MazeEnv
from ch6.maze.core import actions
from ch6.maze.map import map1

x_coord, y_coord, blocks, goal = map1()

seed = 0
random.seed(seed)
env = MazeEnv(x_coord, y_coord, blocks, goal)
env.seed(seed)

env.reset()
action_history = []

for i in range(1000):
    if i % 10 == 0:
        env.render()
    action = random.choice(actions)
    print(f'action: {action}')
    action_history.append(action)
    state, reward, done, debug = env.step(action)
    if done:
        env.render()
        break

env.close()

print(f'Moves: {len(action_history)}')
