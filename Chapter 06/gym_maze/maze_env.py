import gym
from ch6.maze.core import random_state, move, plot_maze_state


class MazeEnv(gym.Env):

    def __init__(self, x_coord, y_coord, blocks, finish_state):
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.blocks = blocks
        self.state = random_state(x_coord, y_coord, blocks)
        self.finish_state = finish_state

        # debug properties
        self._total_actions = 0
        self._total_reward = 0

    def step(self, action):
        prev_state = self.state
        next_state = move(self.state, action, self.x_coord, self.y_coord, self.blocks)
        reward = 0
        done = False
        if next_state == self.finish_state:
            reward = 1000
            done = True
        elif prev_state == next_state:
            reward = -10
        elif prev_state != next_state:
            reward = -1
        self.state = next_state

        self._total_actions += 1
        self._total_reward += reward

        return self.state, reward, done, None

    def reset(self):
        self._total_actions = 0
        self._total_reward = 0
        not_allowed = self.blocks + [self.finish_state]
        self.state = random_state(self.x_coord, self.y_coord, not_allowed)
        return self.state

    def render(self, mode = "ascii"):
        plot_maze_state(
            self.x_coord,
            self.y_coord,
            self.blocks,
            self.state,
            self.finish_state,
            f'Total Reward: {self._total_reward} \n'
            f'Total Actions: {self._total_actions}'
        )
