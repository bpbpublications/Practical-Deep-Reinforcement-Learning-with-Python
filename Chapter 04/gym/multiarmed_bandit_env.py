import random
import gym
import numpy as np


class MultiArmedBanditEnv(gym.Env):

    def __init__(self, bandits):
        self.bandits = bandits
        self.reset()

    def step(self, action):
        p = self.bandits[action]
        r = random.random()
        reward = 1 if r <= p else -1
        self.state[action].append(reward)
        done = False
        debug = None
        return self.state, reward, done, debug

    def reset(self):
        self.state = {}
        for i in range(len(self.bandits)):
            self.state[i] = []
        return self.state

    def render(self, mode = "ascii"):
        returns = {}
        trials = {}
        for i in range(len(self.bandits)):
            returns[i] = sum(self.state[i])
            trials[i] = len(self.state[i])

        for b, r in returns.items():
            t = trials[b]
            print(f'Bandit {b}| returns: {r}, trials: {t}')
        print(f'Total Trials: {sum(trials.values())}')
        print(f'Total Returns: {sum(returns.values())}')


def get_bandit_env_5():
    bandits = [.45, .45, .4, .6, .4]
    return MultiArmedBanditEnv(bandits)


if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    np.random.seed(1)

    balance = 1_000
    env = get_bandit_env_5()

    state = env.reset()
    rewards = []

    for i in range(balance):
        random_bandit = random.randint(0, 4)
        state, reward, done, debug = env.step(random_bandit)
        rewards.append(reward)

    env.render()
    env.close()
