import random
from ch4.gym.multiarmed_bandit_env import get_bandit_env_5
from ch4.policy.greedy_policy import greedy_policy
import matplotlib.pyplot as plt
import numpy as np


def run_greedy_policy(balance, env, exploration = 10):
    state = env.reset()
    rewards = []

    for i in range(balance):
        action = greedy_policy(state, exploration)
        state, reward, done, debug = env.step(action)
        rewards.append(reward)

    env.close()
    return env, rewards


if __name__ == '__main__':
    seed = 0
    random.seed(seed)

    balance = 1_000
    env = get_bandit_env_5()
    env, rewards = run_greedy_policy(balance, env)
    env.render()

    cum_rewards = np.cumsum(rewards)
    plt.plot(cum_rewards)
    plt.title('Greedy Policy')
    plt.xlabel('Trials')
    plt.ylabel('Reward')
    plt.show()
