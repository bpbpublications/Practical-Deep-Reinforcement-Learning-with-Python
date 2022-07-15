import random
import numpy as np


def e_greedy_policy(state, explore = 10, epsilon = .1):
    bandits = len(state)

    trials = sum([len(state[b]) for b in range(bandits)])
    total_explore_trials = bandits * explore

    # exploration
    if trials <= total_explore_trials:
        return trials % bandits

    # random bandit
    if random.random() < epsilon:
        return random.randint(0, bandits - 1)

    # exploitation
    avg_rewards = [sum(state[b]) / len(state[b]) for b in range(bandits)]
    best_bandit = np.argmax(avg_rewards)

    return best_bandit
