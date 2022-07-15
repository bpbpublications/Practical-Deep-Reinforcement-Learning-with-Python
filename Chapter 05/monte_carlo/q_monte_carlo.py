import random

import gym
import numpy as np
from collections import defaultdict
from ch5.monte_carlo.policy import monte_carlo_policy


def train_q_monte_carlo(env, train_episodes, gamma = 1):
    G = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for e in range(1, train_episodes + 1):

        episode = monte_carlo_policy(env)

        states, actions, rewards, next_states = zip(*episode)

        discounts = np.array([gamma**i for i in range(len(rewards) + 1)])

        for i, state in enumerate(states):
            g = sum(rewards[i:] * discounts[:-(1 + i)])
            G[state][actions[i]] += g
            N[state][actions[i]] += 1.0
            Q[state][actions[i]] = G[state][actions[i]] / N[state][actions[i]]

        if e % 10_000 == 0:
            print(f'Episodes: {e}/{train_episodes}')

    return Q


if __name__ == '__main__':

    env = gym.make('Blackjack-v0')

    seed = 0
    random.seed(seed)
    env.seed(seed)

    Q = train_q_monte_carlo(env, 50_000)

    # print several values
    i = 0
    for state in Q:

        if i == 10:
            break

        print(f'State: {state} | action-values: {Q[state]}')

        i += 1
