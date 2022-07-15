import random
import numpy as np
import gym
from ch5.monte_carlo.policy import monte_carlo_policy

seed = 6
random.seed(seed)

a_map = {
    1: 'Hit',
    0: 'Stick'
}

episodes = 10
gamma = 0.8
env = gym.make('Blackjack-v0')
env.seed(seed)

for e in range(1, episodes + 1):
    state_history = monte_carlo_policy(env)
    states, actions, rewards, next_states = zip(*state_history)
    discounts = np.array([gamma**i for i in range(len(rewards) + 1)])

    for i, state in enumerate(states):
        G = round(sum(rewards[i:] * discounts[:-(1 + i)]), 4)
        print(f'{e}: {state}, {a_map[actions[i]]} -> {G}')
