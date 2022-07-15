import gym
import random

env = gym.make('Blackjack-v0')

seed = 1
random.seed(seed)
env.seed(seed)
