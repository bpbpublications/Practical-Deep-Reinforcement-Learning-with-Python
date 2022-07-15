import random
from time import sleep

import gym

env = gym.make('Atlantis-v0')
action_size = env.action_space.n

# Random Agent
env.reset()
while True:
    env.render()
    sleep(.05)
    img, reward, done, _ = env.step(random.randint(0, action_size - 1))
    if done:
        break
