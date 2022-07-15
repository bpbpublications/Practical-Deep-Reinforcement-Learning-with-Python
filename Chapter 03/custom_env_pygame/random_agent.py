import random
from time import sleep
from catch_coins_env import CatchCoinsEnv

env = CatchCoinsEnv()

init_state = env.reset()
action_space = [-1, 0, 1]

for _ in range(1000):
    env.render('human')
    action = random.choice(action_space)
    state, reward, done, debug = env.step(action)
    sleep(.3)

env.close()
