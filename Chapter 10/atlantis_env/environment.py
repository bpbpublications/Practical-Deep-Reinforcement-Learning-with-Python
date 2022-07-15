from time import sleep

import gym

env = gym.make('Atlantis-v0')
action_size = env.action_space.n

# 0: Do Nothing
# 1: Medium Strike
# 2: Right Strike
# 3: Left Strike


env.reset()
while True:
    sleep(.1)
    img, reward, done, _ = env.step(3)
    env.render()
    env.step(0)
    if done:
        break
