from time import sleep
import gym
import random

env = gym.make('CartPole-v1')

seed = 1
random.seed(seed)
env.seed(seed)

episodes = 10

for i in range(episodes):
    state = env.reset()
    reward_sum = 0
    while True:
        env.render()
        action = 1 if state[2] > 0 else 0
        state, reward, done, debug = env.step(action)
        reward_sum += reward
        sleep(.01)
        if done:
            print(f'Episode {i} reward: {reward_sum}')
            sleep(1)
            break

env.close()
