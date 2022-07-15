from time import sleep
import gym

env = gym.make('CartPole-v1')
env.reset()
env.render()
sleep(10)
