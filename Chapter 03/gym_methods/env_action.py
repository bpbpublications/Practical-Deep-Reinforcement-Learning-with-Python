import gym

env = gym.make('CartPole-v1')
env.reset()
state, reward, done, debug = env.step(-1)
