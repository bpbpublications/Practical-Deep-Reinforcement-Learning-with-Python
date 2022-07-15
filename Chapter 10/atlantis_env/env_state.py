import gym

env = gym.make('Atlantis-v0')
action_size = env.action_space.n

state = env.reset()
print(state.shape)
