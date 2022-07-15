import gym

env = gym.make('MountainCar-v0')
seed = 0
env.seed(seed)

state = env.reset()
while True:
    env.render()
    random_action = env.action_space.sample()
    state, reward, done, debug = env.step(random_action)
    print(state)
    if done:
        break
env.close()
