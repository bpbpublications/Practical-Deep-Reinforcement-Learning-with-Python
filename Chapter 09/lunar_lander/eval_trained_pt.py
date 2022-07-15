import os
import random
from time import sleep
import gym
import numpy as np
from ch9.dqn.pt.dqn_agent import DqnPtAgent

cwd = os.path.dirname(os.path.abspath(__file__))

env = gym.make('LunarLander-v2')
seed = 1

random.seed(seed)
env.seed(seed)

# PyTorch Implementation
agent = DqnPtAgent(state_size = 8, action_size = 4)
load_path = cwd + '/saved_models/presaved_dqn_pt_agent.pth'

agent.load(load_path)

episodes = 10

scores = []

for e in range(1, episodes + 1):

    state = env.reset()
    score = 0

    while True:
        env.render()
        sleep(.03)
        action = agent.act(state, mode = 'test')
        next_state, reward, done, _ = env.step(action)
        state = next_state
        score += reward
        if done:
            break

    scores.append(score)

print(f'Episode {e} Average Score: {np.mean(scores)}')
