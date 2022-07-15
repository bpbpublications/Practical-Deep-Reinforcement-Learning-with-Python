import os
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from ch9.dqn.tf.dqn_agent import DqnTfAgent
from ch9.dqn.pt.dqn_agent import DqnPtAgent

cwd = os.path.dirname(os.path.abspath(__file__))

env = gym.make('LunarLander-v2')
seed = 1

random.seed(seed)
env.seed(seed)

# PyTorch Implementation
# agent = DqnPtAgent(state_size = 8, action_size = 4)
# save_path = cwd + '/saved_models/dqn_pt_agent.pth'

# TensorFlow Implementation
agent = DqnTfAgent(state_size = 8, action_size = 4)
save_path = cwd + '/saved_models/dqn_tf_agent'

# Training
episodes = 500

scores = []

for e in range(1, episodes + 1):

    state = env.reset()
    score = 0

    agent.before_episode()

    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break

    scores.append(score)  # save most recent score
    if e % 10 == 0:
        print(f'Episode {e} Average Score: {np.mean(scores[-100:])}')

agent.save(save_path)

# Training Results
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode')
plt.show()
