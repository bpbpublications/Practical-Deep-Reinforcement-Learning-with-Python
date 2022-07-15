import os
import random
from time import sleep

import numpy as np
import gym
from ch10.dqn.pt.ddqn_conv_agent import DdqnConvPtAgent
from ch10.dqn.screen_motion import ScreenMotion
from ch10.utils import image_rgb_to_grayscale

cwd = os.path.dirname(os.path.abspath(__file__))

env = gym.make('Atlantis-v0')
action_size = env.action_space.n

seed = 2
random.seed(seed)
env.seed(seed)

# DDQN Agent
ddqn_agent = DdqnConvPtAgent(
    frames = ScreenMotion.frame_number,
    action_size = action_size
)
load_path = cwd + '/saved_models/dqn_pt_agent_saved.pth'
ddqn_agent.load(load_path)


# Random Agent
def random_agent():
    return random.randint(0, action_size - 1)


ddqn_score_list = []
random_score_list = []
episodes = 10

for e in range(1, episodes + 1):
    state_rgb = env.reset()
    state = image_rgb_to_grayscale(state_rgb)
    score = 0
    motion = ScreenMotion()
    i = 0
    while True:
        i += 1

        env.render()
        sleep(.005)

        # Release button each odd frame
        if i % 2 != 0 or not motion.is_full():
            action = 0  # none
        else:
            action = ddqn_agent.act(motion.get_frames(), mode = 'test')

        next_state_rgb, reward, done, _ = env.step(action)

        next_state = image_rgb_to_grayscale(next_state_rgb, crop_down = 80)
        motion.add(next_state)
        state = next_state
        score += reward

        if done:
            print(f'DDQN. Round {e}: {score}')
            ddqn_score_list.append(score)
            break

for e in range(1, episodes + 1):
    env.reset()
    score = 0
    while True:

        env.render()
        sleep(.005)

        action = random_agent()
        next_state_rgb, reward, done, _ = env.step(action)
        score += reward
        if done:
            print(f'Random. Round {e}: {score}')
            random_score_list.append(score)
            break

print(f'Average DDQN Score: {np.mean(ddqn_score_list)}')
print(f'Average Random Score: {np.mean(random_score_list)}')
