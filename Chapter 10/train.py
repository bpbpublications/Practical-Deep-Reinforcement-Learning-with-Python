import os
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from ch10.dqn.screen_motion import ScreenMotion
from ch10.dqn.pt.ddqn_conv_agent import DdqnConvPtAgent
from ch10.dqn.tf.ddqn_conv_agent import DdqnConvTfAgent
from ch10.utils import image_rgb_to_grayscale

cwd = os.path.dirname(os.path.abspath(__file__))

env = gym.make('Atlantis-v0')
action_size = env.action_space.n
seed = 1

random.seed(seed)
env.seed(seed)

# TensorFlow Implementation
agent = DdqnConvTfAgent(
    frames = ScreenMotion.frame_number,
    action_size = action_size,
    update_period = 5,
    train_batch_size = 64,
    degp_min_epsilon = .1,
    degp_epsilon = .3,
    degp_decay_rate = .9,
    learning_rate = .0001,
    replay_buffer_size = 3_000
)
save_path = os.path.abspath(__file__) + '/saved_models/dqn_tf_agent.tfh'

# PyTorch Implementation
# agent = DdqnConvPtAgent(
#     frames = ScreenMotion.frame_number,
#     action_size = action_size,
#     update_period = 5,
#     train_batch_size = 64,
#     degp_min_epsilon = .1,
#     degp_epsilon = .3,
#     degp_decay_rate = .9,
#     learning_rate = .0001,
#     replay_buffer_size = 3_000
# )
# save_path = cwd + '/saved_models/dqn_pt_agent.pth'

# Training
training_episodes = 10
scores = []
avg_scores = []

last_best_average = 0

for e in range(1, training_episodes + 1):

    state_rgb = env.reset()
    # converting image
    state = image_rgb_to_grayscale(state_rgb, crop_down = 80)
    score = 0

    agent.before_episode()

    motion = ScreenMotion()

    # New Episode
    while True:

        if not motion.is_full():
            action = 0  # none
        else:
            action = agent.act(motion.get_frames())

        next_state_rgb, reward, done, _ = env.step(action)
        # converting image
        next_state = image_rgb_to_grayscale(next_state_rgb, crop_down = 80)

        # Adding image to motion frames
        motion.add(next_state)

        # If motion is full then perform agent.step
        if motion.is_full():
            agent.step(
                motion.get_prev_frames(),
                action,
                reward,
                motion.get_frames(),
                done
            )

        state = next_state
        score += reward
        if done:
            break

    scores.append(score)
    avg_scores.append(np.mean(scores[-min(10, len(scores)):]))

    print(f'Episode: {e}. Score: {round(score)}. '
          f'Avg Score: {round(avg_scores[-1])}')

# Training Results
plt.plot(scores)
plt.plot(avg_scores, label = 'Average Score')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.plot()
plt.show()

# Saving the agent
agent.save(save_path)
