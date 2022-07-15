# import dependencies
import matplotlib.pyplot as plt
import gym
from ch12.a2c.pt.a2c_agent_pt import PtA2CAgent
from ch12.a2c.tf.a2c_agent_tf import TfA2CAgent

# create environment
from ch12.utils import Buffer

env = gym.make("CartPole-v1")

# instantiate the policy
# TensorFlow Implementation
# policy = TfA2CAgent(env.action_space.n)

# PyTorch Implementation
policy = PtA2CAgent(env.observation_space.shape[0], env.action_space.n)

# initialize gamma and stats
gamma = 0.99
episodes = 1_000
total_scores = []
avg_scores = []
buffer = Buffer()

for e in range(1, episodes + 1):
    buffer.clear()
    # reset environment
    state = env.reset()
    epoch_rewards = 0
    while True:
        action = policy.action_sample(state)
        # use that action in the environment
        new_state, reward, done, info = env.step(action)
        epoch_rewards += reward
        # store state, action, reward and done
        buffer.add(reward, state, action, done)

        state = new_state
        if done:
            total_scores.append(epoch_rewards)
            break

    advantage = policy.update(state, buffer)

    if e % 100 == 0:
        plt.plot(advantage)
        plt.axhline(y = 0, color = 'r', linestyle = '-')
        plt.title(f'Advantage Actor Critic. Advantage.\nEpisode: {e}')
        plt.ylabel('Advantage (G - V(s))')
        plt.xlabel('State Number (St)')
        plt.show()

# close environment
env.close()
