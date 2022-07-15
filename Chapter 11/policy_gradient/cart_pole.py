# import dependencies
import numpy as np
import matplotlib.pyplot as plt
import gym
from ch11.policy_gradient.pt_policy_net import PtPolicyNet
from ch11.policy_gradient.tf_policy_net import TfPolicyNet
from ch11.utils import Buffer

# create environment
env = gym.make("CartPole-v1")

# instantiate the policy
# policy = TfPolicyNet(env.action_space.n)
policy = PtPolicyNet(env.observation_space.shape[0], env.action_space.n)

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
        action = policy.act_sample(state)
        # use that action in the environment
        new_state, reward, done, info = env.step(action)
        epoch_rewards += reward
        # store state, action and reward
        buffer.add(reward, action, state)

        state = new_state
        if done:
            total_scores.append(epoch_rewards)
            break

    policy.update(buffer, gamma)

    avg_scores.append(np.mean(total_scores[-min(100, len(total_scores)):]))

    if e % 100 == 0:
        print(f'Episode: {e}. Average Score: {avg_scores[-1]}')

# close environment
env.close()

plt.plot(total_scores)
plt.plot(avg_scores, label = 'average')
plt.title('Policy Gradient. CartPole.')
plt.ylabel('Total Reward')
plt.xlabel('Episodes')
plt.show()
