import random

import matplotlib.pyplot as plt
import numpy as np
from ch4.gym.multiarmed_bandit_env import get_bandit_env_5
from ch4.run.run_e_greedy_policy import run_e_greedy_policy
from ch4.run.run_thompson_sampling import run_thompson_sampling

seed = 0
random.seed(seed)
np.random.seed(seed)

episodes = 50
balance = 1_000

env_gen = get_bandit_env_5
rewards = {
    'Epsilon Greedy':    [np.cumsum(run_e_greedy_policy(balance, env_gen())[1]) for _ in range(episodes)],
    'Thompson Sampling': [np.cumsum(run_thompson_sampling(balance, env_gen())[1]) for _ in range(episodes)],
}

for policy, r in rewards.items():
    plt.plot(np.average(r, axis = 0), label = policy)

plt.legend()
plt.xlabel("Rounds")
plt.ylabel("Average Returns")
plt.title('Battle')
plt.show()
