import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from ch12.a2c.tf.a2c_agent_tf import TfA2CAgent
from ch12.a2c.pt.a2c_agent_pt import PtA2CAgent
from ch12.trading_env.stock_trading_env import StockTradingEnv
from ch12.utils import Buffer

quotes = yf.download('MSFT', start = '2015-01-01', end = '2022-01-01')
close_quotes = quotes['Close'].values

state_size = 10
# 0 - sell, 1 - none, 2 - buy
action_size = 3

env = StockTradingEnv(
    close_quotes,
    name = 'MSFT. A2C',
    state_size = state_size
)


def normalize_state(state):
    return np.array(state) / state[-1]


episodes = 500
avg_period = 50
avg = []
total_rewards = []

# TensorFlow Implementation
a2c_agent = TfA2CAgent(state_size, action_size, lr = 0.0005)

# PyTorch Implementation
# a2c_agent = PtA2CAgent(state_size, action_size)

buffer = Buffer()
for episode in range(1, episodes + 1):
    epoch_rewards = 0
    state = normalize_state(env.reset())
    buffer.clear()

    while True:
        action = a2c_agent.action_sample(state)
        next_state_dirty, reward, done, _ = env.step(action)
        epoch_rewards += reward

        buffer.add(reward, state, action, done)
        state = normalize_state(next_state_dirty)

        if done:
            total_rewards.append(epoch_rewards)
            break

    a2c_agent.update(state, buffer)

    avg.append(np.mean(total_rewards[-min(avg_period, episode):]))

    if episode % avg_period == 0:
        print(f'Episode: {episode}. Average Score: {avg[-1]}')

env.close()

plt.plot(avg)
plt.axhline(y = 0, color = 'r', linestyle = '-')
plt.xlabel('Episodes')
plt.ylabel('Average Rewards')
plt.title('Stock Trading. A2C')
plt.show()
