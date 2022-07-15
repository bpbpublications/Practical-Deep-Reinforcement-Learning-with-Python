from time import sleep
from ch12.trading_env.stock_trading_env import StockTradingEnv
import yfinance as yf
import random

quotes = yf.download('MSFT', start = '2015-1-1', end = '2022-1-1')
close_quotes = quotes['Close'].values

random.seed(1)

state_size = 10
# 0 - sell, 1 - none, 2 - buy
action_size = 3

env = StockTradingEnv(
    close_quotes,
    name = 'MSFT. Random Agent',
    state_size = state_size
)
init_state = env.reset()

total = 0
while True:
    env.render()
    action = random.randint(0, action_size)
    state, reward, done, debug = env.step(action)
    sleep(.3)
    total += reward
    if done:
        break

env.close()

print(f'Total Reward: {total}')
