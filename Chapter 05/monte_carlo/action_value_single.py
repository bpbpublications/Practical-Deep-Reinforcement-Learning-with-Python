import random

import gym

seed = 56
random.seed(seed)

a_map = {
    1: 'Hit',
    0: 'Stick'
}

episodes = 100

env = gym.make('Blackjack-v0')
env.seed(seed)

reward_sum = [0, 0]
a_number = [0, 0]

e = 0
while True:
    state = env.reset()
    agent_sum = state[0]
    dealer_sum = state[1]
    usable_ace = state[2]

    if agent_sum != 20 or dealer_sum != 8 or usable_ace != False:
        continue

    action = random.randint(0, 1)
    next_state, reward, done, _ = env.step(action)

    if done:
        a_number[action] += 1
        reward_sum[action] += reward
        e += 1
        print(f'{e}: {a_number[action]}, {state}, {a_map[action]}, {reward}')

    if e == episodes:
        break

print(f'Stick | rewards:{reward_sum[0]}, trials:{a_number[0]}')
print(f'Hit | rewards:{reward_sum[1]}, trials:{a_number[1]}')

print(f'Q({state}, {a_map[0]})={round(reward_sum[0] / a_number[0], 2)}')
print(f'Q({state}, {a_map[1]})={round(reward_sum[1] / a_number[1], 2)}')
