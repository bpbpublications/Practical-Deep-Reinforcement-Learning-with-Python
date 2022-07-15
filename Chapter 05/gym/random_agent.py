import gym

action_map = {
    1: 'Hit',
    0: 'Stick'
}
seed = 10
episodes = 100

env = gym.make('Blackjack-v0')
env.seed(seed)

wins = 0
loss = 0
draws = 0

for e in range(1, episodes + 1):
    state = env.reset()
    print(f'===== Episode: {e} =====')
    while True:
        agent_sum = state[0]
        dealer_sum = state[1]

        if agent_sum < 18:
            # hit
            action = 1
        else:
            # stand
            action = 0

        next_state, reward, done, _ = env.step(action)
        print(f'state: {state} | action: {action_map[action]} '
              f'| reward: {reward} | next state: {next_state}')

        if done:
            if reward > 0:
                wins += 1
            elif reward < 0:
                loss += 1
            else:
                draws += 1
            break

        state = next_state

print(f'Wins: {wins} | Loss: {loss} | Draws: {draws} ')
