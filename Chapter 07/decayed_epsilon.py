import matplotlib.pyplot as plt

epsilon_original = .3
decay_rate = .9995
train_episodes = 5000
min_threshold = .05

epsilon_history = []
for i in range(0, train_episodes):
    epsilon = max(round(epsilon_original * pow(decay_rate, i), 10), min_threshold)
    epsilon_history.append(epsilon)
    print(f'Episode: {i + 1}| Epsilon: {epsilon}')

plt.plot(epsilon_history)
plt.xlabel('Episodes')
plt.ylabel('Epsilon')
plt.show()
