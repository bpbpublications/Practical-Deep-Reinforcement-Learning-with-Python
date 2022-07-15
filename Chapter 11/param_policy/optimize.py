import numpy as np
import random
import matplotlib.pyplot as plt
from ch11.param_policy.dataset import cats_and_dogs_dataset
from ch11.param_policy.policy import DogVsCatPolicy

seed = 1
np.random.seed(seed)
random.seed(seed)

step_num = 21
a_list = np.linspace(-5, 5, num = step_num)
b_list = np.linspace(-5, 5, num = step_num)

agent = DogVsCatPolicy()

dataset = cats_and_dogs_dataset
accuracy_map = {}
for a in a_list:
    for b in b_list:
        accuracy = 0
        agent.a = a
        agent.b = b
        for el in dataset:
            action = agent.act_sample(el[0])
            if action == el[1]:
                accuracy += 1
        accuracy_map[(a, b)] = accuracy

# Displaying Results
best_params = max(accuracy_map, key = accuracy_map.get)
print(f'Best params: a={best_params[0]}, b={best_params[1]}')

z = np.array([accuracy_map[(a, b)] for a in a_list for b in b_list])
Z = z.reshape(step_num, step_num)

plt.title(f'Best params: a={best_params[0]}, b={best_params[1]}')
plt.xlabel('b')
plt.ylabel('a')
plt.imshow(Z, extent = [-5, 5, -5, 5], cmap = 'gray')
plt.colorbar()
plt.show()

# Visualizing the best policy
agent.a = best_params[0]
agent.b = best_params[1]
w_list = range(0, 20)
is_dog_prob = [agent.act(w)[1] for w in w_list]
plt.title('Dog VS Cat Policy')
plt.xlabel('Weight')
plt.ylabel('Dog Probability')
plt.grid()
plt.plot(is_dog_prob)
plt.show()
