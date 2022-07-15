import random
import numpy as np
import matplotlib.pyplot as plt


class DogVsCatPolicy:

    def __init__(self) -> None:
        """
        a, b - policy parameters
        """
        super().__init__()
        self.a = .1
        self.b = -.5

    def act(self, weight):
        """
        Returns [cat_probability, dog_probability] for given weight
        """
        dog_p = max(0, min(1, self.a * weight + self.b))
        cat_p = 1 - dog_p
        return [cat_p, dog_p]

    def act_sample(self, weight):
        # samples real action by given probabilities
        probs = self.act(weight)
        action = np.random.choice([0, 1], p = [probs[0], probs[1]])
        return action


if __name__ == '__main__':

    agent = DogVsCatPolicy()

    # Evaluating the Policy
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    w = 7
    action = agent.act_sample(w)
    if action == 0:
        print('Cat')
    else:
        print('Dog')
    # Policy Visualization

    w_list = range(0, 20)
    is_dog_prob = [agent.act(w)[1] for w in w_list]
    plt.title('Dog VS Cat Policy')
    plt.xlabel('Weight')
    plt.ylabel('Dog Probability')
    plt.grid()
    plt.plot(is_dog_prob)
    plt.show()
