import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


def thompson_sampling_policy(state, visualize = False, plot_title = ''):
    action = None
    max_bound = 0

    color_list = ['red', 'blue', 'green', 'black', 'yellow']

    for b, trials in state.items():
        w = len([r for r in trials if r == 1])
        l = len([r for r in trials if r == -1])

        if w + l == 0:
            avg = 0
        else:
            avg = round(w / (w + l), 2)

        random_beta = np.random.beta(w + 1, l + 1)
        if random_beta > max_bound:
            max_bound = random_beta
            action = b

        if visualize:
            color = color_list[b % len(color_list)]
            x = np.linspace(beta.ppf(0.01, w, l), beta.ppf(0.99, w, l), 100)
            plt.plot(
                x, beta.pdf(x, w, l),
                label = f'Bandit {b}| avg={avg}, v={round(random_beta,2)}',
                color = color, linewidth = 3)
            plt.axvline(x = random_beta, color = color, linestyle = '--')

    if visualize:
        plt.title('Thompson Sampling: Beta Distribution. ' + plot_title)
        plt.legend()
        plt.show()

    return action
