import matplotlib.pyplot as plt
import numpy as np


def plot_q(Q):
    dealer_min = 4
    player_min = 12
    dealer_axis = range(dealer_min, 11)
    player_axis = range(player_min, 22)
    usable_ace_data = np.zeros((len(dealer_axis), len(player_axis)))
    no_usable_ace_data = np.zeros((len(dealer_axis), len(player_axis)))

    for s, actions in Q.items():
        player_sum = s[0]
        dealer_sum = s[1]
        usable_ace = s[2]
        x = player_sum - player_min
        y = dealer_sum - dealer_min

        if x < 0 or y < 0:
            continue

        s = -1 if actions[0] > actions[1] else 1
        if usable_ace:
            usable_ace_data[y, x] = s
        else:
            no_usable_ace_data[y, x] = s

    fig, (axs_1, axs_2) = plt.subplots(1, 2, constrained_layout = True)
    fig.set_size_inches(10, 5)

    axs_1.matshow(usable_ace_data, cmap = 'bwr', vmin = -1, vmax = 1)
    axs_2.matshow(no_usable_ace_data, cmap = 'bwr', vmin = -1, vmax = 1)

    axs_1.xaxis.set(ticks = np.arange(0, len(player_axis)), ticklabels = player_axis)
    axs_1.yaxis.set(ticks = np.arange(0, len(dealer_axis)), ticklabels = dealer_axis)
    axs_1.set_xlabel('Player', fontsize = 24)
    axs_1.xaxis.set_ticks_position("bottom")
    axs_1.set_ylabel('Dealer', fontsize = 24)
    axs_1.set_title("Usable Ace", fontsize = 24)

    axs_2.xaxis.set(ticks = np.arange(0, len(player_axis)), ticklabels = player_axis)
    axs_2.yaxis.set(ticks = np.arange(0, len(dealer_axis)), ticklabels = dealer_axis)
    axs_2.set_xlabel('Player', fontsize = 24)
    axs_2.xaxis.set_ticks_position("bottom")
    axs_2.set_ylabel('Dealer', fontsize = 24)
    axs_2.set_title("No Usable Ace", fontsize = 24)

    plt.show()

    x = 'aga'
