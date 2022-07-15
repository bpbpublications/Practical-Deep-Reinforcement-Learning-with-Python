import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def create_grid(low, high, bins = (10, 10)):
    g = [
        np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1]
        for dim in range(len(bins))
    ]
    return g


def discretize(v, grid):
    return list(int(np.digitize(s, g)) for s, g in zip(v, grid))


def plot_route(bins, states, labels, title = None):
    nrows = bins[0]
    ncols = bins[1]
    image = np.zeros(nrows * ncols)
    image = image.reshape((nrows, ncols))

    plt.figure(figsize = (round(nrows / 1.5), round(ncols / 1.5)), dpi = 240)

    for s in states:
        if image[s[1], s[0]] == 0:
            image[s[1], s[0]] += .5
        else:
            image[s[1], s[0]] += .1

    plt.matshow(image, cmap = 'binary', fignum = 1)
    if title:
        plt.title(title, pad = 12)

    ax = plt.gca()
    ax.set_xticks(np.arange(0, ncols, 1))
    ax.set_yticks(np.arange(0, nrows, 1))

    ax.set_xticklabels(np.arange(0, ncols, 1))
    ax.set_yticklabels(np.arange(0, nrows, 1))

    ax.set_xticks(np.arange(-.5, ncols, 1), minor = True)
    ax.set_yticks(np.arange(-.5, nrows, 1), minor = True)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    ax.grid(which = 'minor', linestyle = '-', linewidth = 2)
    plt.show()


def plot_q_map(q, labels, title = None):
    nrows = q.shape[0]
    ncols = q.shape[1]

    plt.figure(figsize = (round(nrows / 1.5), round(ncols / 1.5)), dpi = 240)

    for x in range(1, ncols + 1):
        for y in range(1, nrows + 1):
            for a_idx in range(q.shape[2]):
                is_max = np.argmax(q[x - 1, y - 1]) == a_idx
                if is_max:
                    arrow_color = 'green'
                    arrow_len = .04
                else:
                    arrow_color = 'pink'
                    arrow_len = .03

                q_v = round(q[x - 1, y - 1, a_idx])
                if q_v == 0:
                    continue

                if a_idx == 1:
                    arrow_x, arrow_y = 0, arrow_len
                elif a_idx == 2:
                    arrow_x, arrow_y = arrow_len, 0
                elif a_idx == 0:
                    arrow_x, arrow_y = -arrow_len, 0

                plt.arrow(x, y, arrow_x, arrow_y, width = arrow_len, color = arrow_color)

    ax = plt.gca()
    ax.set_xticks(np.arange(0, ncols, 1))
    ax.set_yticks(np.arange(0, nrows, 1))

    ax.set_xticklabels(np.arange(0, ncols, 1))
    ax.set_yticklabels(np.arange(0, nrows, 1))

    ax.set_xticks(np.arange(-.5, ncols, 1), minor = True)
    ax.set_yticks(np.arange(-.5, nrows, 1), minor = True)

    ax.grid(which = 'minor', linestyle = '-', linewidth = 2)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    if title:
        plt.title(title, pad = 12)
    plt.show()


def plot_rolling(data_list, rolling_window = 100, title = ""):
    plt.plot(data_list)
    plt.title(title)
    rolling_mean = pd.Series(data_list).rolling(rolling_window).mean()
    plt.plot(rolling_mean, label = f'Last {rolling_window} Average')
    plt.legend()
    plt.show()
