import matplotlib.pyplot as plt
import numpy as np
from ch6.maze.core import maze_states, actions


def plot_q_map(x_coord, y_coord, blocks, Q, goal, title = None):
    nrows = len(y_coord)
    ncols = len(x_coord)
    image = np.zeros(nrows * ncols)
    image = image.reshape((nrows, ncols))

    for x in range(ncols):
        for y in range(nrows):
            label = f'{x_coord[x]}{y_coord[y]}'
            if label in blocks:
                image[y, x] = 0
            elif label == goal:
                image[y, x] = .5
            else:
                image[y, x] = 1

    plt.figure(figsize = (nrows, ncols), dpi = 240)
    plt.matshow(image, cmap = 'gray', fignum = 1)

    states = maze_states(x_coord, y_coord)
    for x in range(ncols):
        for y in range(nrows):
            label = f'{x_coord[x]}{y_coord[y]}'
            state_idx = states.index(label)
            for a_idx in range(len(actions)):
                is_max = np.argmax(Q[state_idx]) == a_idx
                if is_max:
                    arrow_color = 'green'
                    font_size = 10
                    arrow_len = .04
                else:
                    arrow_color = 'pink'
                    font_size = 7
                    arrow_len = .03

                q_v = round(Q[state_idx, a_idx])
                if q_v == 0:
                    continue

                if actions[a_idx] == 'D':
                    arrow_x, arrow_y = 0, arrow_len
                    annotate_xy = (x, y + .3)
                elif actions[a_idx] == 'U':
                    arrow_x, arrow_y = 0, -arrow_len
                    annotate_xy = (x, y - .2)
                elif actions[a_idx] == 'R':
                    arrow_x, arrow_y = arrow_len, 0
                    annotate_xy = (x + .25, y)
                elif actions[a_idx] == 'L':
                    arrow_x, arrow_y = -arrow_len, 0
                    annotate_xy = (x - .4, y - .1)

                plt.annotate(q_v, xy = annotate_xy, fontsize = font_size)
                plt.arrow(x, y, arrow_x, arrow_y, width = arrow_len, color = arrow_color)

    plt.xticks(range(ncols), x_coord)
    plt.yticks(range(nrows), y_coord)
    if title:
        plt.title(title, pad = 12)
    plt.show()
