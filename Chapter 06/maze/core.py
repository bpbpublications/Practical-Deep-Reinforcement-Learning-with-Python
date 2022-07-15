import matplotlib.pyplot as plt
import random
from ch6.maze.map import map1
import numpy as np

# Up, Right, Down, Left
actions = ['U', 'R', 'D', 'L']


def plot_maze_state(x_coord, y_coord, walls, current_state, goal, title = ''):
    nrows = len(y_coord)
    ncols = len(x_coord)
    image = np.zeros(nrows * ncols)
    image = image.reshape((nrows, ncols))

    for x in range(ncols):
        for y in range(nrows):
            label = f'{x_coord[x]}{y_coord[y]}'
            if label in walls:
                image[y, x] = 0
            else:
                image[y, x] = 1

    plt.figure(figsize = (ncols, nrows), dpi = 240)
    plt.matshow(image, cmap = 'gray', fignum = 1)

    for x in range(ncols):
        for y in range(nrows):
            label = f'{x_coord[x]}{y_coord[y]}'
            if label == goal:
                plt.annotate('O', xy = (x - .2, y + .2), fontsize = 30, weight = 'bold')
            if label == current_state:
                plt.annotate('X', xy = (x - .2, y + .2), fontsize = 30, weight = 'bold')

    plt.xticks(range(ncols), x_coord)
    plt.yticks(range(nrows), y_coord)
    if title:
        plt.title(title)
    plt.show()


def move(current_state, action, x_coord, y_coord, walls):
    x = current_state[0]
    y = int(current_state[1:])
    x_idx = x_coord.index(x)
    y_idx = y_coord.index(y)

    if action == 'U':
        next_state = x + str(y_coord[max([y_idx - 1, 0])])
    elif action == 'R':
        next_state = x_coord[min([x_idx + 1, len(x_coord) - 1])] + str(y)
    elif action == 'D':
        next_state = x + str(y_coord[min([y_idx + 1, len(y_coord) - 1])])
    elif action == 'L':
        next_state = x_coord[max([x_idx - 1, 0])] + str(y)
    else:
        raise Exception(f'Invalid action: {action}')

    if next_state in walls:
        return current_state

    return next_state


def random_state(x_coord, y_coord, walls):
    available_states = []
    for cell in maze_states(x_coord, y_coord):
        if cell in walls:
            continue
        available_states.append(cell)
    return random.choice(available_states)


def maze_states(x_coord, y_coord):
    all_states = []
    for y in y_coord:
        for x in x_coord:
            cell = f'{x}{y}'
            all_states.append(cell)
    return all_states


if __name__ == '__main__':
    x_coord, y_coord, walls, goal = map1()
    state = 'A1'
    plot_maze_state(x_coord, y_coord, walls, state, goal)
    actions = ['D', 'D', 'D', 'D', 'R', 'R', 'R', 'U', 'U', 'L']
    for action in actions:
        state = move(state, action, x_coord, y_coord, walls)
        plot_maze_state(x_coord, y_coord, walls, state, goal)
