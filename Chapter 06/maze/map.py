def map1():
    x_coord = ['A', 'B', 'C', 'D']
    y_coord = [1, 2, 3, 4, 5]
    walls = ['B3', 'B4', 'C2', 'C4', 'D2']
    goal = 'C3'
    return x_coord, y_coord, walls, goal


def map2():
    x_coord = ['A', 'B', 'C', 'D', 'E', 'F',
               'G', 'H', 'I', 'J', 'K', 'L']
    y_coord = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    walls = [
        'B3', 'B4', 'B6', 'B7', 'B8', 'B9', 'B11',
        'C2', 'C4', 'C11',
        'D2', 'D6', 'D7', 'D8', 'D11',
        'E3', 'E7', 'E8', 'E10', 'E11',
        'F4', 'F5', 'F6', 'F9', 'F10',
        'G7', 'G10', 'G12',
        'H4', 'H7', 'H8', 'H11',
        'I3', 'I5', 'I8', 'I11',
        'J1', 'J5', 'J6', 'J8',
        'K1', 'K3', 'K6', 'K7', 'K12',
    ]
    goal = 'G6'
    return x_coord, y_coord, walls, goal


if __name__ == '__main__':
    from ch6.maze.core import plot_maze_state

    x_coord, y_coord, walls, goal = map2()
    state = 'A1'
    plot_maze_state(x_coord, y_coord, walls, state, goal)
