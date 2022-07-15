from ch7.utils import discretize, create_grid

if __name__ == '__main__':
    g = create_grid([0, 0], [200, 150], bins = [4, 3])
    v = [87.02, 105.61]
    dv = discretize(v, g)
    print(dv)
