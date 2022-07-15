import tensorflow_datasets as tfds


def mnist_dataset():
    ds = tfds.as_numpy(tfds.load(
        'mnist',
        batch_size = -1,
        as_supervised = True,
    ))

    (x_train, y_train) = ds['train']
    (x_test, y_test) = ds['test']
    # Normalization from [0, 255] to [0, 1] range
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    ds = tfds.as_numpy(tfds.load(
        'mnist',
        batch_size = -1,
        as_supervised = True,
    ))

    (x_train, y_train) = ds['train']
    (x_test, y_test) = ds['test']

    print(f'x_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'x_test shape: {x_test.shape}')
    print(f'y_test shape: {y_test.shape}')
