import tensorflow as tf


def get_function(x1_val = 0, x2_val = 0, x3_val = 0, x4_val = 0):
    # variables
    x1 = tf.Variable(x1_val, dtype = tf.float32)
    x2 = tf.Variable(x2_val, dtype = tf.float32)
    x3 = tf.Variable(x3_val, dtype = tf.float32)
    x4 = tf.Variable(x4_val, dtype = tf.float32)

    # function
    p1 = tf.math.pow(x1, 3)
    m1 = tf.math.multiply(p1, x2)
    m2 = tf.math.multiply(x3, x4)
    f = tf.math.add(m1, m2)

    vars = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4}

    return f, vars


if __name__ == '__main__':
    f, _ = get_function(2, 4, 3, 5)
    print(f.numpy())
