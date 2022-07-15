import tensorflow as tf

player_characteristics = tf.constant([
    [9, 7, 9],  # Player 1
    [10, 9, 10],  # Player 2
    [8, 10, 8]  # Player 3
], dtype = tf.float32)

# Total Characteristics
player_total = tf.reduce_sum(player_characteristics, axis = 1)

softmax = tf.nn.softmax

player_prob = softmax(player_total)

# Probabilities
print(player_prob.numpy())
