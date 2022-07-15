import tensorflow as tf

prob_vector = tf.constant([0.85, 0.15])

# Correct class vector
correct_vector = tf.constant([1, 0])

ce_loss = tf.keras.losses.CategoricalCrossentropy()
ce_error = ce_loss(correct_vector, prob_vector)

print(f'ce: {ce_error.numpy()}')
