import tensorflow as tf
from sklearn.metrics import accuracy_score
from ch8.tf.mnist.model import MnistClassifier
from ch8.utils import mnist_dataset
import numpy as np

# Making Script Reproducible
tf.random.set_seed(0)

# Initializing Classifier Model
mnist_clf = MnistClassifier()

# Preparing Datasets
(x_train, y_train), (x_test, y_test) = mnist_dataset()
dataset_size = x_train.shape[0]

# Building Model Computation Graph
mnist_clf.build(input_shape = x_train.shape)

# Extracting Model Weights
variables = mnist_clf.variables

# Initializing Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

# Initializing Loss Function
loss = tf.keras.losses.CategoricalCrossentropy()

batch_size = 512

for epoch in range(1, 10 + 1):

    # Random Permutations for Batches
    permutation = np.random.permutation(dataset_size)

    for bi in range(1, dataset_size, batch_size):

        # Indices for new batch
        indices = permutation[bi:bi + batch_size]

        # New Batch
        batch_x, batch_y = x_train[indices], y_train[indices]

        # Converting value tensor to one_hot tensor:
        # [4] => [0 0 0 0 1 0 0 0 0 0]
        batch_y_one_hot = tf.one_hot(batch_y, depth = 10)

        # Registering derivatives
        with tf.GradientTape() as tape:

            # Executing Classifier Model
            # output.shape = (batch_size, 10)
            output = mnist_clf(batch_x)

            # Computing error
            error = loss(batch_y_one_hot, output)

        # Adjusting Model Weights
        gradient = tape.gradient(error, variables)
        optimizer.apply_gradients(zip(gradient, variables))

        # Epoch State %
        print(f'Epoch: {epoch}: {round((bi / dataset_size ) * 100, 2)}%')

    # Epoch metrics report

    # Model output contains probabilities for each class on whole train dataset
    # output.shape = (60000, 10)
    output = mnist_clf(x_train)

    # predict tensor contains classes with highest probability
    # predict.shape = (60000, 1)
    predict = tf.math.argmax(output, axis = 1)

    # Computing accuracy between predicted and correct answers for whole train dataset
    accuracy = round(accuracy_score(predict, y_train), 4)

    print(F'Epoch: {epoch}| Accuracy: {accuracy}')

# Testing Pre-trained Classifier Model on Test Dataset
# Model output contains probabilities for each class for all 10 000 test images
# output.shape = (10000, 10)
output = mnist_clf(x_test)

# predict tensor contains classes with highest probability
# predict.shape = (10000, 1)
predict = tf.math.argmax(output, axis = 1)

# Computing accuracy between predicted and correct answers
accuracy = round(accuracy_score(predict, y_test), 4)

# Final Test Accuracy
print(f'Test Accuracy: {accuracy}')
