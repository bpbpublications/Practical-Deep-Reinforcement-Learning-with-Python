import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from ch8.pt.mnist.model import MnistClassifier
from ch8.utils import mnist_dataset

# Making Script Reproducible
torch.manual_seed(0)

# Initializing Classifier Model
mnist_clf = MnistClassifier()

# Preparing Datasets
(x_train, y_train), (x_test, y_test) = mnist_dataset()
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).long()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).long()

# Permute dimensions for PyTorch Convolutions
x_train = torch.permute(x_train, (0, 3, 1, 2))
x_test = torch.permute(x_test, (0, 3, 1, 2))

dataset_size = x_train.shape[0]

# Initializing Optimizer
optimizer = optim.Adam(mnist_clf.parameters(), lr = 0.001)

# Initializing Loss Function
loss = torch.nn.CrossEntropyLoss()

# Enabling Training Mode
mnist_clf.train()

batch_size = 512

for epoch in range(1, 10 + 1):

    # Random Permutations for Batches
    permutation = torch.randperm(dataset_size)

    for bi in range(1, dataset_size, batch_size):

        # Indices for new batch
        indices = permutation[bi:bi + batch_size]

        # New Batch
        batch_x, batch_y = x_train[indices], y_train[indices]

        # Resetting optimizer gradient to zero
        optimizer.zero_grad()

        # Executing Classifier Model
        # output.shape = (batch_size, 10)
        output = mnist_clf(batch_x)

        # Computing error
        error = loss(output, batch_y)

        # Adjusting Model Weights
        error.backward()
        optimizer.step()

        # Epoch State %
        print(f'Epoch: {epoch}: {round((bi / dataset_size ) * 100, 2)}%')

    # Epoch metrics report

    # Model output contains probabilities for each class on whole train dataset
    # output.shape = (60000, 10)
    output = mnist_clf(x_train)

    # predict tensor contains classes with highest probability
    # predict.shape = (60000, 1)
    predict = output.argmax(dim = 1, keepdim = True)

    # Computing accuracy between predicted and correct answers for whole train dataset
    accuracy = round(accuracy_score(predict, y_train), 4)

    print(f'Epoch: {epoch}| Accuracy: {accuracy}')

# Testing Pre-trained Classifier Model on Test Dataset

# Resetting model to evaluation mode
mnist_clf.eval()

# Model output contains probabilities for each class for all 10 000 test images
# output.shape = (10000, 10)
output = mnist_clf(x_test)

# predict tensor contains classes with highest probability
# predict.shape = (10000, 1)
predict = output.argmax(dim = 1, keepdim = True)

# Computing accuracy between predicted and correct answers
accuracy = round(accuracy_score(predict, y_test), 4)

# Final Test Accuracy
print(f'Test Accuracy: {accuracy}')
