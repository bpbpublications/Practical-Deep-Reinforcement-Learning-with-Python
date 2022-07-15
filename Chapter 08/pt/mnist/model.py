from torch import nn


class MnistClassifier(nn.Module):

    def __init__(self):
        super(MnistClassifier, self).__init__()

        # 1st Convolution Layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 5)

        # Max Pool Layer
        self.max_pool = nn.MaxPool2d(2)

        # Dropout Layer
        self.dropout = nn.Dropout(p = .5)

        # ReLU Activation
        self.relu = nn.ReLU()

        # 2nd Convolution Layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5)

        # Flatten Layer
        self.flatten = nn.Flatten()

        # 1st Linear Layer
        self.lin1 = nn.Linear(1024, 256)

        # 2nd Linear Layer
        self.lin2 = nn.Linear(256, 10)

        # SoftMax Activation
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return self.softmax(x)
