import torch
import torch.nn as nn
import torch.nn.functional as F


class PtQConvNet(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, frames, action_size):
        """Initialize parameters and build model.
        """
        super(PtQConvNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = frames,
            out_channels = 8,
            kernel_size = 7
        )
        self.conv2 = nn.Conv2d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = 5
        )

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(2880, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = F.max_pool2d(x, 3)
        x = torch.relu(self.conv2(x))
        x = F.max_pool2d(x, 3)

        x = self.flat(x)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
