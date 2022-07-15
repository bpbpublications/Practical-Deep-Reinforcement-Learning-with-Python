import torch.nn as nn


class PtQNet(nn.Module):
    """Q-Network. PyTorch Implementation"""

    def __init__(self, state_size, action_size, fc1_units = 64, fc2_units = 64):
        """
            state_size: Dimension of state space
            action_size: Dimension of action space
            fc1_units: First hidden layer size
            fc2_units: Second hidden layer size
        """
        super(PtQNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
