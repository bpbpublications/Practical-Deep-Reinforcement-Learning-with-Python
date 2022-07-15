import torch.nn as nn
import torch.nn.functional as F


class PtActor(nn.Module):

    def __init__(self, state_size, action_size):
        super(PtActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = F.softmax(x, dim = -1)
        return x
