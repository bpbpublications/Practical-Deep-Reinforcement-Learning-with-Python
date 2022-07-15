import numpy as np
import torch
import torch.optim as optim
from ch9.dqn.base_dqn_agent import BaseDqnAgent
from ch9.dqn.pt.q_model import PtQNet


class DqnPtAgent(BaseDqnAgent):

    def __init__(self, *args, **kwargs):
        super(DqnPtAgent, self).__init__(*args, **kwargs)

    def init_q_net(self, state_size, action_size, learning_rate):
        self.q_net = PtQNet(state_size, action_size)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr = learning_rate)
        self.loss = torch.nn.MSELoss()

    def greedy_act(self, state):
        # Greedy Policy
        state = torch.from_numpy(state).float().unsqueeze(0)

        self.q_net.eval()

        with torch.no_grad():  # disabling gradient computation
            action_values = self.q_net(state)
        self.q_net.train()

        action = np.argmax(action_values.data.numpy())
        return action

    def learn(self):
        samples = self.memory.batch()
        s, a, r, s_next, dones = samples

        # to PyTorch Tensors
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).long()
        r = torch.from_numpy(r).float()
        s_next = torch.from_numpy(s_next).float()
        dones = torch.from_numpy(dones).float()

        # V(s') = max(Q(s',a))
        v_s_next = self.q_net(s_next).detach().max(1)[0].unsqueeze(1)

        # Q(s,a)
        q_sa_pure = self.q_net(s)
        q_sa = q_sa_pure.gather(dim = 1, index = a)

        # TD = r + g * V(s') - Q(s,a)
        td = r + (self.gamma * v_s_next * (1 - dones)) - q_sa

        # Compute loss: TD -> 0
        error = self.loss(td, torch.zeros(td.shape))

        self.optimizer.zero_grad()
        error.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
