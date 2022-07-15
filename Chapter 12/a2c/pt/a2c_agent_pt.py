import torch
import torch.optim as optim
from torch.distributions import Categorical
from ch12.a2c.pt.actor import PtActor
from ch12.a2c.pt.critic import PtCritic


class PtA2CAgent:
    """
    PyTorch Implementation of Advantage Actor-Critic Model
    """

    def __init__(self, state_size, action_size) -> None:
        super().__init__()
        self.actor = PtActor(state_size, action_size)
        self.critic = PtCritic(state_size, action_size)

        # Actor and Critic Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters())
        self.critic_optim = optim.Adam(self.critic.parameters())

        # Discount rate
        self.gamma = .99

    def action_sample(self, state):
        """
        Sampling action
        """
        state = torch.FloatTensor(state)
        dist = Categorical(self.actor(state))
        action = dist.sample().item()
        return action

    def update(self, final_state, buffer):
        # Unzipping episode experience
        rewards, states, actions, dones = buffer.unzip()

        # Converting to tensors
        states = torch.tensor(states).float()
        actions = torch.tensor(actions)

        # Calculating discounted cumulative rewards
        final_value = self.critic(torch.FloatTensor(final_state))

        sum_reward = final_value
        discnt_rewards = []
        for step in reversed(range(len(rewards))):
            sum_reward = rewards[step] + self.gamma * sum_reward * (1 - dones[step])
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()

        # Calculating Advantage
        discnt_rewards = torch.cat(discnt_rewards).detach()
        values = self.critic(states).squeeze(1)
        advantage = discnt_rewards - values

        # Calculating Gradient
        probs = self.actor(states)
        sampler = Categorical(probs)

        # log(Pi(at|st))
        log_probs = sampler.log_prob(actions)
        E = (log_probs * advantage.detach()).mean()

        # Since our goal is to maximize E and optimizers are made for
        # minimization, we are changing the sign of E value
        actor_loss = -E

        # Executing gradient shift: theta = theta + a x Gradient
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Optimizing Critic with MSE loss
        critic_loss = advantage.pow(2).mean()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # returns `advantage` for debug purposes
        return advantage.detach().numpy()
