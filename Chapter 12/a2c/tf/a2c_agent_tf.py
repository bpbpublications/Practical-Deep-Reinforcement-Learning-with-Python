import numpy as np
import tensorflow as tf
from tensorflow.python.ops.distributions.categorical import Categorical
from tensorflow.keras.optimizers import Adam
from ch12.a2c.tf.actor import TfActor
from ch12.a2c.tf.critic import TfCritic


class TfA2CAgent:
    """
    TensorFlow Implementation of Advantage Actor-Critic Model
    """

    def __init__(self, state_size, action_size, lr = 0.001) -> None:
        super().__init__()
        self.actor = TfActor(action_size)
        self.critic = TfCritic()

        # Actor and Critic Optimizers
        self.actor_optim = Adam(learning_rate = lr)
        self.critic_optim = Adam(learning_rate = lr)

        # Discount rate
        self.gamma = .99

    def action_sample(self, state):
        """
        Sampling action
        """
        prob = self.actor(np.array([state]))
        dist = Categorical(probs = prob, dtype = tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def update(self, final_state, buffer):
        # Unzipping episode experience
        rewards, states, actions, dones = buffer.unzip()

        # Calculating discounted cumulative rewards
        final_value = self.critic(np.array([final_state]))
        sum_reward = final_value
        discnt_rewards = []
        for step in reversed(range(len(rewards))):
            sum_reward = rewards[step] + self.gamma * sum_reward * (1 - dones[step])
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()
        discnt_rewards = tf.concat(discnt_rewards, 0)

        # Calculating Advantage
        with tf.GradientTape() as critic_tape:
            values = self.critic(np.array(states), training = True)
            advantage = discnt_rewards - values
            critic_loss = tf.reduce_mean(tf.pow(advantage, 2))

        # Calculating Gradient
        with tf.GradientTape() as actor_tape:
            prob = self.actor(np.array(states), training = True)
            dist = Categorical(probs = prob, dtype = tf.float32)
            log_prob = dist.log_prob(actions)
            E = log_prob * advantage
            actor_loss = -E

        # Executing gradient shift: theta = theta + a x Gradient
        actor_g = actor_tape.gradient(actor_loss, self.actor.variables)
        self.actor_optim.apply_gradients(zip(actor_g, self.actor.variables))

        # Optimizing Critic with MSE loss
        critic_g = critic_tape.gradient(critic_loss, self.critic.variables)
        self.critic_optim.apply_gradients(zip(critic_g, self.critic.variables))

        # returns `advantage` for debug purposes
        return advantage.numpy()
