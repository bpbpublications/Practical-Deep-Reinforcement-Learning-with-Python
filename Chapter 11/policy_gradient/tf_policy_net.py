import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.distributions.categorical import Categorical


class TfPolicyNet(tf.keras.Model):

    def __init__(self, action_space):
        super().__init__()
        self.linear1 = Dense(32, activation = 'relu')
        self.linear2 = Dense(action_space, activation = 'softmax')
        self.opt = Adam()

    def call(self, x):
        x = tf.convert_to_tensor(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def act_sample(self, state):
        # probabilities for each action
        prob = self(np.array([state]))
        # distribution
        dist = Categorical(probs = prob, dtype = tf.float32)
        # sampling random action from distribution 'dist'
        action = dist.sample()
        return int(action.numpy()[0])

    def update(self, buffer, gamma):
        """
        Updates Neural Network using Policy Gradient method
        """

        # unpacks episode history
        rewards, actions, states = buffer.unzip()

        # Counting discounted rewards
        sum_reward = 0
        discnt_rewards = []
        rewards.reverse()
        for r in rewards:
            sum_reward = r + gamma * sum_reward
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()

        with tf.GradientTape() as tape:

            # Calculating Gradient
            prob = self(np.array(states), training = True)
            dist = Categorical(probs = prob, dtype = tf.float32)
            log_prob = dist.log_prob(actions)
            E = log_prob * discnt_rewards

            # Since our goal is to maximize E and optimizers are made for
            # minimization, we are changing the sign of E value
            loss = -E

        # Executing gradient shift: theta = theta + a x Gradient
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
