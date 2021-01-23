import gym
from gym.wrappers import Monitor
from collections import deque
import tensorflow as tf
import numpy as np
import random
import math
import time
import glob
import io
import base64

NUM_ACTIONS = 14


class DQN(tf.keras.Model):
    """Dense neural network class."""
    def __init__(self):
        super(DQN, self).__init__()
        tf.keras.layers.InputLayer( input_shape=(None,21,41,4) )
        tf.keras.layers.Conv2D( filters = 32, kernel_size = (6,6), strides=(1, 1), activation='relu')
        tf.keras.layers.Conv2D( filters = 64, kernel_size = (2,2), strides=(1, 1), activation='relu')
        tf.keras.layers.Conv2D( filters = 96, kernel_size = (2,2), strides=(1, 1), activation='relu')
        tf.keras.layers.Flatten()
        tf.keras.layers.Dense(512, activation='relu')
        tf.keras.layers.Dense(512, activation='relu')
        tf.keras.layers.Dense(NUM_ACTIONS, activation='sigmoid')

    def call(self, x):
        """Forward pass."""
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


main_nn = DQN()
target_nn = DQN()

optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()


class ReplayBuffer(object):
    """Experience replay buffer that samples uniformly."""
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)
        for i in idx:
            elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)
        return states, actions, rewards, next_states, dones


def select_epsilon_greedy_action(state, epsilon):
    """Take random action with probability epsilon, else take best action."""
    result = tf.random.uniform((1,))
    if result < epsilon:
        return env.action_space.sample()    # Random action (left or right).
    else:
        return tf.argmax(main_nn(state)[0]).numpy()
        # Greedy action for state.


@tf.function
def train_step(states, actions, rewards, next_states, dones):
    """Perform a training iteration on a batch of data sampled from the experience
    replay buffer."""
    # Calculate targets.
    next_qs = target_nn(next_states)
    max_next_qs = tf.reduce_max(next_qs, axis=-1)
    target = rewards + (1. - dones) * discount * max_next_qs
    with tf.GradientTape() as tape:
        qs = main_nn(states)
        action_masks = tf.one_hot(actions, NUM_ACTIONS)
        masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
        loss = mse(target, masked_qs)
    grads = tape.gradient(loss, main_nn.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
    return loss