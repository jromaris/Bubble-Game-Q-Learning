import gym
from gym.wrappers import Monitor
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
from objs.constants import GAMEOVER_ROWS, GRID_COLS

container = dict()


def DQN(num_actions, activate):
    """https://keras.io/examples/rl/deep_q_network_breakout/"""

    input_lay = layers.Input(shape=(GAMEOVER_ROWS + 5, 2*GRID_COLS + 1, 5,))

    conv1 = layers.Conv2D(32, 4, strides=1, activation=activate)(input_lay)
    conv2 = layers.Conv2D(64, 2, strides=1, activation=activate)(conv1)

    flat = layers.Flatten()(conv2)
    dense1 = layers.Dense(512, activation=activate)(flat)
    dense2 = layers.Dense(512, activation=activate)(dense1)
    dense3 = layers.Dense(num_actions, activation='linear')(dense2)

    return keras.Model(inputs=input_lay, outputs=dense3)


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
            states.append(np.array(state, copy=False, dtype=np.float32))
            next_states.append(np.array(next_state, copy=False, dtype=np.float32))
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))
        states = np.array(states)
        actions = actions
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = tf.convert_to_tensor(dones)
        return states, actions, rewards, next_states, dones


def select_epsilon_greedy_action(main_nn, state, epsilon, num_actions):
    """Take random action with probability epsilon, else take best action."""
    result = tf.random.uniform((1,))
    if result < epsilon:
        return random.randint(0, num_actions-1)
    else:
        action_probs = main_nn(state, training=False)
        return tf.argmax(action_probs[0]).numpy()
        # Greedy action for state.


def do_trained_action(main_nn, state):
    action_probs = main_nn(state, training=False)
    return tf.argmax(action_probs[0]).numpy()


def train_step(states, actions, rewards, next_states, dones):

    """Perform a training iteration on a batch of data sampled from the experience
    replay buffer."""

    future_rewards = container['target_nn'].predict(next_states)
    updated_q_values = rewards + container['discount'] * tf.reduce_max(future_rewards, axis=1)
    updated_q_values = updated_q_values * (1. - dones) - dones

    action_masks = tf.one_hot(actions, container['num_actions'])
    with tf.GradientTape() as tape:
        qs = container['main_nn'](states)
        masked_qs = tf.reduce_sum(tf.multiply(qs, action_masks), axis=1)
        loss = container['mse'](updated_q_values, masked_qs)
    grads = tape.gradient(loss, container['main_nn'].trainable_variables)
    container['optimizer'].apply_gradients(zip(grads, container['main_nn'].trainable_variables))
    return loss
