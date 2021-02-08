import gym
from gym.wrappers import Monitor
from collections import deque
import tensorflow as tf
import numpy as np
import random
from objs.constants import GAMEOVER_ROWS, GRID_COLS

container = dict()

class DQN(tf.keras.Model):
    """Dense neural network class. https://keras.io/examples/rl/deep_q_network_breakout/"""
    def __init__(self, num_actions, activate, batch_size):
        super(DQN, self).__init__()

        self.input_lay = tf.keras.layers.InputLayer(input_shape=(batch_size, GAMEOVER_ROWS+2, GRID_COLS+1, 1))
        # self.input_lay = tf.keras.layers.InputLayer(input_shape=(batch_size, GAMEOVER_ROWS+2, GRID_COLS*2+1, 1))

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(1, 2), activation=activate,
                                            bias_initializer=tf.constant_initializer(value=0.1))
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), activation=activate,
                                            bias_initializer=tf.constant_initializer(value=0.1))
        self.conv3 = tf.keras.layers.Conv2D(filters=96, kernel_size=(2, 2), strides=(1, 1), activation=activate,
                                            bias_initializer=tf.constant_initializer(value=0.1))
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation=activate,
                                            bias_initializer=tf.constant_initializer(value=0.1))
        # self.dense2 = tf.keras.layers.Dense(512, activation=activate,
        #                                     bias_initializer=tf.constant_initializer(value=0.1))
        # self.dense3 = tf.keras.layers.Dense(num_actions, activation='sigmoid')
        self.dense3 = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, x, training=True, mask=None):
        """Forward pass."""
        x = self.input_lay(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flat(x)

        x = self.dense1(x)
        # x = self.dense2(x)
        x = self.dense3(x)
        return x


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
            actions.append(np.array(action, copy=False, dtype=np.int32))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False, dtype=np.float32))
            dones.append(done)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)
        return states, actions, rewards, next_states, dones


def select_epsilon_greedy_action(main_nn, state, epsilon, num_actions):
    """Take random action with probability epsilon, else take best action."""
    result = tf.random.uniform((1,))
    if result < epsilon:
        return random.randint(0, num_actions-1)
    else:
        return tf.argmax(main_nn(state)[0]).numpy()
        # Greedy action for state.


def do_trained_action(main_nn, state):
    return tf.argmax(main_nn(state)[0]).numpy()


# @tf.function
def train_step(states, actions, rewards, next_states, dones):

    """Perform a training iteration on a batch of data sampled from the experience
    replay buffer."""
    # Calculate targets.
    # main_nn, target_nn, mse, optimizer, states, actions,
    # rewards, next_states, dones, discount, num_actions):

    next_qs = container['target_nn'](next_states)
    target = rewards + container['discount'] * tf.reduce_max(next_qs, axis=1)
    target = target*(1. - dones) - dones

    action_masks = tf.one_hot(actions, container['num_actions'])
    with tf.GradientTape() as tape:
        qs = container['main_nn'](states)
        masked_qs = tf.reduce_sum(tf.multiply(qs, action_masks), axis=1)
        loss = container['mse'](target, masked_qs)
    grads = tape.gradient(loss, container['main_nn'].trainable_variables)
    container['optimizer'].apply_gradients(zip(grads, container['main_nn'].trainable_variables))
    return loss
