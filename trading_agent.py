import random
import numpy as np
import tensorflow as tf
from collections import deque
from q_network import QNetwork

# The Trading Agent
class TradingAgent:
    def __init__(self, state_dim, action_dim):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.set_weights(self.q_network.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.replay_buffer = deque(maxlen=50000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        self.update_target_every = 20
        self.target_update_count = 0

    def select_action(self, state, position):
        if np.random.rand() < self.epsilon:
            if position == 0:
                return np.random.choice([0, 1])  # Buy or sell
            elif position == 1:
                return np.random.choice([1, 2])  # Sell or Hold
            elif position == -1:
                return np.random.choice([0, 2])  # Buy or Hold
        state = np.expand_dims(state, axis=0).astype(np.float32)
        q_values = self.q_network(state)
        return np.argmax(q_values[0])

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        next_q_values = self.target_network(next_states)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            one_hot_actions = tf.one_hot(actions, 3)
            q_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)
            loss = tf.keras.losses.huber(target_q_values, q_values)

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states = states.astype(np.float32)
        actions = actions.astype(np.int32)
        rewards = rewards.astype(np.float32)
        next_states = next_states.astype(np.float32)
        dones = dones.astype(np.float32)

        self._train_step(states, actions, rewards, next_states, dones)

        self.target_update_count += 1
        if self.target_update_count % self.update_target_every == 0:
            self.target_network.set_weights(self.q_network.get_weights())
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay