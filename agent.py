import numpy as np
from gymnasium.spaces import Discrete, Dict, Tuple, Sequence, MultiDiscrete
from itertools import combinations_with_replacement

class Agent:

    def __init__(self, name, action_space, capture_power):
        self.name = name
        self.action_space = action_space
        self.capture_power = capture_power

    def play(self, obs):
        return self.action_space.sample()

class EpsilonGreedyQLAgent:

    def __init__(self, name, step_size, n_actions, epsilon=0.9, alpha=0.1, gamma=0.9):
        self.name = name
        self.step_size = step_size
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        self.episode_transitions = []
        self.combinations = list(combinations_with_replacement(range(self.n_actions), self.step_size))
        # self.combinations.reverse()
        self.comb_len = len(self.combinations)

    def encode_obs(self, obs):
        return str(obs)

    def train_policy(self, encoded_obs):
        if encoded_obs not in self.q_table:
            self.q_table[encoded_obs] = np.zeros(self.comb_len)

        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.comb_len)
            return action
        else:
            action = np.argmax(self.q_table[encoded_obs])
            return action

    def play_normal(self, encoded_obs):
        if encoded_obs not in self.q_table:
            self.q_table[encoded_obs] = np.zeros(self.comb_len)
        # print(self.q_table[encoded_obs])
        action = np.argmax(self.q_table[encoded_obs])
        return action

    def store_transition(self, encoded_obs, action, reward):
        self.episode_transitions.append((encoded_obs, action, reward))

    def update_end_of_episode(self):
        total_reward = 0
        for encoded_obs, action, reward in reversed(self.episode_transitions):
            if encoded_obs not in self.q_table:
                self.q_table[encoded_obs] = np.zeros(self.comb_len)
            total_reward = reward + self.gamma * total_reward
            self.q_table[encoded_obs][action] += self.alpha * (total_reward - self.q_table[encoded_obs][action])
            # print(f"Updated {encoded_obs}, {action} with {self.q_table[encoded_obs][action]}")
        self.episode_transitions.clear()

    def set_spi(self, epsilon):
        self.epsilon = epsilon