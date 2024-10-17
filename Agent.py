import numpy as np
import random
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, agent_id, alpha=0.1, gamma=0.95, epsilon=0.1, temperature=1.0):
        self.agent_id = agent_id
        self.actions = ['A', 'B']
        self.q_values = {'A': 0.0, 'B': 0.0}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.temperature = temperature

    def choose_action_epsilon_greedy(self):
        """Epsilon-greedy action selection."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Exploration
        return max(self.q_values, key=self.q_values.get)  # Exploitation

    def choose_action_boltzmann(self):
        """Boltzmann distribution-based action selection."""
        q_values_array = np.array(list(self.q_values.values()))
        exp_q_values = np.exp(q_values_array / self.temperature)
        probabilities = exp_q_values / np.sum(exp_q_values)

        return np.random.choice(self.actions, p=probabilities)

    def update_q_value(self, action, reward):
        """Update Q-value based on the selected action and received reward."""
        max_future_q = max(self.q_values.values())
        self.q_values[action] += self.alpha * (reward + self.gamma * max_future_q - self.q_values[action])

    def get_total_q_value(self):
        """Return the sum of Q-values for all actions."""
        return sum(self.q_values.values())