import numpy as np
import random


class Agent:
    def __init__(self, agent_id, alpha=0.05, gamma=0.95, epsilon=0.1, temperature=100):
        self.agent_id = agent_id
        self.actions = ['A', 'B']
        self.q_values = {'A': 0.0, 'B': 0.0}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.temperature = temperature
        self.last_action = None
        self.final_q_values = None

    def choose_action_epsilon_greedy(self):
        """Epsilon-greedy action selection."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Exploration
        self.last_action = max(self.q_values, key=self.q_values.get)
        return max(self.q_values, key=self.q_values.get)  # Exploitation

    def choose_action_boltzmann(self, timestep):
        """Boltzmann distribution-based action selection."""
        q_values_array = np.array(list(self.q_values.values()))
        exp_q_values = np.exp(q_values_array / self.temperature)
        if self.temperature > 0.05:
            self.temperature *= 0.995  # Update temperature to decrease over time
        probabilities = exp_q_values / np.sum(exp_q_values)

        self.last_action = np.random.choice(self.actions, p=probabilities)
        return np.random.choice(self.actions, p=probabilities)

    def update_q_value(self, action, reward):
        """Update Q-value based on the selected action and received reward."""
        self.q_values[action] += self.alpha * (reward - self.q_values[action])
