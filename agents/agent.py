import numpy as np
from environment.topology import Topology


class Agent:
    TRENDSETTER_THRESHOLD = 0.95  # Q(B) threshold for trendsetters
    def __init__(
        self,
        agent_id,
        alpha=0.05,
        gamma=0.95,
        epsilon=0.1,
        temperature=100,
        w1=0.4,
        w2=0.3,
        w3=0.3,
        num_agents=40,
        observation_beta=0.5,
        window_size=5,
        network_graph=None,
        is_trendsetter=False
    ):
        self.agent_id = agent_id
        self.actions = ['A', 'B']
        self.q_values = {'A': 0.96, 'B': 0.96 if is_trendsetter else 0.04}
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.temperature = temperature  # Temperature for softmax (if used)
        self.last_action = None  # Track the last action taken
        self.num_agents = num_agents
        self.grid_width = int(num_agents**0.5)
        self.grid_height = int(num_agents**0.5)
        self.window_size = window_size  # Window size for experience calculation
        self.is_trendsetter = is_trendsetter

        # Utility function weights
        self.w1 = w1  # Q-value weight
        self.w2 = w2  # Experience weight
        self.w3 = w3  # Neighbor influence weight

        # Store past actions and rewards for experience calculation
        self.past_window = {'actions': [], 'rewards': []}

        # Social network and observation parameters
        self.network_graph = network_graph
        self.observation_beta = observation_beta

    def compute_utility(self, action):
        """Compute utility for a given action based on Q-values, experience, and social influence."""
        q_val = self.q_values[action]
        e_val = self.compute_experience(action)
        o_val = self.compute_social_influence(action)
        return self.w1 * q_val + self.w2 * e_val + self.w3 * o_val

    def compute_social_influence(self, action):
        """Compute social influence based on the actions of observable neighbors."""
        neighbors = self.get_observable_neighbors()
        if not neighbors:
            return 0

        action_frequencies = []
        for neighbor in neighbors:
            last_actions = neighbor.past_window['actions'][-self.window_size:]
            if last_actions:
                action_count = last_actions.count(action)
                action_ratio = action_count / len(last_actions)
                action_frequencies.append(action_ratio)

        return np.mean(action_frequencies) if action_frequencies else 0

    def get_observable_neighbors(self):
        """Get observable neighbors based on the agent's position and observation beta."""
        row, col = divmod(self.agent_id, self.grid_width)
        topology = Topology(self.num_agents)
        neighbors = topology.calculate_beta_distance(
            row, col, self.grid_height, self.grid_width, self.observation_beta
        )
        return [Agent(r * self.grid_width + c) for r, c in neighbors]

    def compute_experience(self, action):
        """Compute normalized average cumulative experience for a specific action."""
        if not self.past_window['actions'] or not self.past_window['rewards']:
            return 0

        # Extract rewards corresponding to the given action
        action_rewards = [
            self.past_window['rewards'][i]
            for i, a in enumerate(self.past_window['actions'])
            if a == action
        ]

        if not action_rewards:
            return 0

        # Compute mean reward and normalize it
        mean_reward = np.mean(action_rewards)
        return self.normalize_experience(mean_reward)

    def normalize_experience(self, mean_reward, min_possible_reward=-1, max_possible_reward=1):
        """Normalize experience value between 0 and 1."""
        return (mean_reward - min_possible_reward) / (max_possible_reward - min_possible_reward)

    def choose_action(self):
        """Choose an action based on the computed utility values while tracking Q-values."""
        utilities = {action: self.compute_utility(action) for action in self.actions}

        chosen_action = max(utilities, key=utilities.get)

        self.last_action = chosen_action

        return chosen_action

    def update_q_value(self, action, reward):
        """Update the Q-value for the given action based on the received reward."""

        self.q_values[action] += self.alpha * (reward - self.q_values[action])
        self.past_window['actions'].append(action)
        self.past_window['rewards'].append(reward)

        # Ensure the window size is maintained
        if len(self.past_window['actions']) > self.window_size:
            self.past_window['actions'].pop(0)
            self.past_window['rewards'].pop(0)

    def update_q_value(self, action, reward):
        """Update the Q-value for the given action based on the received reward."""
        self.q_values[action] += self.alpha * (reward - self.q_values[action])

        #original q value :
        #self.q_values[action] += self.alpha * (
        #            reward + self.gamma * max(self.q_values.values()) - self.q_values[action])

        # Store action and reward in the past window
        self.past_window['actions'].append(action)
        self.past_window['rewards'].append(reward)

        # Maintain window size
        if len(self.past_window['actions']) > self.window_size:
            self.past_window['actions'].pop(0)
            self.past_window['rewards'].pop(0)

    def take_action(self):
        """Take an action and store it for observation."""
        self.last_action = self.choose_action()
        return self.last_action