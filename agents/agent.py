import numpy as np


class Agent:
    def __init__(
        self,
        agent_id,
        alpha=0.05, # Q-learning'de öğrenme oranı. Yeni ödül bilgisine ne kadar ağırlık verileceğini belirler.
        gamma=0.95,
        epsilon=0.15,
        temperature=0.1,
        weights=None,
        num_agents=40,
        beta=None,
        window_size=10,
        network_graph=None,
        simulation=None
    ):
        if weights is None:
            weights = [0, 0, 1]
        self.weights = weights
        self.agent_id = agent_id
        self.actions = ['A', 'B']
        self.q_values = {'A': 0.96, 'B': 0.04}
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.temperature = temperature
        self.num_agents = num_agents
        self.grid_width = int(num_agents**0.5)
        self.grid_height = int(num_agents**0.5)
        self.window_size = window_size

        self.past_window = {'actions': []}

        self.network_graph = network_graph
        self.beta = beta
        self.simulation = simulation

    def choose_max_utility_action(self):
        """Choose the action with the maximum utility."""
        utilities = {action: self.compute_utility(action) for action in self.actions}
        max_action = max(utilities, key=utilities.get)
        return max_action

    def compute_utility(self, action):
        """Compute utility for a given action based on Q-values, experience, and social influence."""
        q_val = self.q_values[action]
        e_val = self.compute_experience(action)
        o_val = self.compute_observation_with_past_window(action)
        return self.weights[0] * q_val + self.weights[1] * e_val + self.weights[2] * o_val

    def compute_observations(self, action):
        """Compute social influence based on the actions of observable neighbors."""
        neighbors = self.get_observable_neighbors()
        if not neighbors:
            return 0

        action_frequencies = []
        for neighbor in neighbors:
            last_actions = neighbor.past_window['actions'][-self.window_size:]
            if last_actions:
                # Flatten the list of lists to get all actions in the window
                flattened_actions = [a for sublist in last_actions for a in sublist]
                action_count = flattened_actions.count(action)
                action_ratio = action_count / len(flattened_actions) if flattened_actions else 0
                action_frequencies.append(action_ratio)

        return np.mean(action_frequencies) if action_frequencies else 0

    def compute_observation_with_past_window(self, action):
        """Compute social influence based on the actions of observable neighbors."""
        neighbors = self.get_observable_neighbors()
        if not neighbors:
            return 0

        action_frequencies = []
        for neighbor in neighbors:
            action_ratio = neighbor.compute_experience(action)
            action_frequencies.append(action_ratio)
        return np.mean(action_frequencies) if action_frequencies else 0
    def get_observable_neighbors_old(self):
        "Get observable neighbors based on the agent's position and observation beta."""
        row, col = divmod(self.agent_id, self.grid_width)
        topology = self.simulation.topology
        neighbors = topology.calculate_beta_distance(row, col, self.grid_height, self.grid_width, self.beta)
        neighbor_ids = [r * self.grid_width + c for r, c in neighbors]
        print(f"Agent {self.agent_id} observed neighbors: {neighbor_ids}")

        return [self.simulation.agents[r * self.grid_width + c] for r, c in neighbors]

    def get_observable_neighbors(self):
        """
        Get observable neighbors using topology-aware, beta-scaled distance.
        """
        topology = self.simulation.topology

        if self.simulation.topology_type == "toroidal":
            row, col = divmod(self.agent_id, self.grid_width)
            neighbors = topology.calculate_beta_distance(
                row, col,
                self.grid_height, self.grid_width,
                self.beta
            )
            return [
                self.simulation.agents[r * self.grid_width + c]
                for r, c in neighbors
                if 0 <= r * self.grid_width + c < len(self.simulation.agents)
            ]

        elif self.simulation.topology_type in ["scale_free", "small_world"]:
            neighbors = topology.calculate_beta_graph_neighbors(self.agent_id,self.simulation.num_agents, self.beta)
            #print(neighbors)
            return [self.simulation.agents[nid] for nid in neighbors]

        else:  # fallback for random or undefined
            neighbor_ids = topology.get_neighbors(self.agent_id, self.simulation.circle_degree)
            return [self.simulation.agents[nid] for nid in neighbor_ids]

    def compute_experience(self, action):
        """Compute the proportion of times the agent chose a given action in the past window."""
        if not self.past_window['actions']:
            return 0.5

        # Flatten the list of lists to get all actions in the window
        flattened_actions = [a for sublist in self.past_window['actions'][-self.window_size:] for a in sublist]
        action_count = flattened_actions.count(action)
        action_rate = action_count / len(flattened_actions) if flattened_actions else 0

        return action_rate

    def update_q_value(self, action, reward):
        """
        Update the Q-value for the given action based on the received reward.

        :param action: The action taken during the current timestep.
        :param reward: Reward received for the action.
        """
        self.q_values[action] += self.alpha * (reward - self.q_values[action])

    def update_past_actions(self, action):
        """
        Update the past actions window with the latest action.

        :param action: The action to be added to the past window.
        """
        self.past_window['actions'].append(action)

        if len(self.past_window['actions']) > self.window_size:
            self.past_window['actions'].pop(0)

    def choose_action_boltzmann(self):
        """Boltzmann distribution-based action selection."""
        utility_values = {'A': 0, 'B': 0}
        for action in self.actions:
            utility = self.compute_utility(action)
            utility_values[action] = utility

        utility_values_array = np.array(list(utility_values.values()))
        exp_q_values = np.exp(utility_values_array / self.temperature)
        if self.temperature > 0.05:
            self.temperature *= 0.995  # Update temperature to decrease over time
        probabilities = exp_q_values / np.sum(exp_q_values)

        return np.random.choice(self.actions, p=probabilities)

    def choose_action_boltzmann1(self):
        """Boltzmann distribution-based action selection."""
        utility_values = {'A': 0, 'B': 0}
        for action in self.actions:
            utility = self.compute_utility(action)
            utility_values[action] = utility

        utility_values_array = np.array(list(utility_values.values()))
        exp_q_values = np.exp(utility_values_array / self.temperature)
        if self.temperature < 20:
            self.temperature *= 1.5
        probabilities = exp_q_values / np.sum(exp_q_values)

        return np.random.choice(self.actions, p=probabilities)

    def choose_action_epsilon_greedy(self):
        """
        Choose an action using epsilon-greedy strategy with utility function.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)  # Explore: Choose a random action
        else:
            utilities = {action: self.compute_utility(action) for action in self.actions}
            return max(utilities, key=utilities.get)  # Exploit: Choose the best action
