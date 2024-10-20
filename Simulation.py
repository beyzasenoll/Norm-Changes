import random
import matplotlib.pyplot as plt
from Agent import Agent
import numpy as np


class Simulation:
    def __init__(self, num_agents, num_steps, alpha=0.1, gamma=0.95, epsilon=0.1, temperature=1.0):
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.agents = [Agent(i, alpha, gamma, epsilon, temperature) for i in range(num_agents)]
        self.scores_history = [[] for _ in range(num_agents)]
        self.action_combinations = {'AA': [], 'BB': [], 'AB': [], 'BA': []}
        self.total_A_count = 0
        self.total_B_count = 0

        # Grid dimensions
        self.grid_width = int(np.sqrt(self.num_agents))
        self.grid_height = self.grid_width

        # Ensure total agents fit into grid
        assert self.grid_width * self.grid_height == self.num_agents, "Number of agents must be a perfect square."

    def run(self):
        """Run the simulation for a specified number of steps."""
        for step in range(self.num_steps):
            count_AA, count_BB, count_AB, count_BA = 0, 0, 0, 0

            pairs = self.form_pairs_with_toroidal_topology(step)

            for agent1_id, agent2_id in pairs:
                agent1 = self.agents[agent1_id]
                agent2 = self.agents[agent2_id]

                action1 = agent1.choose_action_boltzmann(step)
                action2 = agent2.choose_action_boltzmann(step)

                self._update_action_counts(action1, action2)

                if action1 == 'A' and action2 == 'A':
                    count_AA += 1
                elif action1 == 'B' and action2 == 'B':
                    count_BB += 1
                elif action1 == 'A' and action2 == 'B':
                    count_AB += 1
                elif action1 == 'B' and action2 == 'A':
                    count_BA += 1

                reward1, reward2 = self._calculate_rewards(action1, action2)

                agent1.update_q_value(action1, reward1)
                agent2.update_q_value(action2, reward2)

                self.scores_history[agent1.agent_id].append(agent1.get_total_q_value())
                self.scores_history[agent2.agent_id].append(agent2.get_total_q_value())

            self._update_action_combinations(count_AA, count_BB, count_AB, count_BA)

    def form_pairs_with_toroidal_topology(self, episode):
        """Form pairs of agents based on toroidal grid topology."""
        pairs = []

        if episode % 4 == 0:
            # Right neighbor
            for row in range(self.grid_height):
                for col in range(0, self.grid_width, 2):
                    agent1_id = row * self.grid_width + col
                    agent2_id = row * self.grid_width + (col + 1) % self.grid_width
                    pairs.append((agent1_id, agent2_id))

        elif episode % 4 == 1:
            # Left neighbor
            for row in range(self.grid_height):
                for col in range(0, self.grid_width, 2):
                    agent1_id = row * self.grid_width + col
                    agent2_id = row * self.grid_width + (col - 1) % self.grid_width
                    pairs.append((agent1_id, agent2_id))

        elif episode % 4 == 2:
            # Below neighbor
            for col in range(self.grid_width):
                for row in range(0, self.grid_height, 2):
                    agent1_id = row * self.grid_width + col
                    agent2_id = ((row + 1) % self.grid_height) * self.grid_width + col
                    pairs.append((agent1_id, agent2_id))

        else:
            # Above neighbor
            for col in range(self.grid_width):
                for row in range(0, self.grid_height, 2):
                    agent1_id = row * self.grid_width + col
                    agent2_id = ((row - 1) % self.grid_height) * self.grid_width + col
                    pairs.append((agent1_id, agent2_id))

        return pairs

    def _update_action_counts(self, action1, action2):
        """Update the total count of 'A' and 'B' actions."""
        if action1 == 'A':
            self.total_A_count += 1
        else:
            self.total_B_count += 1

        if action2 == 'A':
            self.total_A_count += 1
        else:
            self.total_B_count += 1

    def _calculate_rewards(self, action1, action2):
        """Calculate rewards based on actions."""
        if action1 == action2:
            return 1, 1
        return -1, -1

    def _update_action_combinations(self, count_AA, count_BB, count_AB, count_BA):
        """Track the action combinations over time."""
        self.action_combinations['AA'].append(count_AA)
        self.action_combinations['BB'].append(count_BB)
        self.action_combinations['AB'].append(count_AB)
        self.action_combinations['BA'].append(count_BA)

    def plot_action_combinations(self):
        """Plot the frequencies of different action combinations over time."""
        plt.plot(self.action_combinations['AA'], label='Both A')
        plt.plot(self.action_combinations['BB'], label='Both B')
        plt.plot(self.action_combinations['AB'], label='A vs B')
        plt.plot(self.action_combinations['BA'], label='B vs A')

        plt.xlabel('Step')
        plt.ylabel('Frequency')
        plt.title('Action Combinations Over Time')
        plt.legend()
        plt.show()

    def print_action_counts(self):
        """Print the total count of 'A' and 'B' actions."""
        print(f"Total A selections: {self.total_A_count}")
        print(f"Total B selections: {self.total_B_count}")
