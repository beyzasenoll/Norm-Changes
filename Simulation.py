import random
import matplotlib.pyplot as plt
from Agent import Agent
import networkx as nx


class Simulation:
    def __init__(self, num_agents, num_steps, alpha=0.1, gamma=0.95, epsilon=0.1, temperature=100,
                 topology_type="toroidal"):
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.agents = [Agent(i, alpha, gamma, epsilon, temperature) for i in range(num_agents)]
        self.scores_history = [{'A': [], 'B': []} for _ in range(num_agents)]
        self.action_combinations = {'AA': [], 'BB': [], 'AB': [], 'BA': []}
        self.topology_type = topology_type
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.temperature = temperature
        self.pairs = []
        self.norm_changed = False

        self.grid_width = 4
        self.grid_height = int(self.num_agents / self.grid_width)

        if self.topology_type == 'scale-free':
            self.scale_free_graph = nx.barabasi_albert_graph(self.num_agents, 2)

    def run_with_emergence_check(self):
        self.run_simulation()
        if not self.check_norm_emergence():
            print("Norm couldn't emerge")
            less_action = self.determine_less_norm_action()
            self.update_non_emerging_agents_q_values(less_action)

            self.keep_q_values()
            self.reset_to_final_q_values()

            self.run_simulation()

            if self.check_norm_emergence():
                print("When we execute again, norm emerged.")
            else:
                print("When we execute again, norm couldn't emerge.")

        if self.topology_type == 'random':
            self.norm_changed = True

            self.keep_q_values()
            self.reset_to_final_q_values()

            self.run_simulation()

    def run_simulation(self):
        """Run the simulation for a specified number of steps."""
        for step in range(self.num_steps):
            count_AA, count_BB, count_AB, count_BA = 0, 0, 0, 0

            if self.topology_type == 'toroidal':
                self.pairs = self._form_pairs_with_toroidal_topology(step)
            elif self.topology_type == 'scale-free':
                self.pairs = self._form_pairs_with_scale_free_topology()
            elif self.topology_type == 'random':
                self.pairs = self._form_pairs_randomly()

            for agent1_id, agent2_id in self.pairs:
                agent1 = self.agents[agent1_id]
                agent2 = self.agents[agent2_id]

                action1 = agent1.choose_action_boltzmann(step)
                action2 = agent2.choose_action_boltzmann(step)

                if action1 == 'A' and action2 == 'A':
                    count_AA += 1
                elif action1 == 'B' and action2 == 'B':
                    count_BB += 1
                elif action1 == 'A' and action2 == 'B':
                    count_AB += 1
                elif action1 == 'B' and action2 == 'A':
                    count_BA += 1
                if self.topology_type == 'random' and self.norm_changed:
                    reward1, reward2 = self._calculate_rewards_for_random_topology(action1, action2)
                else:
                    reward1, reward2 = self._calculate_rewards(action1, action2)

                agent1.update_q_value(action1, reward1)
                agent2.update_q_value(action2, reward2)

                self.scores_history[agent1.agent_id]['A'].append(agent1.q_values['A'])
                self.scores_history[agent1.agent_id]['B'].append(agent1.q_values['B'])

                self.scores_history[agent2.agent_id]['A'].append(agent2.q_values['A'])
                self.scores_history[agent2.agent_id]['B'].append(agent2.q_values['B'])

            self._update_action_combinations(count_AA, count_BB, count_AB, count_BA)

        self.plot_action_combinations()
        self.plot_q_values()
        self.plot_agent_actions_graph()

    def check_norm_emergence(self):
        count_A = sum(1 for agent in self.agents if agent.last_action == 'A')
        count_B = self.num_agents - count_A

        if count_A >= self.num_agents * 0.9 or count_B >= self.num_agents * 0.9:
            return True
        else:
            return False

    def determine_less_norm_action(self):
        count_A = sum(1 for agent in self.agents if agent.last_action == 'A')
        count_B = self.num_agents - count_A

        if count_A > count_B:
            return 'B'
        else:
            return 'A'

    def update_non_emerging_agents_q_values(self, action):
        agents_choosing_action = [agent for agent in self.agents if agent.last_action == action]

        num_agents_to_update = int(len(agents_choosing_action) * 0.3)
        agents_to_update = random.sample(agents_choosing_action, num_agents_to_update)
        print("Agents choosing action", action, ":", [agent.agent_id for agent in agents_choosing_action])

        for agent in agents_to_update:
            if action == 'B':
                agent.q_values['A'] = 1.0
                agent.q_values['B'] = -1.0
            else:
                agent.q_values['B'] = 1.0
                agent.q_values['A'] = -1.0

    def keep_q_values(self):
        for agent in self.agents:
            agent.final_q_values = agent.q_values.copy()
            agent.fixed_q_values = True

    def reset_to_final_q_values(self):

        for agent in self.agents:
            if agent.final_q_values:
                agent.q_values = agent.final_q_values.copy()

    def _form_pairs_with_toroidal_topology(self, episode):
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

    def _form_pairs_with_scale_free_topology(self):
        edges = list(self.scale_free_graph.edges)

        random.shuffle(edges)

        paired_agents = set()
        pairs = []

        for edge in edges:
            agent1_id, agent2_id = edge
            if agent1_id not in paired_agents and agent2_id not in paired_agents:
                pairs.append((agent1_id, agent2_id))
                paired_agents.add(agent1_id)
                paired_agents.add(agent2_id)

            if len(paired_agents) >= self.num_agents:
                break

        return pairs

    def _form_pairs_randomly(self):
        """Pair agents randomly for each timestep, returning agent indices."""
        agent_indices = list(range(self.num_agents))
        random.shuffle(agent_indices)
        pairs = [(agent_indices[i], agent_indices[i + 1]) for i in range(0, len(agent_indices) - 1, 2)]
        return pairs

    def _calculate_rewards(self, action1, action2):
        """Calculate rewards based on actions."""
        if action1 == action2:
            return 1, 1
        return -1, -1

    def _calculate_rewards_for_random_topology(self, action1, action2):
        """Calculate rewards for changing norm."""
        if action1 == action2:
            return 0.5, 0.5
        return -1, -1

    def _update_action_combinations(self, count_AA, count_BB, count_AB, count_BA):
        """Track the action combinations over time."""
        self.action_combinations['AA'].append(count_AA)
        self.action_combinations['BB'].append(count_BB)
        self.action_combinations['AB'].append(count_AB)
        self.action_combinations['BA'].append(count_BA)

    def run_multiple_simulations(self, num_simulations=20):
        """Run multiple simulations and track whether 'AA' or 'BB' dominates."""
        aa_wins = 0
        bb_wins = 0

        for sim in range(num_simulations):
            self.run_simulation()

            count_A = 0
            count_B = 0

            for agent in range(self.num_agents):
                agent = self.agents[agent]
                last_action_1 = agent.last_action

                if last_action_1 == 'A':
                    count_A += 1

                elif last_action_1 == 'B':
                    count_B += 1
            if count_A >= self.num_agents * 0.9:
                aa_wins += 1
            elif count_B >= self.num_agents * 0.9:
                bb_wins += 1

            print(sim)
            self.reset_simulation()

        self.plot_aa_vs_bb_results(aa_wins, bb_wins)

    def reset_simulation(self):
        """Reset the simulation to run it again with the same agents."""
        self.scores_history = [{'A': [], 'B': []} for _ in range(self.num_agents)]
        self.action_combinations = {'AA': [], 'BB': [], 'AB': [], 'BA': []}
        self.agents = [Agent(i, self.alpha, self.gamma, self.epsilon, self.temperature) for i in range(self.num_agents)]

    def simulation_different_agent_size(self):
        agent_sizes = [40, 80, 120, 200]
        norm_counts = []

        for agent_size in agent_sizes:
            self.num_agents = agent_size
            self.grid_height = (self.num_agents + self.grid_width - 1) // self.grid_width  # Adjust grid height

            self.agents = [Agent(i, self.alpha, self.gamma, self.epsilon, self.temperature) for i in
                           range(self.num_agents)]
            self.scores_history = [{'A': [], 'B': []} for _ in range(self.num_agents)]  # Reinitialize scores_history

            self.run_with_emergence_check()

            count_A = sum(1 for agent in self.agents if agent.last_action == 'A')
            count_B = self.num_agents - count_A

            if count_A >= self.num_agents * 0.9:
                norm_counts.append((count_A // agent_size) * 100)
            elif count_B >= self.num_agents * 0.9:
                norm_counts.append((count_B // agent_size) * 100)
            else:
                norm_counts.append(0)  # If no norm emerged

            self.reset_simulation()

        plt.figure(figsize=(10, 6))
        plt.plot(agent_sizes, norm_counts, marker='o')
        plt.title("Norm Emergence Percentage")
        plt.xlabel("Number of Agents")
        plt.ylabel("Percentage Count of Dominant Action")
        plt.show()

    def plot_action_combinations(self):
        """Plot the frequencies of different action combinations over time."""
        plt.figure(figsize=(6, 4))

        plt.plot(self.action_combinations['AA'], label='Both A')
        plt.plot(self.action_combinations['BB'], label='Both B')
        plt.plot(self.action_combinations['AB'], label='A vs B')
        plt.plot(self.action_combinations['BA'], label='B vs A')

        plt.xlabel('Step')
        plt.ylabel('Frequency')
        plt.title('Action Combinations Over Time')
        plt.legend()
        plt.show()

    def plot_q_values(self):
        """Plot the evolution of average Q-values for actions 'A' and 'B' over time for all agents."""
        plt.figure(figsize=(8, 6))

        num_timesteps = len(self.scores_history[0]['A'])

        avg_qval_A = []
        avg_qval_B = []

        # Loop through each timestep
        for t in range(num_timesteps):
            sum_qval_A = 0
            sum_qval_B = 0

            for agent_id in range(self.num_agents):
                sum_qval_A += self.scores_history[agent_id]['A'][t]
                sum_qval_B += self.scores_history[agent_id]['B'][t]

            avg_qval_A.append(sum_qval_A / self.num_agents)
            avg_qval_B.append(sum_qval_B / self.num_agents)

        plt.plot(avg_qval_A, label='Average Q-value for Action A', color='blue')
        plt.plot(avg_qval_B, label='Average Q-value for Action B', color='orange')

        plt.xlabel('Timestep')
        plt.ylabel('Average Q-value')
        plt.title('Average Q-values for Actions A and B Over Time')

        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_aa_vs_bb_results(self, aa_wins, bb_wins):
        """Plot the result of AA vs BB wins in the final timestep across multiple simulations."""
        labels = ['AA Wins', 'BB Wins']
        counts = [aa_wins, bb_wins]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, counts, color=['blue', 'green'])
        plt.xlabel('Action')
        plt.ylabel('Number of Wins')
        plt.title('AA vs BB Wins in the Final Timestep (Across Simulations)')
        plt.show()

    def plot_agent_actions_graph(self):
        """Plot a graph-grid showing agents' final actions with colors and IDs."""
        G = nx.grid_2d_graph(self.grid_height, self.grid_width)
        pos = {node: (node[1], -node[0]) for node in G.nodes()}

        colors = []
        labels = {}

        for agent in self.agents:
            color = 'blue' if agent.last_action == 'A' else 'yellow'
            colors.append(color)
            labels[(agent.agent_id // self.grid_width, agent.agent_id % self.grid_width)] = str(agent.agent_id)

        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, node_color=colors, with_labels=True, labels=labels, node_size=500, font_size=10,
                font_color='white', font_weight='bold')

        plt.title('Final Actions of Agents (Blue: A, Red: B)')
        plt.show()
