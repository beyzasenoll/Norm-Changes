import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class PlotManager:
    """Class responsible for visualizing simulation data."""

    @staticmethod
    def plot_action_combinations(action_combinations):
        """Plot the frequencies of different action combinations over time."""
        plt.figure(figsize=(6, 4))
        plt.plot(action_combinations['AA'], label='Both A')
        plt.plot(action_combinations['BB'], label='Both B')
        plt.plot(action_combinations['AB'], label='A vs B')
        plt.plot(action_combinations['BA'], label='B vs A')
        plt.xlabel('Step')
        plt.ylabel('Frequency')
        plt.title('Action Combinations Over Time')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_q_values(scores_history, num_agents):
        """Plot the evolution of average Q-values for actions 'A' and 'B' over time for all agents."""
        plt.figure(figsize=(8, 6))
        num_timesteps = len(scores_history[0]['A'])
        avg_qval_A, avg_qval_B = [], []

        for t in range(num_timesteps):
            sum_qval_A = sum(scores_history[agent_id]['A'][t] for agent_id in range(num_agents))
            sum_qval_B = sum(scores_history[agent_id]['B'][t] for agent_id in range(num_agents))
            avg_qval_A.append(sum_qval_A / num_agents)
            avg_qval_B.append(sum_qval_B / num_agents)

        plt.plot(avg_qval_A, label='Average Q-value for Action A', color='blue')
        plt.plot(avg_qval_B, label='Average Q-value for Action B', color='orange')
        plt.xlabel('Timestep')
        plt.ylabel('Average Q-value')
        plt.title('Average Q-values for Actions A and B Over Time')
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_aa_vs_bb_results(aa_wins, bb_wins):
        """Plot the result of AA vs BB wins in the final timestep across multiple simulations."""
        labels = ['AA Wins', 'BB Wins']
        counts = [aa_wins, bb_wins]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, counts, color=['blue', 'green'])
        plt.xlabel('Action')
        plt.ylabel('Number of Wins')
        plt.title('AA vs BB Wins in the Final Timestep (Across Simulations)')
        plt.show()

    @staticmethod
    def plot_agent_actions_graph(agents, grid_height, grid_width):
        """Plot a graph-grid showing agents' final actions with colors and IDs."""
        G = nx.grid_2d_graph(grid_height, grid_width)
        pos = {node: (node[1], -node[0]) for node in G.nodes()}
        colors, labels = [], {}

        for agent in agents:
            color = 'blue' if agent.last_action == 'A' else 'yellow'
            colors.append(color)
            labels[(agent.agent_id // grid_width, agent.agent_id % grid_width)] = str(agent.agent_id)

        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, node_color=colors, with_labels=True, labels=labels, node_size=500, font_size=10,
                font_color='white', font_weight='bold')
        plt.title('Final Actions of Agents (Blue: A, Yellow: B)')
        plt.show()

    @staticmethod
    def plot_aa_vs_bb_results(aa_wins, bb_wins):
        """Plot the result of AA vs BB wins in the final timestep across multiple simulations."""
        labels = ['AA Wins', 'BB Wins']
        counts = [aa_wins, bb_wins]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, counts, color=['blue', 'green'])
        plt.xlabel('Action')
        plt.ylabel('Number of Wins')
        plt.title('AA vs BB Wins in the Final Timestep (Across Simulations)')
        plt.show()

    def plot_norm_abandonment_vs_reward(self,reward_values, abandonment_percentages):
        plt.figure(figsize=(10, 6))
        plt.plot(reward_values, abandonment_percentages, marker='o', linestyle='--', color='b')
        plt.xlabel("Reward for emerged norm")
        plt.ylabel("Percentage of population abandoning previous emerged norm")
        plt.title("Norm Abandonment vs. Reward for Emerged Norm")
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_norm_abandonment_vs_reward_multiple_topologies(reward_values, abandonment_percentages_by_topology):
        plt.figure(figsize=(10, 6))
        for topology, abandonment_percentages in abandonment_percentages_by_topology.items():
            plt.plot(reward_values, abandonment_percentages, marker='o', linestyle='--', label=f'{topology.capitalize()} Topology')
        plt.xlabel("Reward for emerged norm")
        plt.ylabel("Percentage of population abandoning previous emerged norm")
        plt.title("Norm Abandonment vs. Reward for Emerged Norm (Multiple Topologies)")
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_trendsetters_vs_norms(trendsetter_ratios, emergence_percentages, abandonment_percentages):
        plt.figure(figsize=(10, 6))

        plt.plot(trendsetter_ratios, emergence_percentages, marker='o', linestyle='-', color='g',
                 label='Percentage of Norm Emergence')
        plt.plot(trendsetter_ratios, abandonment_percentages, marker='x', linestyle='--', color='r',
                 label='Percentage of Norm Abandonment')

        plt.xlabel("Trendsetters Ratio (0.1 to 0.5)")
        plt.ylabel("Percentage (%)")
        plt.title("Norm Emergence and Abandonment vs. Trendsetters Ratio")
        plt.xticks(np.arange(0.1, 0.6, 0.05))  # X ekseni için etiketler
        plt.ylim(0, 110)  # Y eksenini 0-100 arasında tut
        plt.grid(True)
        plt.legend()
        plt.show()