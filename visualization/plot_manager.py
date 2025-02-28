import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class PlotManager:
    """Class responsible for visualizing simulation data."""

    @staticmethod
    def plot_action_combinations(action_combinations, timesteps, topology_type):
        """
        Plot the frequencies of different action combinations over time.

        :param action_combinations: Dictionary of action combination counts.
        :param timesteps: List of timesteps for the x-axis.
        :param topology_type: Type of network topology (e.g., 'Small-World', 'Toroidal').
        """
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, action_combinations['AA'], label='Both A', color='blue')
        plt.plot(timesteps, action_combinations['BB'], label='Both B', color='orange')
        plt.plot(timesteps, action_combinations['AB'], label='A vs B', color='green')
        plt.plot(timesteps, action_combinations['BA'], label='B vs A', color='red')
        plt.xlabel('Timestep')
        plt.ylabel('Frequency')
        plt.title(f'Action Combinations Over Time ({topology_type} Topology)')
        plt.legend()
        plt.show()

    @staticmethod
    @staticmethod
    def plot_q_values(scores_history, num_agents, num_steps, topology_type):
        """
        Plot the evolution of average Q-values for actions 'A' and 'B' over time.

        :param scores_history: List of Q-values for each agent.
        :param num_agents: Number of agents.
        :param num_steps: Number of timesteps.
        :param topology_type: Type of network topology (e.g., 'Small-World', 'Toroidal').
        """
        avg_qval_A, avg_qval_B = [], []

        for t in range(num_steps):
            sum_qval_A = sum(scores_history[agent_id]['A'][t] for agent_id in range(num_agents))
            sum_qval_B = sum(scores_history[agent_id]['B'][t] for agent_id in range(num_agents))
            avg_qval_A.append(sum_qval_A / num_agents)
            avg_qval_B.append(sum_qval_B / num_agents)

        plt.figure(figsize=(10, 6))
        plt.plot(avg_qval_A, label='Average Q-value for Action A', color='blue')
        plt.plot(avg_qval_B, label='Average Q-value for Action B', color='orange')
        plt.xlabel('Timestep')
        plt.ylabel('Average Q-value')
        plt.title(f'Average Q-values for Actions A and B Over Time ({topology_type} Topology)')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_agent_actions_graph_small_world(agents, num_agents, k, p):
        """
        Plot a Small-World topology graph showing agents' final actions.

        :param agents: List of agents.
        :param num_agents: Number of agents.
        :param k: Number of neighbors for Small-World topology.
        :param p: Rewiring probability for Small-World topology.
        """
        topology_graph = nx.watts_strogatz_graph(num_agents, k=k, p=p)
        pos = nx.spring_layout(topology_graph)

        colors = ['blue' if agent.last_action == 'A' else 'orange' for agent in agents]
        labels = {agent.agent_id: str(agent.agent_id) for agent in agents}

        plt.figure(figsize=(10, 6))
        nx.draw(topology_graph, pos, node_color=colors, with_labels=True, labels=labels,
                node_size=500, font_size=10, font_color='white', font_weight='bold', edge_color='gray')
        plt.title(f'Final Actions of Agents (Blue: A, Orange: B) in Small-World Topology (k={k}, p={p})')
        plt.show()

    @staticmethod
    def plot_agent_actions_graph_toroidal(agents, grid_height, grid_width):
        """
        Plot a graph-grid showing agents' final actions with colors and IDs.

        :param agents: List of agents.
        :param grid_height: Height of the toroidal grid.
        :param grid_width: Width of the toroidal grid.
        """
        G = nx.grid_2d_graph(grid_height, grid_width)
        pos = {node: (node[1], -node[0]) for node in G.nodes()}
        colors, labels = [], {}

        for agent in agents:
            color = 'blue' if agent.last_action == 'A' else 'orange'
            colors.append(color)
            labels[(agent.agent_id // grid_width, agent.agent_id % grid_width)] = str(agent.agent_id)

        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, node_color=colors, with_labels=True, labels=labels, node_size=500, font_size=10,
                font_color='white', font_weight='bold')
        plt.title(f'Final Actions of Agents (Blue: A, Orange: B) in Toroidal Topology')
        plt.show()
    @staticmethod
    def plot_aa_vs_bb_results(aa_wins, bb_wins):
        """Plot the result of AA vs BB wins in the final timestep across multiple simulations."""
        labels = ['AA Wins', 'BB Wins']
        counts = [aa_wins, bb_wins]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, counts, color=['blue', 'orange'])
        plt.xlabel('Action')
        plt.ylabel('Number of Wins')
        plt.title('AA vs BB Wins in the Final Timestep (Across Simulations)')
        plt.show()


    @staticmethod
    def plot_simulation_results(action_combinations, scores_history, agents, grid_height, grid_width):
        PlotManager.plot_action_combinations(action_combinations)
        PlotManager.plot_q_values(scores_history, len(agents))
        PlotManager.plot_agent_actions_graph_toroidal(agents, grid_height, grid_width)


    def plot_action_percentages(stepwise_percentages):
        """Plot the percentage of actions A and B across simulation steps."""
        steps = np.arange(len(stepwise_percentages))
        percentages_A = [p[0] for p in stepwise_percentages]
        percentages_B = [p[1] for p in stepwise_percentages]

        bar_width = 0.4
        plt.figure(figsize=(10, 6))

        plt.bar(steps - bar_width / 2, percentages_A, width=bar_width, label='Action A (%)', color='#4CAF50')
        plt.bar(steps + bar_width / 2, percentages_B, width=bar_width, label='Action B (%)', color='#FF5733')

        plt.xlabel('Simulation Steps', fontsize=12)
        plt.ylabel('Percentage of Actions (%)', fontsize=12)
        plt.title('Percentage of Actions A and B Over Simulation Steps', fontsize=14)
        plt.xticks(steps)
        plt.legend()
        plt.tight_layout()
        plt.show()


