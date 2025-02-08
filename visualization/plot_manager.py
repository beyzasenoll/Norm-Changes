import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class PlotManager:
    """Class responsible for visualizing simulation data."""

    @staticmethod
    def plot_action_combinations(action_combinations):
        """Plot the frequencies of different action combinations over time."""
        plt.figure(figsize=(6, 4))
        plt.plot(action_combinations['AA'], label='Both A',color='blue')
        plt.plot(action_combinations['BB'], label='Both B',color='orange')
        plt.plot(action_combinations['AB'], label='A vs B',color='green')
        plt.plot(action_combinations['BA'], label='B vs A', color='red')
        plt.xlabel('Step')
        plt.ylabel('Frequency')
        plt.title('Action Combinations Over Time')
        plt.legend()
        plt.show()


    @staticmethod
    def plot_q_values(scores_history, num_agents,num_steps):
        """Plot the evolution of average Q-values for actions 'A' and 'B' over time for all agents."""
        plt.figure(figsize=(8, 6))

        avg_qval_A, avg_qval_B = [], []

        for t in range(num_steps):
            sum_qval_A = 0
            sum_qval_B = 0
            count = 0

            for agent_id in range(num_agents):
                if t < len(scores_history[agent_id]['A']) and t < len(scores_history[agent_id]['B']):
                    sum_qval_A += scores_history[agent_id]['A'][t]
                    sum_qval_B += scores_history[agent_id]['B'][t]
                    count += 1

            avg_qval_A.append(sum_qval_A / count if count > 0 else 0.0)
            avg_qval_B.append(sum_qval_B / count if count > 0 else 0.0)

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
        plt.bar(labels, counts, color=['blue', 'orange'])
        plt.xlabel('Action')
        plt.ylabel('Number of Wins')
        plt.title('AA vs BB Wins in the Final Timestep (Across Simulations)')
        plt.show()

    @staticmethod
    def plot_agent_actions_graph_toroidal(agents, grid_height, grid_width):
        """Plot a graph-grid showing agents' final actions with colors and IDs."""
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
        plt.title('Final Actions of Agents (Blue: A, Yellow: B)')
        plt.show()

    def plot_agent_actions_graph_small_world(agents, num_agents, k ,p):
        """Plot a Small-World topology graph showing agents' final actions with colors and IDs."""
        topology_graph = nx.watts_strogatz_graph(num_agents, k=k, p=p)
        pos = nx.spring_layout(topology_graph)

        colors = ['blue' if agent.last_action == 'A' else 'orange' for agent in agents]
        labels = {agent.agent_id: str(agent.agent_id) for agent in agents}

        plt.figure(figsize=(8, 6))
        nx.draw(topology_graph, pos, node_color=colors, with_labels=True, labels=labels,
                node_size=500, font_size=10, font_color='white', font_weight='bold', edge_color='gray')
        plt.title('Final Actions of Agents (Blue: A, Yellow: B) in Small-World Topology')
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


