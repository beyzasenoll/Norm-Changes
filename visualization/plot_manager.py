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
            sum_qval_A, sum_qval_B, count = 0.0, 0.0, 0
            for agent_id in range(num_agents):
                try:
                    sum_qval_A += scores_history[agent_id]['A'][t]
                    sum_qval_B += scores_history[agent_id]['B'][t]
                    count += 1
                except IndexError:
                    continue  # Bu ajan o timestep'te yoksa atla
            avg_qval_A.append(sum_qval_A / count if count > 0 else 0)
            avg_qval_B.append(sum_qval_B / count if count > 0 else 0)

        plt.figure(figsize=(10, 6))
        plt.plot(avg_qval_A, label='Average Q-value for Action A', color='blue')
        plt.plot(avg_qval_B, label='Average Q-value for Action B', color='orange')
        plt.xlabel('Timestep')
        plt.ylabel('Average Q-value')
        plt.title(f'Average Q-values for Actions A and B Over Time ({topology_type} Topology)')
        plt.legend()
        plt.show()


    @staticmethod
    def plot_agent_actions_graph_small_world(agents, num_agents, k, p, trendsetters_ids):
        """
        Plot a Small-World topology graph showing agents' final actions.
        Trendsetters are highlighted with black borders and bold labels.

        :param agents: List of agents.
        :param num_agents: Number of agents.
        :param k: Number of neighbors for Small-World topology.
        :param p: Rewiring probability for Small-World topology.
        :param trendsetters_ids: List of trendsetter agent IDs.
        """
        topology_graph = nx.watts_strogatz_graph(num_agents, k=k, p=p)
        pos = nx.spring_layout(topology_graph)

        colors = []
        labels = {}
        edge_colors = {}
        font_weights = {}

        for agent in agents:
            actions = agent.past_window['actions']
            actionCountA = actions.count('A')
            actionCountB = actions.count('B')

            if actionCountA > actionCountB:
                color = 'blue'
            elif actionCountB > actionCountA:
                color = 'orange'
            else:
                color = 'gray'

            colors.append(color)
            labels[agent.agent_id] = str(agent.agent_id)

            if agent.agent_id in trendsetters_ids:
                edge_colors[agent.agent_id] = 'black'
                font_weights[agent.agent_id] = 'bold'
            else:
                edge_colors[agent.agent_id] = 'gray'
                font_weights[agent.agent_id] = 'normal'

        plt.figure(figsize=(10, 6))
        nx.draw_networkx_edges(topology_graph, pos, edge_color='gray')

        nx.draw_networkx_nodes(
            topology_graph, pos,
            node_color=colors,
            node_size=500,
            edgecolors=[edge_colors[n] for n in topology_graph.nodes()],
            linewidths=2
        )

        for node_id in topology_graph.nodes():
            nx.draw_networkx_labels(
                topology_graph, pos,
                labels={node_id: labels[node_id]},
                font_size=10,
                font_color='white',
                font_weight=font_weights[node_id]
            )

        plt.title(
            f'Final Actions of Agents in Small-World Topology (Blue: A, Orange: B, Gray: Tie)\nTrendsetters have black borders and bold labels (k={k}, p={p})')
        plt.axis('off')
        plt.show()

    @staticmethod
    def plot_agent_actions_graph_toroidal(agents, trendsetters_ids, grid_height=None, grid_width=None):
        """
        Plot a graph-grid showing agents' final actions with colors, IDs, and borders for trendsetters.
        Grid size is automatically adjusted to the number of agents if not specified.

        :param agents: List of agents.
        :param trendsetters_ids: List of trendsetter agent IDs.
        :param grid_height: (Optional) Grid height.
        :param grid_width: (Optional) Grid width.
        """
        num_agents = len(agents)

        if grid_height is None or grid_width is None:
            grid_size = int(np.ceil(np.sqrt(num_agents)))
            grid_height = grid_size
            grid_width = grid_size

        G = nx.grid_2d_graph(grid_height, grid_width)
        all_nodes = list(G.nodes())[:num_agents]
        G = G.subgraph(all_nodes)

        pos = {node: (node[1], -node[0]) for node in G.nodes()}
        colors = []
        labels = {}
        edge_colors = []

        for agent, node in zip(agents, all_nodes):
            actions = agent.past_window['actions']
            actionCountA = actions.count('A')
            actionCountB = actions.count('B')

            if actionCountA > actionCountB:
                color = 'blue'
            elif actionCountB > actionCountA:
                color = 'orange'
            else:
                color = 'gray'

            colors.append(color)
            labels[node] = str(agent.agent_id)

            if agent.agent_id in trendsetters_ids:
                edge_colors.append('black')
            else:
                edge_colors.append('gray')

        plt.figure(figsize=(8, 6))
        nx.draw_networkx_edges(G, pos, edge_color='black')

        nx.draw_networkx_nodes(
            G, pos,
            node_color=colors,
            node_size=500,
            edgecolors=edge_colors,
            linewidths=2
        )

        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=10,
            font_color='white'
        )

        plt.title(
            'Final Actions of Agents (Blue: A, Orange: B, Gray: Tie) in Toroidal Topology\nTrendsetters have black borders')
        plt.show()

    @staticmethod
    def plot_agent_actions_graph_scale_free(agents, k, trendsetters_ids):
        """
        Plot a Scale-Free topology graph showing agents' final actions.
        Trendsetters will be highlighted with a border, but all nodes have same size.
        """
        num_agents = len(agents)
        G = nx.barabasi_albert_graph(num_agents, k)
        pos = nx.spring_layout(G, seed=42)

        colors = []
        labels = {}
        node_sizes = [500] * num_agents  # all nodes same size
        node_edge_colors = []

        for agent in agents:
            actions = agent.past_window['actions']
            count_A = actions.count('A')
            count_B = actions.count('B')

            color = 'blue' if count_A >= count_B else 'orange'
            colors.append(color)

            labels[agent.agent_id] = str(agent.agent_id)

            # Edge color: black if trendsetter, gray otherwise
            if agent.agent_id in trendsetters_ids:
                node_edge_colors.append('black')
            else:
                node_edge_colors.append('gray')

        plt.figure(figsize=(10, 6))
        nx.draw_networkx_edges(G, pos, edge_color='gray')

        nx.draw_networkx_nodes(
            G, pos,
            node_color=colors,
            node_size=node_sizes,
            edgecolors=node_edge_colors,
            linewidths=2
        )

        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=10,
            font_color='white'
        )

        plt.title('Final Actions of Agents (Blue: A, Orange: B)\nTrendsetters have black borders')
        plt.axis('off')
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


