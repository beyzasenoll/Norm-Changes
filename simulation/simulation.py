import logging
from random import random

import networkx as nx
import numpy as np

from agents.agent import Agent
from environment.reward import Reward
from environment.topology import Topology
from visualization.plot_manager import PlotManager

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Simulation:
    """
    A simulation environment for agents interacting in a network topology.
    """

    def __init__(self, num_agents, num_steps, topology_type='small_world', beta=0.5, k=4, p=0.2,
                 circle_degree=None, trendsetter_percent=10, epsilon=0.2, weights=None):
        if circle_degree is None:
            circle_degree = [1, 2, 3]
        if weights is None:
            weights = [0, 0, 1]
        self.action_combinations = {'AA': [], 'BB': [], 'AB': [], 'BA': []}
        self.scores_history = [{'A': [], 'B': []} for _ in range(num_agents)]
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.topology_type = topology_type
        self.pairs = None
        self.agents = [ Agent(i, simulation=self, observation_beta=beta, epsilon=epsilon, weights=weights, num_agents=num_agents)
            for i in range(num_agents)]

        self.beta = beta
        self.k = k
        self.p = p
        self.circle_degree = circle_degree
        self.trendsetter_percent = trendsetter_percent

        self.topology = Topology(num_agents, topology_type=topology_type, k=k, p=p)
        self.trendsetter_ids = self._select_trendsetters()
        self._apply_trendsetter_q_values()

    def run_simulation(self):
        """
        Run the simulation for the specified number of timesteps.
        Trendsetter agents choose action 'B' only in step 0.
        """
        for step in range(self.num_steps):
            if step % 1500 == 0:
                logger.info(f"Step {step}: Running simulation step.")

            count_AA, count_BB, count_AB, count_BA = 0, 0, 0, 0
            self.pairs = self.topology.form_pairs(self.circle_degree)

            for agent1_id, agent2_id in self.pairs:
                if agent2_id >= self.num_agents:
                    logger.error(f"Invalid agent2_id: {agent2_id}. Skipping this pair.")
                    continue

                agent1 = self.agents[agent1_id]
                agent2 = self.agents[agent2_id]

                # Trendsetters only act in step 0
                if step == 0:
                    action1 = 'B' if agent1_id in self.trendsetter_ids else agent1.choose_action_epsilon_greedy()
                    action2 = 'B' if agent2_id in self.trendsetter_ids else agent2.choose_action_epsilon_greedy()
                else:
                    action1 = agent1.choose_action_epsilon_greedy()
                    action2 = agent2.choose_action_epsilon_greedy()

                # Update action counts
                if action1 == 'A' and action2 == 'A':
                    count_AA += 1
                elif action1 == 'B' and action2 == 'B':
                    count_BB += 1
                elif action1 == 'A' and action2 == 'B':
                    count_AB += 1
                elif action1 == 'B' and action2 == 'A':
                    count_BA += 1

                reward1, reward2 = Reward.calculate_rewards(action1, action2)
                agent1.update_q_value(action1, reward1)
                agent1.update_past_actions(action1)
                agent2.update_q_value(action2, reward2)
                agent2.update_past_actions(action2)

                self._update_scores_history(agent1, agent2)

            self._update_action_counts(count_AA, count_BB, count_AB, count_BA)

        print(f"sum of action count", count_AA + count_BB + count_AB + count_BA)
        print(f"Trendsetter olarak seçilen ajan ID'leri: {self.trendsetter_ids}")

    def _select_trendsetters(self):
        num_trendsetters = max(1, int(self.num_agents * self.trendsetter_percent / 100))

        if self.topology_type == "toroidal":
            width = self.topology.grid_width
            height = self.topology.grid_height
            center = np.array([height // 2, width // 2])
            distances = []
            for agent_id in range(self.num_agents):
                pos = np.array(divmod(agent_id, width))
                dist = np.linalg.norm(center - pos)
                distances.append((agent_id, dist))
            sorted_ids = sorted(distances, key=lambda x: x[1])
            return [agent_id for agent_id, _ in sorted_ids[:num_trendsetters]]

        elif self.topology.graph:
            centrality = nx.degree_centrality(self.topology.graph)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            return [node_id for node_id, _ in sorted_nodes[:num_trendsetters]]

        else:  # fallback
            return random.sample(range(self.num_agents), num_trendsetters)

    def _apply_trendsetter_q_values(self):
        for tid in self.trendsetter_ids:
            self.agents[tid].q_values = {'A': 0.4, 'B': 0.96}

    def _update_action_counts(self, count_AA, count_BB, count_AB, count_BA):
        """
        Update the action combinations dictionary with counts from the current timestep.

        :param count_AA: Count of AA interactions.
        :param count_BB: Count of BB interactions.
        :param count_AB: Count of AB interactions.
        :param count_BA: Count of BA interactions.
        """
        self.action_combinations['AA'].append(count_AA)
        self.action_combinations['BB'].append(count_BB)
        self.action_combinations['AB'].append(count_AB)
        self.action_combinations['BA'].append(count_BA)

    def _update_scores_history(self, agent1, agent2):
        """
        Update the scores history for both agents.

        :param agent1: First agent in the pair.
        :param agent2: Second agent in the pair.
        """
        self.scores_history[agent1.agent_id]['A'].append(agent1.q_values['A'])
        self.scores_history[agent1.agent_id]['B'].append(agent1.q_values['B'])
        self.scores_history[agent2.agent_id]['A'].append(agent2.q_values['A'])
        self.scores_history[agent2.agent_id]['B'].append(agent2.q_values['B'])

    def plot_simulation_results(self):
        """
        Plot the results of the simulation.
        """
        logger.info("Plotting simulation results...")
        timesteps = range(self.num_steps)
        PlotManager.plot_action_combinations(self.action_combinations, timesteps, self.topology_type)
        PlotManager.plot_q_values(self.scores_history, self.num_agents, self.num_steps, self.topology_type)
        if self.topology_type == "small_world":
            PlotManager.plot_agent_actions_graph_small_world(self.agents, self.num_agents, self.k, self.p)
        elif self.topology_type == "toroidal":
            PlotManager.plot_agent_actions_graph_toroidal(self.agents)
