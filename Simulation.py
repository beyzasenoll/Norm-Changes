import random
import matplotlib.pyplot as plt
import numpy as np

from Agent import Agent
import networkx as nx

from RewardManager import Reward
from PlotManager import PlotManager
from Topology import Topology


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
        self.less_action= None
        self.norm_changed=False

        self.grid_width = 10
        self.grid_height = int(self.num_agents / self.grid_width)

        if self.topology_type == 'scale-free':
            self.scale_free_graph = nx.barabasi_albert_graph(self.num_agents, 2)

    def run_simulation(self,reward=None):
        """Run the simulation for a specified number of steps."""
        for step in range(self.num_steps):
            count_AA, count_BB, count_AB, count_BA = 0, 0, 0, 0

            if self.topology_type == 'toroidal':
                self.pairs = Topology._form_pairs_with_toroidal_topology(self, step, self.grid_height, self.grid_width)
            elif self.topology_type == 'scale-free':
                self.pairs = Topology._form_pairs_with_scale_free_topology(self, self.num_agents, self.scale_free_graph)
            elif self.topology_type == 'random':
                self.pairs = Topology._form_pairs_randomly(self, self.num_agents)

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
                if self.norm_changed:
                    reward1, reward2 = Reward._calculate_rewards_norm_change_(action1, action2,self.less_action,reward)
                else:
                    reward1, reward2 = Reward._calculate_rewards(action1, action2)

                if reward1 is None or reward2 is None:
                    raise ValueError(f"Reward values should not be None: reward1={reward1}, reward2={reward2}")


                agent1.update_q_value(action1, reward1)
                agent2.update_q_value(action2, reward2)

                self.scores_history[agent1.agent_id]['A'].append(agent1.q_values['A'])
                self.scores_history[agent1.agent_id]['B'].append(agent1.q_values['B'])

                self.scores_history[agent2.agent_id]['A'].append(agent2.q_values['A'])
                self.scores_history[agent2.agent_id]['B'].append(agent2.q_values['B'])

            self._update_action_combinations(count_AA, count_BB, count_AB, count_BA)

    def run_with_emergence_check(self, drawPlot=True):
        self.run_simulation()
        if not self.check_norm_emergence():
            print("Norm couldn't emerge")
            less_action = self.determine_less_norm_action()
            self.update_non_emerging_agents_q_values(less_action)

            self.keep_q_values()
            self.reset_to_final_q_values()

            self.run_simulation()
            if drawPlot:
                self.plot_simulation_results()

            if self.check_norm_emergence():
                print("When we execute again, norm emerged.")
                if drawPlot:
                    self.plot_simulation_results()
            else:
                print("When we execute again, norm couldn't emerge.")
                if drawPlot:
                    self.plot_simulation_results()

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

        num_agents_to_update = int(len(agents_choosing_action) * 0.35)
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

        PlotManager.plot_aa_vs_bb_results(aa_wins, bb_wins)

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
            self.grid_height = (self.num_agents + self.grid_width - 1) // self.grid_width

            self.agents = [Agent(i, self.alpha, self.gamma, self.epsilon, self.temperature) for i in
                           range(self.num_agents)]
            self.scores_history = [{'A': [], 'B': []} for _ in range(self.num_agents)]

            self.run_with_emergence_check(False)

            count_A = sum(1 for agent in self.agents if agent.last_action == 'A')
            count_B = self.num_agents - count_A

            if count_A >= self.num_agents * 0.9:
                norm_counts.append((count_A // self.num_agents) * 100)
            elif count_B >= self.num_agents * 0.9:
                norm_counts.append((count_B // self.num_agents) * 100)
            else:
                norm_counts.append(0)

            self.reset_simulation()

        plt.figure(figsize=(10, 6))
        plt.plot(agent_sizes, norm_counts, marker='o')
        plt.title("Norm Emergence Percentage")
        plt.xlabel("Number of Agents")
        plt.ylabel("Percentage Count of Dominant Action")
        plt.show()

    def calculate_norm_abandonment(self, emerged_norm):
        num_abandoning = sum(
            1 for agent in self.agents if agent.last_action == emerged_norm
        )
        return (num_abandoning / self.num_agents) * 100

    def run_after_update_reward(self):
        reward_values = [-0.5, 0, 0.5]
        topologies = ["random", "toroidal"]
        abandonment_percentages_by_topology = {topology: [] for topology in topologies}

        for topology in topologies:
            print(f"Running simulation for topology: {topology}")
            self.topology_type = topology
            abandonment_percentages = []

            for reward in reward_values:
                self.run_with_emergence_check(False)
                self.less_action = self.determine_less_norm_action()
                print("---------------------------------------------------------------------------------")
                print("NORM IS CHANGING")
                self.keep_q_values()
                self.reset_to_final_q_values()
                self.norm_changed = True

                print(f"Running simulation with norm changed and reward={reward} for topology {topology}")

                self.run_simulation(reward)
                abandonment_percentage = self.calculate_norm_abandonment(self.less_action)
                abandonment_percentages.append(abandonment_percentage)

                print("After change the norm")
                self.reset_simulation()
                self.norm_changed = False

            abandonment_percentages_by_topology[topology] = abandonment_percentages

        # Plotting the results for both topologies
        PlotManager.plot_norm_abandonment_vs_reward_multiple_topologies(reward_values,
                                                                        abandonment_percentages_by_topology)
    def plot_simulation_results(self):
        PlotManager.plot_action_combinations(self.action_combinations)
        PlotManager.plot_q_values(self.scores_history, self.num_agents)
        PlotManager.plot_agent_actions_graph(self.agents, self.grid_height, self.grid_width)
