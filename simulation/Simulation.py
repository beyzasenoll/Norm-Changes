import random
import matplotlib.pyplot as plt
import networkx as nx

from agents.agent import Agent
from environment.reward import Reward
from environment.topology import Topology
from simulation.norm_checker import NormChecker
from simulation.reset_manager import ResetManager
from visualization.plot_manager import PlotManager


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
        self.pairs = []

        self.norm_checker = NormChecker(self.agents, self.num_agents)
        self.reset_manager = ResetManager(self.agents, self.num_agents)

        self.grid_width = 10
        self.grid_height = int(self.num_agents / self.grid_width)

        if self.topology_type == 'scale-free':
            self.scale_free_graph = nx.barabasi_albert_graph(self.num_agents, 2)

    def run_simulation(self, reward=None):
        """Run the simulation for a specified number of steps."""
        for step in range(self.num_steps):
            count_AA, count_BB, count_AB, count_BA = 0, 0, 0, 0

            # Generate pairs based on topology
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
                if self.norm_checker.norm_changed:
                    reward1, reward2 = Reward._calculate_rewards_norm_change_(action1, action2,
                                                                              self.norm_checker.less_action, reward)
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
        """Run the simulation with norm emergence check and reset if necessary."""
        self.run_simulation()
        if drawPlot:
            self.plot_simulation_results()

        if not self.norm_checker.check_norm_emergence():
            print("Norm couldn't emerge")
            less_action = self.norm_checker.determine_less_norm_action()
            self.update_agents_q_values(less_action)

            self.reset_manager.keep_q_values()
            self.reset_manager.reset_to_final_q_values()

            self.run_simulation()

            if self.norm_checker.check_norm_emergence():
                print("When we execute again, norm emerged.")
                if drawPlot:
                    self.plot_simulation_results()
            else:
                print("When we execute again, norm couldn't emerge.")
                if drawPlot:
                    self.plot_simulation_results()

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
            self.reset_manager.reset_simulation(self)

        PlotManager.plot_aa_vs_bb_results(aa_wins, bb_wins)

    def simulation_different_agent_size(self, num_simulations=50):
        """Simulate different agent sizes and calculate norm emergence percentage over multiple runs."""
        agent_sizes = [40, 80, 120, 200]
        norm_counts_by_size = {size: [] for size in agent_sizes}

        for sim in range(num_simulations):
            print(f"Simulation {sim + 1}/{num_simulations}")

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
                    norm_counts_by_size[agent_size].append((count_A / self.num_agents) * 100)
                elif count_B >= self.num_agents * 0.9:
                    norm_counts_by_size[agent_size].append((count_B / self.num_agents) * 100)
                else:
                    norm_counts_by_size[agent_size].append(0)

                self.reset_manager.reset_simulation(self)

        avg_norm_counts = {
            size: (sum(values) / len(values)) if len(values) > 0 else 0
            for size, values in norm_counts_by_size.items()
        }

        PlotManager.plot_norm_emergence(agent_sizes, list(avg_norm_counts.values()))

    def plot_simulation_results(self):
        PlotManager.plot_action_combinations(self.action_combinations)
        PlotManager.plot_q_values(self.scores_history, self.num_agents)
        PlotManager.plot_agent_actions_graph(self.agents, self.grid_height, self.grid_width)

    def update_agents_q_values(self, action, trendsetters_ratio=0.5):
        """Update Q-values for agents who chose the less dominant action."""
        agents_choosing_action = [agent for agent in self.agents if agent.last_action == action]
        num_agents_to_update = int(len(agents_choosing_action) * trendsetters_ratio)
        agents_to_update = random.sample(agents_choosing_action, num_agents_to_update)

        for agent in agents_to_update:
            if action == 'B':
                agent.q_values['A'] = 1.0
                agent.q_values['B'] = -1.0
            else:
                agent.q_values['B'] = 1.0
                agent.q_values['A'] = -1.0

    def _update_action_combinations(self, count_AA, count_BB, count_AB, count_BA):
        """Track the action combinations over time."""
        self.action_combinations['AA'].append(count_AA)
        self.action_combinations['BB'].append(count_BB)
        self.action_combinations['AB'].append(count_AB)
        self.action_combinations['BA'].append(count_BA)

    def run_with_emergence_check_with_different_trendsetters(self, num_simulations=50):
        """Run the simulation with norm emergence check, averaging norm abandonment over multiple runs for trendsetter
        ratios."""
        trendsetters_ratios = [round(i * 0.1, 2) for i in range(1, 6)]
        abandonment_percentages_by_ratio = {ratio: [] for ratio in trendsetters_ratios}

        sim = 0

        while sim < num_simulations:
            print(f"Simulation {sim + 1}/{num_simulations}")
            self.run_simulation()

            if not self.norm_checker.check_norm_emergence():
                less_action = self.norm_checker.determine_less_norm_action()
                initial_q_values = {agent.agent_id: agent.q_values.copy() for agent in self.agents}
                initial_actions = {agent.agent_id: agent.last_action for agent in self.agents}

                for ratio in trendsetters_ratios:
                    for agent in self.agents:
                        agent.q_values = initial_q_values[agent.agent_id].copy()

                    self.update_agents_q_values(less_action, ratio)
                    self.reset_manager.keep_q_values()
                    self.reset_manager.reset_to_final_q_values()
                    self.run_simulation()

                    initial_less_action_agents = [agent for agent in self.agents if
                                                  initial_actions[agent.agent_id] == less_action]
                    abandonment_count = sum(
                        1 for agent in initial_less_action_agents if agent.last_action != less_action)
                    abandonment_percentage = (abandonment_count / len(initial_less_action_agents) * 100
                                              if len(initial_less_action_agents) > 0 else 0)
                    abandonment_percentages_by_ratio[ratio].append(abandonment_percentage)

                    self.reset_manager.reset_simulation(self)

                sim += 1
            else:
                print("Norm already emerged in initial run.")

        avg_abandonment_percentages = {
            ratio: (sum(values) / len(values)) if len(values) > 0 else 0
            for ratio, values in abandonment_percentages_by_ratio.items()
        }

        PlotManager.plot_abandonment_percentage(avg_abandonment_percentages)

