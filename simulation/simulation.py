import logging
import networkx as nx

from agents.agent import Agent
from environment.reward import Reward
from environment.topology import Topology
from simulation.norm_changes.norm_checker import NormChecker
from simulation.reset_manager import ResetManager
from visualization.plot_manager import PlotManager

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Simulation:
    def __init__(self, num_agents, num_steps, alpha=0.1, gamma=0.95, epsilon=0.1, temperature=100,
                 topology_type="toroidal"):
        self.pairs = None
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.topology_type = topology_type

        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.agents = [Agent(i, alpha, gamma, epsilon, temperature) for i in range(num_agents)]
        self.scores_history = [{'A': [], 'B': []} for _ in range(num_agents)]
        self.action_combinations = {'AA': [], 'BB': [], 'AB': [], 'BA': []}

        self.norm_checker = NormChecker(self.agents, self.num_agents)
        self.reset_manager = ResetManager(self.agents, self.num_agents)

        self.grid_width = 10
        self.grid_height = num_agents // self.grid_width
        self.topology = Topology()

        if topology_type == 'scale-free':
            self.scale_free_graph = nx.barabasi_albert_graph(num_agents, 2)

        logger.info(f"Simulation initialized with {num_agents} agents and {num_steps} steps.")

    def run_simulation(self, reward=None):
        """Run the simulation for a specified number of steps."""
        for step in range(self.num_steps):
            logger.info(f"Step {step}/{self.num_steps}")
            count_AA, count_BB, count_AB, count_BA = 0, 0, 0, 0

            self.pairs = self._generate_pairs()

            for agent1_id, agent2_id in self.pairs:
                agent1 = self.agents[agent1_id]
                agent2 = self.agents[agent2_id]

                action1 = agent1.choose_action_boltzmann(step)
                action2 = agent2.choose_action_boltzmann(step)

                logger.debug(f"Agent {agent1_id} chose {action1}, Agent {agent2_id} chose {action2}")

                if action1 == 'A' and action2 == 'A':
                    count_AA += 1
                elif action1 == 'B' and action2 == 'B':
                    count_BB += 1
                elif action1 == 'A' and action2 == 'B':
                    count_AB += 1
                elif action1 == 'B' and action2 == 'A':
                    count_BA += 1

                self._calculate_and_update_rewards_default(agent1, agent2, action1, action2)

            self._update_action_counts(count_AA, count_BB, count_AB, count_BA)
            logger.info(
                f"Step {step} completed. Action counts: AA={count_AA}, BB={count_BB}, AB={count_AB}, BA={count_BA}")

    def _generate_pairs(self):
        if self.topology_type == 'toroidal':
            return self.topology.form_pairs_with_toroidal_topology(self.num_agents, self.grid_height, self.grid_width)
        elif self.topology_type == 'scale-free':
            return self.topology.form_pairs_with_scale_free_topology(self.num_agents, self.scale_free_graph)
        elif self.topology_type == 'random':
            return self.topology.form_pairs_randomly(self.num_agents)
        else:
            raise ValueError(f"Unsupported topology type: {self.topology_type}")

    def _calculate_and_update_rewards_norm_change(self, agent1, agent2, action1, action2, reward):
        # Calculate rewards based on norm change
        logger.debug(f"Calculating rewards for action pair: ({action1}, {action2})")
        reward1, reward2 = Reward.calculate_rewards_norm_change_(
            action1, action2, self.norm_checker.less_action, reward
        )

        if reward1 is None or reward2 is None:
            logger.error(f"Reward values should not be None: reward1={reward1}, reward2={reward2}")
            raise ValueError(f"Reward values should not be None: reward1={reward1}, reward2={reward2}")

        # Update Q-values for both agents
        agent1.update_q_value(action1, reward1)
        agent2.update_q_value(action2, reward2)

        # Update scores history
        self.scores_history[agent1.agent_id]['A'].append(agent1.q_values['A'])
        self.scores_history[agent1.agent_id]['B'].append(agent1.q_values['B'])
        self.scores_history[agent2.agent_id]['A'].append(agent2.q_values['A'])
        self.scores_history[agent2.agent_id]['B'].append(agent2.q_values['B'])

    def _calculate_and_update_rewards_default(self, agent1, agent2, action1, action2):
        # Calculate rewards using default method
        logger.debug(f"Calculating default rewards for action pair: ({action1}, {action2})")
        reward1, reward2 = Reward.calculate_rewards(action1, action2)

        if reward1 is None or reward2 is None:
            logger.error(f"Reward values should not be None: reward1={reward1}, reward2={reward2}")
            raise ValueError(f"Reward values should not be None: reward1={reward1}, reward2={reward2}")

        # Update Q-values for both agents
        agent1.update_q_value(action1, reward1)
        agent2.update_q_value(action2, reward2)

        # Update scores history
        self.scores_history[agent1.agent_id]['A'].append(agent1.q_values['A'])
        self.scores_history[agent1.agent_id]['B'].append(agent1.q_values['B'])
        self.scores_history[agent2.agent_id]['A'].append(agent2.q_values['A'])
        self.scores_history[agent2.agent_id]['B'].append(agent2.q_values['B'])

    def _update_action_counts(self, count_AA, count_BB, count_AB, count_BA):
        self.action_combinations['AA'].append(count_AA)
        self.action_combinations['BB'].append(count_BB)
        self.action_combinations['AB'].append(count_AB)
        self.action_combinations['BA'].append(count_BA)

    def plot_simulation_results(self):
        logger.info("Plotting simulation results...")
        PlotManager.plot_action_combinations(self.action_combinations)
        PlotManager.plot_q_values(self.scores_history, self.num_agents)
        PlotManager.plot_agent_actions_graph(self.agents, self.grid_height, self.grid_width)
