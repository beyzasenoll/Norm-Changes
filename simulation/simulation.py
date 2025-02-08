import logging

from matplotlib import pyplot as plt

from agents.agent import Agent
from environment.reward import Reward
from environment.topology import Topology
from visualization.plot_manager import PlotManager

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Simulation:
    def __init__(self, num_agents, num_steps, topology_type='small_world', beta=0.5, k=4, p=0.2):
        self.action_combinations = {'AA': [], 'BB': [], 'AB': [], 'BA': []}
        self.scores_history = [{'A': [], 'B': []} for _ in range(num_agents)]
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.beta = beta
        self.topology = Topology(num_agents, topology_type=topology_type, k=k, p=p)
        self.pairs = self.topology.form_pairs_with_small_world_topology([1, 2])
        self.agents = [Agent(i, observation_beta=beta) for i in range(num_agents)]
        self.k = k
        self.p = p

    def run_simulation(self):
        """Run the simulation using both utility-based decision making and Q-learning updates."""
        for step in range(self.num_steps):
            logging.info(f"Step {step}: Running simulation step.")
            count_AA, count_BB, count_AB, count_BA = 0, 0, 0, 0
            self.pairs = self.topology.form_pairs_with_small_world_topology([1, 2])

            for agent1_id, agent2_id in self.pairs:
                agent1 = self.agents[agent1_id]
                agent2 = self.agents[agent2_id]

                action1 = agent1.choose_action()
                action2 = agent2.choose_action()

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
                agent2.update_q_value(action2, reward2)

                self.scores_history[agent1.agent_id]['A'].append(agent1.q_values['A'])
                self.scores_history[agent1.agent_id]['B'].append(agent1.q_values['B'])
                self.scores_history[agent2.agent_id]['A'].append(agent2.q_values['A'])
                self.scores_history[agent2.agent_id]['B'].append(agent2.q_values['B'])

                self._update_action_counts(count_AA, count_BB, count_AB, count_BA)
                logging.info(
                    f"Agent1 {agent1.agent_id}: Action {action1}, Reward {reward1}, Updated Q-Value {agent1.q_values[action1]}"
                    f"Agent2 {agent2.agent_id} : Action {action2}, Reward {reward2}, Updated Q-Values {agent2.q_values[action1]}")

    def _update_action_counts(self, count_AA, count_BB, count_AB, count_BA):
        self.action_combinations['AA'].append(count_AA)
        self.action_combinations['BB'].append(count_BB)
        self.action_combinations['AB'].append(count_AB)
        self.action_combinations['BA'].append(count_BA)

    def plot_simulation_results(self):
        logger.info("Plotting simulation results...")
        PlotManager.plot_action_combinations(self.action_combinations)
        PlotManager.plot_q_values(self.scores_history, self.num_agents, self.num_steps)
        PlotManager.plot_agent_actions_graph_small_world(self.agents, self.num_agents ,self.k,self.p)
