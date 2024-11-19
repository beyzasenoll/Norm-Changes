import numpy as np

from simulation.norm_checker import NormChecker
from visualization.plot_manager import PlotManager


class NormChanges:
    def __init__(self, agents, reset_manager, norm_checker, simulation_instance):
        self.agents = agents
        self.reset_manager = reset_manager
        self.norm_checker = norm_checker
        self.simulation_instance = simulation_instance

    def run_after_update_reward(self, drawPlot=False, num_runs=50):
        reward_values = [-0.5, 0, 0.5]
        topologies = ["random", "toroidal"]
        abandonment_percentages_by_topology = {topology: [] for topology in topologies}

        for topology in topologies:
            print(f"Running simulation for topology: {topology}")
            self.simulation_instance.topology_type = topology
            all_abandonment_percentages = {reward: [] for reward in reward_values}

            for reward in reward_values:
                for _ in range(num_runs):
                    self.simulation_instance.run_with_emergence_check(drawPlot=False)
                    self.norm_checker.less_action = self.norm_checker.determine_less_norm_action()

                    print("---------------------------------------------------------------------------------")
                    print("NORM IS CHANGING")

                    self.reset_manager.keep_q_values()
                    self.reset_manager.reset_to_final_q_values()
                    self.norm_checker.norm_changed = True

                    print(f"Running simulation with norm changed and reward={reward} for topology {topology}")
                    self.simulation_instance.run_simulation(reward=reward)

                    abandonment_percentage = NormChecker.calculate_norm_abandonment(
                        self.simulation_instance, self.norm_checker.less_action
                    )
                    all_abandonment_percentages[reward].append(abandonment_percentage)

                    self.norm_checker.norm_changed = False
                    self.reset_manager.reset_simulation(self.simulation_instance)

                if drawPlot:
                    self.simulation_instance.plot_simulation_results()

            abandonment_percentages_by_topology[topology] = [
                np.mean(all_abandonment_percentages[reward]) for reward in reward_values
            ]

        PlotManager.plot_norm_abandonment_vs_reward_multiple_topologies(reward_values,
                                                                        abandonment_percentages_by_topology)
