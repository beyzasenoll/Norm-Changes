from visualization.plot_manager import PlotManager
from simulation.norm_changes.emergence_check import SimulationWithEmergence


class SimulationTrendsetterAnalysis:
    def __init__(self, simulation):
        self.simulation = simulation

    def run_with_emergence_check_with_different_trendsetters(self, num_simulations=50):
        """Run the simulation with norm emergence check, averaging norm abandonment over multiple runs for trendsetter
        ratios."""
        trendsetters_ratios = [round(i * 0.1, 2) for i in range(1, 6)]
        abandonment_percentages_by_ratio = {ratio: [] for ratio in trendsetters_ratios}

        sim = 0

        while sim < num_simulations:
            print(f"Simulation {sim + 1}/{num_simulations}")
            self.simulation.run_simulation()

            if not self.simulation.norm_checker.check_norm_emergence():
                less_action = self.simulation.norm_checker.determine_less_norm_action()
                initial_q_values = {agent.agent_id: agent.q_values.copy() for agent in self.simulation.agents}
                initial_actions = {agent.agent_id: agent.last_action for agent in self.simulation.agents}

                for ratio in trendsetters_ratios:
                    for agent in self.simulation.agents:
                        agent.q_values = initial_q_values[agent.agent_id].copy()

                    SimulationWithEmergence.update_agents_q_values(less_action, ratio)
                    self.simulation.reset_manager.keep_q_values()
                    self.simulation.reset_manager.reset_to_final_q_values()
                    self.simulation.run_simulation()

                    initial_less_action_agents = [agent for agent in self.simulation.agents if
                                                  initial_actions[agent.agent_id] == less_action]
                    abandonment_count = sum(
                        1 for agent in initial_less_action_agents if agent.last_action != less_action)
                    abandonment_percentage = (abandonment_count / len(initial_less_action_agents) * 100
                                              if len(initial_less_action_agents) > 0 else 0)
                    abandonment_percentages_by_ratio[ratio].append(abandonment_percentage)

                    self.simulation.reset_manager.reset_simulation(self.simulation)

                sim += 1
            else:
                print("Norm already emerged in initial run.")

        avg_abandonment_percentages = {
            ratio: (sum(values) / len(values)) if len(values) > 0 else 0
            for ratio, values in abandonment_percentages_by_ratio.items()
        }

        PlotManager.plot_abandonment_percentage(avg_abandonment_percentages)