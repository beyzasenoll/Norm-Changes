from visualization.plot_manager import PlotManager


class SimulationMultipleRunAnalysis:
    def __init__(self, simulation):
        self.simulation = simulation

    def run_multiple_simulations(self, num_simulations=20):
        """Run multiple simulations and track whether 'AA' or 'BB' dominates."""
        aa_wins = 0
        bb_wins = 0
        stepwise_percentages = []

        for sim in range(num_simulations):
            print(f"Running Simulation {sim + 1}/{num_simulations}")
            self.simulation.run_simulation()

            count_A = 0
            count_B = 0

            for agent in self.simulation.agents:
                if agent.last_action == 'A':
                    count_A += 1
                elif agent.last_action == 'B':
                    count_B += 1

            percentage_A = (count_A / self.simulation.num_agents) * 100
            percentage_B = (count_B / self.simulation.num_agents) * 100
            stepwise_percentages.append((percentage_A, percentage_B))

            if percentage_A >= 90:
                aa_wins += 1
            elif percentage_B >= 90:
                bb_wins += 1

            self.simulation.reset_manager.reset_simulation(self.simulation)

        PlotManager._plot_action_percentages(stepwise_percentages)
        PlotManager.plot_aa_vs_bb_results(aa_wins, bb_wins)
