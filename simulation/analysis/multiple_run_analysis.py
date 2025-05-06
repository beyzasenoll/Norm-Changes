from simulation.reset_manager import ResetManager
from simulation.simulation import Simulation
from visualization.plot_manager import PlotManager

class SimulationMultipleRunAnalysis:
    def __init__(self, simulation):
        self.simulation = simulation

    def run_multiple_simulations(self, num_simulations=3):
        aa_wins, bb_wins = 0, 0
        stepwise_percentages = []

        for sim in range(num_simulations):
            print(f"Running Simulation {sim + 1}/{num_simulations}")
            self.simulation.run_simulation()
            count_A, count_B = self.count_agent_actions()

            percentage_A = count_A / self.simulation.num_agents * 100
            percentage_B = count_B / self.simulation.num_agents * 100
            stepwise_percentages.append((percentage_A, percentage_B))

            if percentage_A >= 90:
                aa_wins += 1
            elif percentage_B >= 90:
                bb_wins += 1

            ResetManager.reset_simulation(self.simulation)

        print(f"Total emerged: {aa_wins + bb_wins}")
        PlotManager.plot_aa_vs_bb_results(aa_wins, bb_wins)

    def count_agent_actions(self):
        count_A, count_B = 0, 0
        for agent in self.simulation.agents:
            action_A, action_B = 0, 0
            for action in agent.past_window['actions']:
                if action == 'A':
                    action_A += 1
                elif action == 'B':
                    action_B += 1
            if action_A > action_B:
                count_A += 1
            elif action_B > action_A:
                count_B += 1
        return count_A, count_B
