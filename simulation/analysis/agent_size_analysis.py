from agents.agent import Agent
from simulation.norm_changes.emergence_check import SimulationWithEmergence
from visualization.plot_manager import PlotManager


class SimulationAgentSizeAnalysis:
    def __init__(self, simulation):
        self.simulation = simulation

    def simulation_different_agent_size(self, num_simulations=50):
        """Simulate different agent sizes and calculate norm emergence percentage over multiple runs."""
        agent_sizes = [40, 80, 120, 200]
        norm_counts_by_size = {size: [] for size in agent_sizes}

        for sim in range(num_simulations):
            print(f"Simulation {sim + 1}/{num_simulations}")

            for agent_size in agent_sizes:
                self.simulation.num_agents = agent_size
                self.simulation.grid_height = (self.simulation.num_agents + self.simulation.grid_width - 1) // self.simulation.grid_width

                self.simulation.agents = [Agent(i, self.simulation.alpha, self.simulation.gamma, self.simulation.epsilon, self.simulation.temperature) for i in
                                   range(self.simulation.num_agents)]
                self.simulation.scores_history = [{'A': [], 'B': []} for _ in range(self.simulation.num_agents)]

                SimulationWithEmergence(self.simulation).run_with_emergence_check(False)

                count_A = sum(1 for agent in self.simulation.agents if agent.last_action == 'A')
                count_B = self.simulation.num_agents - count_A

                if count_A >= self.simulation.num_agents * 0.9:
                    norm_counts_by_size[agent_size].append((count_A / self.simulation.num_agents) * 100)
                elif count_B >= self.simulation.num_agents * 0.9:
                    norm_counts_by_size[agent_size].append((count_B / self.simulation.num_agents) * 100)

                self.simulation.reset_manager.reset_simulation(self.simulation)

        avg_norm_counts = {
            size: (sum(values) / len(values)) if len(values) > 0 else 0
            for size, values in norm_counts_by_size.items()
        }

        PlotManager.plot_norm_emergence(agent_sizes, list(avg_norm_counts.values()))
