from simulation.norm_changes import NormChanges
from simulation.simulation import Simulation
from simulation.norm_changes.emergence_check import SimulationWithEmergence
from simulation.analysis.trendsetter_analysis import SimulationTrendsetterAnalysis
from simulation.analysis.agent_size_analysis import SimulationAgentSizeAnalysis
from simulation.analysis.multiple_run_analysis import SimulationMultipleRunAnalysis

if __name__ == '__main__':
    num_agents = 200
    num_steps = 1500
    num_simulations = 10

    # Initialize the main simulation
    simulation = Simulation(
        num_agents=num_agents,
        num_steps=num_steps,
        alpha=0.05,
        gamma=0.95,
        epsilon=0.1,
        temperature=100,
        topology_type="toroidal"
    )

    # Initialize analysis modules
    norm_changes_instance = NormChanges(simulation.agents, simulation.reset_manager, simulation.norm_checker,
                                        simulation)
    emergence_analysis = SimulationWithEmergence(simulation)
    trendsetter_analysis = SimulationTrendsetterAnalysis(simulation)
    agent_size_analysis = SimulationAgentSizeAnalysis(simulation)
    multiple_run_analysis = SimulationMultipleRunAnalysis(simulation)

    # Run simulation
    simulation.run_simulation()
    simulation.plot_simulation_results()

    # Run a single simulation with emergence check
    # emergence_analysis.run_with_emergence_check()

    # Analyze trendsetters' impact
    # trendsetter_analysis.run_with_emergence_check_with_different_trendsetters(num_simulations=10)

    # Analyze norm emergence across different agent sizes
    # agent_size_analysis.simulation_different_agent_size(num_simulations=10)

    # Run simulations with reward adjustments for norm changes
    # norm_changes_instance.run_after_update_reward(num_runs=10)

    # Run multiple simulations and evaluate dominance outcomes
    # multiple_run_analysis.run_multiple_simulations(num_simulations=num_simulations)
