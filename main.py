from simulation.analysis.multiple_run_analysis import SimulationMultipleRunAnalysis
from simulation.simulation import Simulation


if __name__ == '__main__':
    num_agents = 40
    num_steps = 2000
    num_simulations = 10

    simulation = Simulation(
        num_agents=num_agents,
        num_steps=num_steps,
        topology_type="toroidal",
        k=4,
        p=0.2
    )
    SimulationMultipleRunAnalysis = SimulationMultipleRunAnalysis(simulation)

    # Run simulation
    simulation.run_simulation()
    simulation.plot_simulation_results()
    #SimulationMultipleRunAnalysis.run_multiple_simulations(num_simulations)
    #agent = Agent(agent_id=0, is_trendsetter=True)