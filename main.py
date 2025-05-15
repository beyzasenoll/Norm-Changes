from simulation.analysis.multiple_run_analysis import SimulationMultipleRunAnalysis
from simulation.simulation import Simulation

if __name__ == '__main__':
    num_agents = 49
    num_steps = 6000
    num_simulations = 50

    simulation = Simulation(
        num_agents=num_agents,
        num_steps=num_steps,
        topology_type="random",
        k=4,
        p=0.2,
        beta=0.5,
        circle_degree=[1, 2, 3],
        trendsetter_percent=8,
        epsilon=0.2,
        weights=[0, 1, 0],
        distance_type="close"
    )
    SimulationMultipleRunAnalysis = SimulationMultipleRunAnalysis(simulation)

    # Run simulation
    ##SimulationMultipleRunAnalysis.run_multiple_simulations(num_simulations)
    simulation.run_simulation()
    simulation.plot_simulation_results()

    # SimulationMultipleRunAnalysis.run_multiple_simulations(num_simulations)
    # agent = Agent(agent_id=0, is_trendsetter=True)
