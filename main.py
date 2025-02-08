from simulation.simulation import Simulation


if __name__ == '__main__':
    num_agents = 40
    num_steps = 1500
    num_simulations = 10

    # Initialize the main simulation
    simulation = Simulation(
        num_agents=num_agents,
        num_steps=num_steps,
        k=4,
        p=0.2
    )

    # Run simulation
    simulation.run_simulation()
    simulation.plot_simulation_results()