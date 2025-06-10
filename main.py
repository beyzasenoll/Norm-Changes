from simulation.simulation import Simulation

if __name__ == '__main__':
    num_agents = 100
    num_steps = 1500
    num_simulations = 15

    simulation = Simulation(
        num_agents=num_agents,
        num_steps=num_steps,
        topology_type="small_world",
        k=4,
        p=0.2,
        beta=0.05,
        circle_degree=[1,2,3],
        trendsetter_percent=5,
        epsilon=0.2,
        weights=[0, 0, 1],
        trendsetter_choosing_type = 'by degree',
        window_size=5
    )

    # Run simulation
    simulation.run_simulation()
    simulation.plot_simulation_results()

