from simulation.simulation import Simulation

if __name__ == '__main__':
    num_agents = 100
    num_steps = 1000

    simulation = Simulation(
        num_agents=num_agents,
        num_steps=num_steps,
        topology_type="scale_free",
        k=4,
        p=0.2,
        beta=0,
        circle_degree=[1],
        trendsetter_percent=10,
        epsilon=0.1,
        weights=[1,0,0],
        trendsetter_choosing_type ='by degree',
        window_size=5
    )

    # Run simulation
    simulation.run_simulation()
    simulation.plot_simulation_results()

