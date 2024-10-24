from Simulation import Simulation

if __name__ == '__main__':
    num_agents = 40
    num_steps = 1500
    num_simulations = 300

    simulation = Simulation(num_agents, num_steps, alpha=0.1, gamma=0.95, epsilon=0.1, temperature=100,
                            topology_type="toroidal")
    simulation.run()
    simulation.plot_action_combinations()
    simulation.plot_q_values()
    simulation.run_multiple_simulations(num_simulations=num_simulations)

