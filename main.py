from Simulation import Simulation

if __name__ == '__main__':
    num_agents = 80
    num_steps = 1500
    num_simulations = 300

    simulation = Simulation(num_agents, num_steps, alpha=0.05, gamma=0.95, epsilon=0.1, temperature=100,
                            topology_type="random")
    simulation.run_with_emergence_check()
    #simulation.run_multiple_simulations(num_simulations=num_simulations)