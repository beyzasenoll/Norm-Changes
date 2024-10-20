from Simulation import Simulation

if __name__ == '__main__':
    num_agents = 100
    num_steps = 1500

    simulation = Simulation(num_agents, num_steps, alpha=0.1, gamma=0.95, epsilon=0.1, temperature=100, topology_type= "toroidal")
    simulation.run()
    simulation.plot_action_combinations()
    simulation.print_action_counts()
