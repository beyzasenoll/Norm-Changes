from Simulation import Simulation

if __name__ == '__main__':
    num_agents = 100
    num_steps = 300
    temperature = 0.5

    simulation = Simulation(num_agents, num_steps, temperature)
    simulation.run()
    simulation.plot_action_combinations()
    simulation.print_action_counts()
