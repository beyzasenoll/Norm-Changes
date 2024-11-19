from simulation.norm_changes import NormChanges
from simulation.Simulation import Simulation

if __name__ == '__main__':
    num_agents = 200
    num_steps = 1500
    num_simulations = 10

    simulation = Simulation(num_agents, num_steps, alpha=0.05, gamma=0.95, epsilon=0.1, temperature=100,
                            topology_type="toroidal")
    norm_changes_instance = NormChanges(simulation.agents, simulation.reset_manager, simulation.norm_checker, simulation)

    simulation.run_with_emergence_check()
    #simulation.run_with_emergence_check_with_different_trendsetters(num_simulations=10)
    #simulation.simulation_different_agent_size(num_simulations=10)
    #norm_changes_instance.run_after_update_reward(num_runs=10)
    #simulation.run_multiple_simulations(num_simulations=num_simulations)

    ### toroidal da % kaçı emerge ediyor
    ###toroidal grid ile norm emergence ile çalışma
    ### trendsetter dağılımı
