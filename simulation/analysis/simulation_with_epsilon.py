from simulation.simulation import Simulation
import pandas as pd
import numpy as np
import os

def run_simulations_varying_epsilon(weight, epsilon_values, num_trials=5, num_agents=100, num_steps=1500):
    results = []
    for epsilon in epsilon_values:
        for trial in range(num_trials):
            simulation = Simulation(
                num_agents=num_agents,
                num_steps=num_steps,
                topology_type="toroidal",
                k=4,
                p=0.2,
                beta=0.5,
                trendsetter_percent=10
            )


            for agent in simulation.agents:
                agent.weights = weight
                agent.epsilon = epsilon

            simulation.run_simulation()

            count_A, count_B = 0,0
            for agent in simulation.agents:
                actionCountA, actionCountB = 0, 0
                for action in agent.past_window['actions']:
                    if action == 'A':
                        actionCountA += 1
                    elif action == 'B':
                        actionCountB += 1
                if actionCountA > actionCountB:
                    count_A += 1
                elif actionCountB > actionCountA:
                    count_A += 1
            percent_A = (count_A / num_agents) * 100
            percent_B = (count_B / num_agents) * 100

            results.append({
                'Agent_size': num_agents,
                'Weight': weight,
                'Epsilon': epsilon,
                'Trial': trial + 1,
                'Percent of A': percent_A,
                'Percent of B': percent_B
            })
    return results

def save_results_to_csv(results, filename):
    """
    Save the simulation results to a CSV file in the outputs directory.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

if __name__ == '__main__':
    outputs_dir = "/Users/beyzasenol/Desktop/Norm-Emergence/MAS/norm-changes-emergence/outputs/custom_epsilon_simulation_results.csv"
    epsilon_values = np.round(np.arange(0, 0.55, 0.05), 2)
    fixed_weight = [0, 0, 1]
    results = run_simulations_varying_epsilon(fixed_weight, epsilon_values)
    save_results_to_csv(results,outputs_dir)
