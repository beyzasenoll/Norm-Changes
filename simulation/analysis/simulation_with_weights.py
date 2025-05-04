from simulation.simulation import Simulation
from simulation.analysis.multiple_run_analysis import SimulationMultipleRunAnalysis
import pandas as pd
import os

def run_custom_simulations(weight_sets, num_trials=10, num_agents=100, num_steps=1500):
    """
    Run simulations for predefined weight sets, each repeated multiple times.
    """
    results = []
    for weights in weight_sets:
        print(f"Running simulations for weights {weights}...")
        for trial in range(num_trials):
            simulation = Simulation(
                num_agents=num_agents,
                num_steps=num_steps,
                topology_type="toroidal",
                k=4,
                p=0.2,
                beta=0.5,
                circle_degree=[1, 2, 3],
                trendsetter_percent=10,
                epsilon=0.2,
                weights=weights
            )

            simulation_analysis = SimulationMultipleRunAnalysis(simulation)

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
                'Q Weight': weights[0],
                'E Weight': weights[1],
                'O Weight': weights[2],
                'Trial': trial + 1,
                'Percent of A': percent_A,
                'Percent of B': percent_B
            })
    return results



def save_results_to_csv(results, filename="/Users/beyzasenol/Desktop/Norm-Emergence/MAS/norm-changes-emergence/outputs/custom_weight_simulation_results.csv"):
    """
    Save the simulation results to a CSV file in the outputs directory.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

if __name__ == '__main__':
    weight_sets = [
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ]
    results = run_custom_simulations(weight_sets, num_trials=10)
    save_results_to_csv(results)