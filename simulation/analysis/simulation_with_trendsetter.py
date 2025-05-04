from simulation.simulation import Simulation
import pandas as pd
import numpy as np
import os

def run_simulations_varying_trendsetter(
    weight, trendsetter_percents,
    beta=0.4, epsilon=0.15,
    num_trials=5, num_agents=100, num_steps=1500
):
    results = []
    for percent in trendsetter_percents:
        for trial in range(num_trials):
            simulation = Simulation(
                num_agents=num_agents,
                num_steps=num_steps,
                topology_type="toroidal",
                k=4,
                p=0.2,
                beta=beta,
                trendsetter_percent=percent,
                weights=weight,
                epsilon=epsilon
            )

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
                'Trendsetter Percent': percent,
                'Trial': trial + 1,
                'Beta': beta,
                'Epsilon': epsilon,
                'Percent of A': percent_A,
                'Percent of B': percent_B
            })
    return results

def save_results_to_csv(results, filename="outputs/trendsetter_variation_simulation_results.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

if __name__ == '__main__':
    trendsetter_percents = [2,4,6,8,10]
    fixed_weight = [0, 0, 1]
    results = run_simulations_varying_trendsetter(fixed_weight, trendsetter_percents)
    save_results_to_csv(results)
