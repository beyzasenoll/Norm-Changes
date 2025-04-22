from simulation.simulation import Simulation
import pandas as pd
import numpy as np
import os

def run_simulations_varying_trendsetter(weight, trendsetter_percents, beta=0.4, epsilon=0.15, num_trials=5, num_agents=100, num_steps=1500):
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
                trendsetter_percent=percent
            )

            for agent in simulation.agents:
                agent.weights = weight
                agent.epsilon = epsilon

            simulation.run_simulation()

            last_actions = [agent.last_action for agent in simulation.agents]
            percent_A = (last_actions.count('A') / num_agents) * 100
            percent_B = (last_actions.count('B') / num_agents) * 100

            results.append({
                'Trendsetter Percent': percent,
                'Trial': trial + 1,
                'Beta': beta,
                'Epsilon': epsilon,
                'Percent of A': percent_A,
                'Percent of B': percent_B
            })
    return results

def save_results_to_excel(results, filename="outputs/trendsetter_variation_simulation_results.xlsx"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)

if __name__ == '__main__':
    trendsetter_percents = np.arange(0, 45, 5)  # [0, 5, 10, ..., 40]
    fixed_weight = [0, 0, 1]
    results = run_simulations_varying_trendsetter(fixed_weight, trendsetter_percents)
    save_results_to_excel(results)
