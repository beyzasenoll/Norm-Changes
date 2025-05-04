from simulation.simulation import Simulation
import pandas as pd
import numpy as np
import os


def run_simulations_varying_beta(weight, beta_values, epsilon=0.15, num_trials=5, num_agents=100, num_steps=1500):
    results = []
    for beta in beta_values:
        for trial in range(num_trials):
            simulation = Simulation(
                num_agents=num_agents,
                num_steps=num_steps,
                topology_type="toroidal",
                k=4,
                p=0.2,
                beta=beta,
                trendsetter_percent=10
            )

            for agent in simulation.agents:
                agent.weights = weight
                agent.epsilon = epsilon

            simulation.run_simulation()

            last_actions = [agent.last_action for agent in simulation.agents]
            percent_A = (last_actions.count('A') / num_agents) * 100
            percent_B = (last_actions.count('B') / num_agents) * 100

            results.append({
                'Agent_size': num_agents,
                'Weight': weight,
                'Epsilon': epsilon,
                'Beta': beta,
                'Trial': trial + 1,
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
    output_dir = "beta_variation_simulation_results.xlsx"
    beta_values = np.round(np.arange(0.2, 0.8, 0.1), 2)  # [0.2, 0.3, ..., 0.7]
    fixed_weight = [0, 0, 1]
    results = run_simulations_varying_beta(fixed_weight, beta_values)
    save_results_to_csv(results, output_dir)
