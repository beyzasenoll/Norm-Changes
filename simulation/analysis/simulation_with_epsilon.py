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
            count_A, count_B = count_agent_actions(simulation, num_agents)
            results.append({
                'Agent_Size': num_agents,
                'Weight': weight,
                'Epsilon': epsilon,
                'Trial': trial + 1,
                'Percent_A': count_A,
                'Percent_B': count_B
            })
    return results

def count_agent_actions(simulation, num_agents):
    count_A, count_B = 0, 0
    for agent in simulation.agents:
        action_A, action_B = 0, 0
        for action in agent.past_window['actions']:
            if action == 'A':
                action_A += 1
            elif action == 'B':
                action_B += 1
        if action_A > action_B:
            count_A += 1
        elif action_B > action_A:
            count_B += 1
    percent_A = (count_A / num_agents) * 100
    percent_B = (count_B / num_agents) * 100
    return percent_A, percent_B

def save_results_to_csv(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pd.DataFrame(results).to_csv(filename, index=False)
    print(f"Results saved to {filename}")

if __name__ == '__main__':
    output_file = "outputs/epsilon_variation_results.csv"
    epsilon_values = np.round(np.arange(0, 0.55, 0.05), 2)
    fixed_weight = [0, 0, 1]
    results = run_simulations_varying_epsilon(fixed_weight, epsilon_values)
    save_results_to_csv(results, output_file)
