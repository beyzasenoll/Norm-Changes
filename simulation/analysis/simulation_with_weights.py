from simulation.simulation import Simulation
import pandas as pd
import os

def run_simulations_varying_weights(weight_sets, num_trials=10, num_agents=100, num_steps=1500):
    results = []
    for weights in weight_sets:
        print(f"Running simulations for weights {weights}...")
        for trial in range(num_trials):
            simulation = Simulation(
                num_agents=num_agents,
                num_steps=num_steps,
                topology_type="random",
                k=4,
                p=0.2,
                beta=0.5,
                trendsetter_percent=10,
                epsilon=0.2,
                weights=weights
            )
            simulation.run_simulation()
            count_A, count_B = count_agent_actions(simulation, num_agents)
            results.append({
                'Q_Weight': weights[0],
                'E_Weight': weights[1],
                'O_Weight': weights[2],
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
    output_file = "../../outputs/weight_variation_results_random_topology.csv"
    weight_sets = [[1, 0, 0],[0.5,0.5,0] ,[0, 1, 0]]
    results = run_simulations_varying_weights(weight_sets)
    save_results_to_csv(results, output_file)
