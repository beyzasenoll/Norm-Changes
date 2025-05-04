
import pandas as pd
from simulation.simulation import Simulation

def run_simulation_with_params(params):
    sim = Simulation(
        num_agents=params['num_agents'],
        num_steps=1500,
        topology_type=params['topology_type'],
        beta=params['beta'],
        k=4,
        p=0.2,
        circle_degree=[1, 2],
        trendsetter_percent=params['trendsetter_percent']
    )

    for agent in sim.agents:
        agent.epsilon = params['epsilon']
        agent.weights = [
            params['weight_q'],
            params['weight_exp'],
            params['weight_obs']
        ]
        agent.observation_beta = params['beta']

    sim.run_simulation()

    count_A, count_B = 0
    for agent in sim.agents:
        actionCountA, actionCountB = 0, 0
        for action in agent.past_window['actions']:
            if action == 'A':
                actionCountA += 1
            elif action == 'B':
                actionCountB += 1
        if actionCountA > actionCountB:
            count_A += 1
        elif actionCountB > actionCountA:
            count_B += 1

    total_actions = count_A + count_B
    percent_A = 100 * count_A / total_actions
    percent_B = 100 * count_B / total_actions

    emerged = "Yes" if percent_A >= 90 or percent_B >= 90 else "No"


    return {
        "A": count_A,
        "B": count_B,
        "percent_A": percent_A,
        "percent_B": percent_B,
        "emerged": emerged
    }

if __name__ == "__main__":
    input_path = "/Users/beyzasenol/Desktop/Norm-Emergence/MAS/norm-changes-emergence/inputs/experiment_parameters_sample_500.xlsx"
    output_path = "/Users/beyzasenol/Desktop/Norm-Emergence/MAS/norm-changes-emergence/outputs/experiment_results1.xlsx"

    input_df = pd.read_excel(input_path)
    results = []

    for _, row in input_df.iterrows():
        params = row.to_dict()
        try:
            result = run_simulation_with_params(params)
            combined = {**params, **result}
            results.append(combined)
        except Exception as e:
            print(f"Error with params {params}: {e}")

    output_df = pd.DataFrame(results)
    output_df.to_excel(output_path, index=False)
