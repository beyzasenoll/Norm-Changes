from itertools import product
import pandas as pd
import os
from simulation.simulation import Simulation

def save_single_result_to_csv(result, filename):
    """
    Append a single simulation result to a CSV file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame([result])
    file_exists = os.path.isfile(filename)

    df.to_csv(filename, mode='a', header=not file_exists, index=False)

def parameter_grid_search(
    agent_sizes, epsilons, trendsetters, betas, weights,
    num_trials=5, num_steps=1500,
    topology_type="toroidal", k=4, p=0.2,
    output_filename=None
):
    if output_filename is None:
        output_filename = "/Users/beyzasenol/Desktop/Norm-Emergence/MAS/norm-changes-emergence/outputs/grid_search_b_emergence.csv"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    combinations = list(product(agent_sizes,epsilons, trendsetters, betas, weights))

    for agent_sizes ,epsilon, trendsetter_percent, beta, weight in combinations:
        for trial in range(num_trials):
            simulation = Simulation(
                num_agents=agent_sizes,
                num_steps=num_steps,
                topology_type=topology_type,
                k=k,
                p=p,
                beta=beta,
                trendsetter_percent=trendsetter_percent,
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
                    count_B += 1
            percent_A = (count_A / agent_sizes) * 100
            percent_B = (count_B / agent_sizes) * 100

            result = {
                'Agent Number': agent_sizes,
                'Epsilon': epsilon,
                'Trendsetter %': trendsetter_percent,
                'Beta': beta,
                'Weight': str(weight),
                'Trial': trial + 1,
                'Percent of A': percent_A,
                'Percent of B': percent_B
            }
            save_single_result_to_csv(result, output_filename)

def analyze_results(filename):
    df = pd.read_csv(filename)
    grouped = df.groupby(['Epsilon', 'Trendsetter %', 'Beta', 'Weight'])['Percent of B'].mean().reset_index()
    best = grouped.loc[grouped['Percent of B'].idxmax()]

    print("\n En iyi parametre kombinasyonu:")
    print(f" - Epsilon: {best['Epsilon']}")
    print(f" - Trendsetter %: {best['Trendsetter %']}")
    print(f" - Beta: {best['Beta']}")
    print(f" - Weight: {best['Weight']}")
    print(f" - Ortalama Percent of B: {best['Percent of B']:.2f}")

if __name__ == '__main__':
    agent_sizes = [100]
    epsilons = [0.1,0.15,0.2]
    trendsetters = [2, 4, 6, 8, 10]
    betas = [0.3, 0.4, 0.5]
    weights = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    output_file = "/Users/beyzasenol/Desktop/Norm-Emergence/MAS/norm-changes-emergence/outputs/grid_search_b_emergence_new.csv"

    parameter_grid_search(
        agent_sizes,epsilons, trendsetters, betas, weights,
        num_trials=5,
        num_steps=1500,
        output_filename=output_file
    )

    print(f"\n CSV dosyası oluşturuldu: {output_file}")

    analyze_results(output_file)
