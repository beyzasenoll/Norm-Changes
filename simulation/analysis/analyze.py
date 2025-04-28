from itertools import product
import pandas as pd
import os
from simulation.simulation import Simulation  # Kendi Simulation sınıfını burada kullan


def parameter_grid_search(
        epsilons, trendsetters, betas, weights,
        num_trials=5, num_agents=100, num_steps=1000,
        topology_type="toroidal", k=4, p=0.2
):
    results = []
    combinations = list(product(epsilons, trendsetters, betas, weights))

    for epsilon, trendsetter_percent, beta, weight in combinations:
        for trial in range(num_trials):
            simulation = Simulation(
                num_agents=num_agents,
                num_steps=num_steps,
                topology_type=topology_type,
                k=k,
                p=p,
                beta=beta,
                trendsetter_percent=trendsetter_percent
            )
            for agent in simulation.agents:
                agent.weights = weight
                agent.epsilon = epsilon

            simulation.run_simulation()

            last_actions = [agent.last_action for agent in simulation.agents]
            percent_A = (last_actions.count('A') / num_agents) * 100
            percent_B = (last_actions.count('B') / num_agents) * 100

            results.append({
                'Epsilon': epsilon,
                'Trendsetter %': trendsetter_percent,
                'Beta': beta,
                'Weight': str(weight),
                'Trial': trial + 1,
                'Percent of A': percent_A,
                'Percent of B': percent_B
            })
    return results


def analyze_results(df):
    grouped = df.groupby(['Epsilon', 'Trendsetter %', 'Beta', 'Weight'])['Percent of B'].mean().reset_index()
    best = grouped.loc[grouped['Percent of B'].idxmax()]

    print("\n🔍 En iyi parametre kombinasyonu:")
    print(f" - Epsilon: {best['Epsilon']}")
    print(f" - Trendsetter %: {best['Trendsetter %']}")
    print(f" - Beta: {best['Beta']}")
    print(f" - Weight: {best['Weight']}")
    print(f" - Ortalama Percent of B: {best['Percent of B']:.2f}")


if __name__ == '__main__':
    epsilons = [0.1, 0.15, 0.2, 0.25, 0.3 ,0.4]
    trendsetters = [0, 10,15, 20,25,30]
    betas = [0.2,0.3,0.4,0.5,0.6,0.7]
    weights = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0.1, 0.1, 0.8]
    ]

    results = parameter_grid_search(epsilons, trendsetters, betas, weights)

    # Excel'e yaz
    output_dir = "../../outputs/outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "grid_search_b_emergence.xlsx")
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)
    print(f"\n📁 Excel oluşturuldu: {output_path}")
