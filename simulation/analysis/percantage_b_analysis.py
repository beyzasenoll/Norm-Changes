import os
import csv
import pandas as pd
import ast
from simulation.reset_manager import ResetManager
from simulation.simulation import Simulation


def run_multiple_simulations(agent_sizes, num_steps, k, p, beta, trendsetter_percent, weight, epsilon,
                             num_simulations=1, topology_type="toroidal"):
    aa_wins, bb_wins = 0, 0

    for _ in range(num_simulations):
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
        count_A, count_B = count_agent_actions(simulation)

        percentage_A = count_A / simulation.num_agents * 100
        percentage_B = count_B / simulation.num_agents * 100

        if percentage_A >= 90:
            aa_wins += 1
        elif percentage_B >= 90:
            bb_wins += 1

        ResetManager.reset_simulation(simulation)

    total_emerged = aa_wins + bb_wins
    b_emerged_percentage = bb_wins / num_simulations

    return {
        "A_emerged_count": aa_wins,
        "B_emerged_count": bb_wins,
        "Total_emerged": total_emerged,
        "B_emerged_percentage": b_emerged_percentage
    }


def count_agent_actions(simulation):
    count_A, count_B = 0, 0
    for agent in simulation.agents:
        action_A = agent.past_window['actions'].count('A')
        action_B = agent.past_window['actions'].count('B')
        if action_A > action_B:
            count_A += 1
        elif action_B > action_A:
            count_B += 1
    return count_A, count_B


if __name__ == '__main__':
    # Excel dosyasını oku
    df = pd.read_excel(
        "/Users/beyzasenol/Desktop/Norm-Emergence/MAS/norm-changes-emergence/inputs/try_100_for_all_topologies.xlsx")

    # Weight sütununu doğru formata çevir
    df["Weight"] = df["Weight"].astype(str)

    # Çıktı dosyası
    output_file = ("/Users/beyzasenol/Desktop/Norm-Emergence/MAS/norm-changes-emergence/outputs"
                   "/try_100_random_results_new.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Başlıkları yaz
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Topology", "A_emerged_count", "B_emerged_count", "Total_emerged", "B_emerged_percentage",
            "Agent_Number", "Epsilon", "Trendsetter_Percent", "Beta", "Weight"
        ])

    # Her satır için simülasyonları çalıştır
    for _, row in df.iterrows():
        agent_sizes = int(row["Agent_Number"])
        epsilon = float(str(row["Epsilon"]).replace(",", "."))
        trendsetter_percent = int(row["Trendsetter_Percent"])
        beta = float(str(row["Beta"]).replace(",", "."))
        weight = ast.literal_eval(row["Weight"])  # Güvenli dönüşüm
        topology_type = str(row["Topology"])

        # Simülasyonları çalıştır
        result = run_multiple_simulations(agent_sizes, 3000, 4, 0.2, beta, trendsetter_percent, weight, epsilon,
                                          topology_type=topology_type)

        # Sonuçları dosyaya yaz
        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                topology_type,
                result["A_emerged_count"],
                result["B_emerged_count"],
                result["Total_emerged"],
                result["B_emerged_percentage"],
                agent_sizes,
                epsilon,
                trendsetter_percent,
                beta,
                row["Weight"]
            ])