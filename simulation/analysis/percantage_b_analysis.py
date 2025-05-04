import os
import pandas as pd
from simulation.reset_manager import ResetManager
from simulation.simulation import Simulation


def run_multiple_simulations(agent_sizes, num_steps, k, p,
                             beta, trendsetter_percent, weight, epsilon,
                             num_simulations=1, topology_type="toroidal"):

    aa_wins = 0
    bb_wins = 0

    for sim in range(num_simulations):
        print(f"Running Simulation {sim + 1}/{num_simulations}")

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
        count_A ,count_B = 0
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

#        count_A = sum(1 for agent in simulation.agents if agent.last_action == 'A')
#        count_B = sum(1 for agent in simulation.agents if agent.last_action == 'B')

        percentage_A = (count_A / simulation.num_agents) * 100
        percentage_B = (count_B / simulation.num_agents) * 100

        if percentage_A >= 90:
            aa_wins += 1
        elif percentage_B >= 90:
            bb_wins += 1

        ResetManager.reset_simulation(simulation)

    b_emerged_percentage = bb_wins / num_simulations

    print(f"B norm emerged in {bb_wins}/{num_simulations} simulations")

    return {
        "A_emerged_count": aa_wins,
        "B_emerged_count": bb_wins,
        "Total_emerged": aa_wins + bb_wins,
        "B_emerged_percentage": b_emerged_percentage
    }


if __name__ == '__main__':
    import csv

    df = pd.read_excel("/Users/beyzasenol/Desktop/Norm-Emergence/MAS/norm-changes-emergence/inputs/percantage_b_emerged_100.xlsx")
    df["Weight"] = df["Weight"].astype(str)

    output_file = "/Users/beyzasenol/Desktop/Norm-Emergence/MAS/norm-changes-emergence/outputs/percentage_b_emergence_results_2.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # CSV başlığı bir kez yaz
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "A_emerged_count", "B_emerged_count", "Total_emerged", "B_emerged_percentage",
            "Agent Number", "Epsilon", "Trendsetter %", "Beta", "Weight"
        ])

    for _, row in df.iterrows():
        agent_sizes = int(row["Agent Number"])
        epsilon = float(str(row["Epsilon"]).replace(",", "."))
        trendsetter_percent = int(row["Trendsetter %"])
        beta = float(str(row["Beta"]).replace(",", "."))
        weight = eval(row["Weight"])

        num_steps = 1500
        k = 4
        p = 0.2

        result = run_multiple_simulations(agent_sizes, num_steps, k, p,
                                          beta, trendsetter_percent, weight, epsilon,
                                          num_simulations=10)

        result.update({
            "Agent Number": agent_sizes,
            "Epsilon": epsilon,
            "Trendsetter %": trendsetter_percent,
            "Beta": beta,
            "Weight": row["Weight"]
        })

        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                result["A_emerged_count"],
                result["B_emerged_count"],
                result["Total_emerged"],
                result["B_emerged_percentage"],
                result["Agent Number"],
                result["Epsilon"],
                result["Trendsetter %"],
                result["Beta"],
                result["Weight"]
            ])

        print(f"Saved result for Agent Number {agent_sizes} to {output_file}")
