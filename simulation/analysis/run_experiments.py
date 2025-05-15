import os
import csv
import pandas as pd
from simulation.reset_manager import ResetManager
from simulation.simulation import Simulation

def run_multiple_simulations(agent_sizes, num_steps, k, p, beta, trendsetter_percent,
                             weight, epsilon, trendsetter_choosing_type, num_simulations=50, topology_type="scale_free"):
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
            epsilon=epsilon,
            trendsetter_choosing_type=trendsetter_choosing_type
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

    b_emerged_percentage = bb_wins / num_simulations
    return {
        "A_emerged_count": aa_wins,
        "B_emerged_count": bb_wins,
        "Total_emerged": aa_wins + bb_wins,
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
    df = pd.read_excel("/Users/beyzasenol/Desktop/Norm-Emergence/MAS/norm-changes-emergence/inputs/b_emergence_check_new.xlsx")

    df.columns = df.columns.str.strip().str.replace(" ", "_")

    df["Weight"] = df["Weight"].astype(str)
    df["Epsilon"] = df["Epsilon"].astype(float)

    output_file = "/Users/beyzasenol/Desktop/Norm-Emergence/MAS/norm-changes-emergence/outputs/calculate_emerge_rate_with_trendsetter_type_results_scale_free.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Agent_Number", "Epsilon", "Trendsetter_Percent", "Beta", "Weight", "Trendsetter_Choosing_Type",
            "A_emerged_count", "B_emerged_count", "Total_emerged", "B_emerged_percentage"
        ])

    for _, row in df.iterrows():
        result = run_multiple_simulations(
            agent_sizes=int(row["Agent_Number"]),
            num_steps=1500,
            k=4,
            p=0.2,
            beta=float(row["Beta"]),
            trendsetter_percent=int(row["Trendsetter"]),
            weight=eval(row["Weight"]),
            epsilon=float(row["Epsilon"]),
            trendsetter_choosing_type=row["Trendsetter_Type"]
        )

        result.update({
            "Agent_Number": row["Agent_Number"],
            "Epsilon": row["Epsilon"],
            "Trendsetter_Percent": row["Trendsetter"],
            "Beta": row["Beta"],
            "Weight": row["Weight"],
            "Trendsetter_Choosing_Type": row["Trendsetter_Type"]
        })

        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                result["Agent_Number"],
                result["Epsilon"],
                result["Trendsetter_Percent"],
                result["Beta"],
                result["Weight"],
                result["Trendsetter_Choosing_Type"],
                result["A_emerged_count"],
                result["B_emerged_count"],
                result["Total_emerged"],
                result["B_emerged_percentage"]
            ])
