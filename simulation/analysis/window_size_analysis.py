import os
import csv
import pandas as pd
from simulation.reset_manager import ResetManager
from simulation.simulation import Simulation


def run_multiple_simulations_for_window_size_analysis(agent_sizes, num_steps, k, p, beta, trendsetter_percent,
                                                      weight, epsilon, topology_type, trendsetter_choosing_type,
                                                      window_size, observable_window_size=10, num_simulations=50):
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
            trendsetter_choosing_type=trendsetter_choosing_type,
            window_size=observable_window_size
        )

        simulation.run_simulation()

        count_A, count_B = count_agent_actions(simulation, window_size)

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


def count_agent_actions(simulation, window_size, observable_window_size):
    count_A, count_B = 0, 0
    for agent in simulation.agents:
        actions = agent.past_window['actions']
        if not actions:
            continue

        if window_size == 1:
            last_action = actions[-1]
            if last_action == 'A':
                count_A += 1
            elif last_action == 'B':
                count_B += 1
        else:
            recent_actions = actions[-observable_window_size:]
            action_A = recent_actions.count('A')
            action_B = recent_actions.count('B')
            if action_A > action_B:
                count_A += 1
            elif action_B > action_A:
                count_B += 1
    return count_A, count_B


if __name__ == '__main__':
    df = pd.read_excel("/Users/beyzasenol/Desktop/norm-changes-emergence/inputs/check_for_window_size.xlsx")

    df.columns = df.columns.str.strip().str.replace(" ", "_")

    df["Weight"] = df["Weight"].astype(str)
    df["Epsilon"] = df["Epsilon"].astype(float)

    output_file = "/Users/beyzasenol/Desktop/Norm-Emergence/MAS/norm-changes-emergence/outputs/calculate_emerge_rate_with_trendsetter_type_and_weights.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Agent_Number", "Epsilon", "Trendsetter_Percent", "Beta", "Weight", "Topology", "Trendsetter_Choosing_Type",
            "Window_Size",
            "A_emerged_count", "B_emerged_count", "Total_emerged", "B_emerged_percentage"
        ])

    for _, row in df.iterrows():
        result = run_multiple_simulations_for_window_size_analysis(
            agent_sizes=int(row["Agent_Number"]),
            num_steps=1500,
            k=4,
            p=0.2,
            beta=float(row["Beta"]),
            trendsetter_percent=int(row["Trendsetter"]),
            weight=eval(row["Weight"]),
            epsilon=float(row["Epsilon"]),
            topology_type=row["Topology"],
            trendsetter_choosing_type=row["Trendsetter_Type"],
            window_size=row["Window_size"]
        )

        result.update({
            "Agent_Number": row["Agent_Number"],
            "Epsilon": row["Epsilon"],
            "Trendsetter_Percent": row["Trendsetter"],
            "Beta": row["Beta"],
            "Weight": row["Weight"],
            "Topology": row["Topology"],
            "Trendsetter_Choosing_Type": row["Trendsetter_Type"],
            "Window_Size": row["Window_Size"]
        })

        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                result["Agent_Number"],
                result["Epsilon"],
                result["Trendsetter_Percent"],
                result["Beta"],
                result["Weight"],
                result["Topology"],
                result["Trendsetter_Choosing_Type"],
                result["Window_Size"],
                result["A_emerged_count"],
                result["B_emerged_count"],
                result["Total_emerged"],
                result["B_emerged_percentage"]
            ])
