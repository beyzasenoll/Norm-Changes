import os
import csv
import pandas as pd
from simulation.reset_manager import ResetManager
from simulation.simulation import Simulation


def run_multiple_simulations(agent_sizes, num_steps, k, p, beta, trendsetter_percent,
                             weight, epsilon, trendsetter_choosing_type, topology_type, circle_degree, window_size,
                             num_simulations=50):
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
            circle_degree=circle_degree,
            weights=weight,
            epsilon=epsilon,
            trendsetter_choosing_type=trendsetter_choosing_type,
            window_size=window_size
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
    input_output_paths = [
        {
            "input": "/Users/beyzasenol/Desktop/norm-changes-emergence/inputs/new_observation/window_size_analysis.xlsx",
            "output": "/Users/beyzasenol/Desktop/norm-changes-emergence/outputs/new_observations/window_size_analysis_result21.csv"
        },
    ]

    for paths in input_output_paths:
        input_path = paths["input"]
        output_path = paths["output"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df = pd.read_excel(input_path)
        df.columns = df.columns.str.strip().str.replace(" ", "_")

        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Agent_Number", "Topology", "Beta", "Circle_Degree", "Trendsetter_Percent",
                "Epsilon", "Weight", "Trendsetter_Choosing_Type", "Window_Size",
                "A_emerged_count", "B_emerged_count", "Total_emerged", "B_emerged_percentage"
            ])

        for _, row in df.iterrows():
            result = run_multiple_simulations(
                agent_sizes=int(row["num_agents"]),
                num_steps=1500,
                k=4,
                p=0.2,
                beta=float(row["beta"]),
                trendsetter_percent=int(row["trendsetter_percent"]),
                weight=eval(str(row["weights"])),
                epsilon=float(row["epsilon"]),
                trendsetter_choosing_type=row["trendsetter_choosing_type"],
                topology_type=row["topology_type"],
                circle_degree=eval(str(row["circle_degree"])),
                window_size=int(row["window_size"])
            )

            result.update({
                "Agent_Number": row["num_agents"],
                "Topology": row["topology_type"],
                "Beta": row["beta"],
                "Circle_Degree": row["circle_degree"],
                "Trendsetter_Percent": row["trendsetter_percent"],
                "Epsilon": row["epsilon"],
                "Weight": row["weights"],
                "Trendsetter_Choosing_Type": row["trendsetter_choosing_type"],
                "Window_Size": row["window_size"]
            })

            with open(output_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    result["Agent_Number"],
                    result["Topology"],
                    result["Beta"],
                    result["Circle_Degree"],
                    result["Trendsetter_Percent"],
                    result["Epsilon"],
                    result["Weight"],
                    result["Trendsetter_Choosing_Type"],
                    result["Window_Size"],
                    result["A_emerged_count"],
                    result["B_emerged_count"],
                    result["Total_emerged"],
                    result["B_emerged_percentage"]
                ])

