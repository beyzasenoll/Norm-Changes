import os
import csv
import pandas as pd
from simulation.reset_manager import ResetManager
from simulation.simulation import Simulation

def run_multiple_simulations(agent_sizes, num_steps, k, p, beta, trendsetter_percent, weight, epsilon,
                             num_simulations=10, topology_type="toroidal"):
    aa_wins, bb_wins = 0, 0

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
        count_A, count_B = count_agent_actions(simulation)

        percentage_A = count_A / simulation.num_agents * 100
        percentage_B = count_B / simulation.num_agents * 100

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

def count_agent_actions(simulation):
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
    return count_A, count_B

if __name__ == '__main__':
    df = pd.read_excel("inputs/percentage_b_emerged_100.xlsx")
    df["Weight"] = df["Weight"].astype(str)
    output_file = "outputs/percentage_b_emergence_results.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "A_emerged_count", "B_emerged_count", "Total_emerged", "B_emerged_percentage",
            "Agent_Number", "Epsilon", "Trendsetter_Percent", "Beta", "Weight"
        ])

    for _, row in df.iterrows():
        agent_sizes = int(row["Agent Number"])
        epsilon = float(str(row["Epsilon"]).replace(",", "."))
        trendsetter_percent = int(row["Trendsetter %"])
        beta = float(str(row["Beta"]).replace(",", "."))
        weight = eval(row["Weight"])

        result = run_multiple_simulations(agent_sizes, 1500, 4, 0.2, beta, trendsetter_percent, weight, epsilon)

        result.update({
            "Agent_Number": agent_sizes,
            "Epsilon": epsilon,
            "Trendsetter_Percent": trendsetter_percent,
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
                result["Agent_Number"],
                result["Epsilon"],
                result["Trendsetter_Percent"],
                result["Beta"],
                result["Weight"]
            ])

        print(f"Saved result for Agent Number {agent_sizes} to {output_file}")
