import os
import pandas as pd
from simulation.reset_manager import ResetManager
from simulation.simulation import Simulation
import csv

def run_multiple_simulations(params):

    aa_wins = 0
    bb_wins = 0


    simulation = Simulation(
            num_agents=params['Agent Number'],
            num_steps=1500,
            topology_type= 'toroidal',
            k=4,
            p=0.2,
            beta=params['Beta'],
            trendsetter_percent=params['Trendsetter %'],
            weights= [0,0,1],
            epsilon=params['Epsilon']
        )

    simulation.run_simulation()

    count_A, count_B = 0
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

    percentage_A = (count_A / simulation.num_agents) * 100
    percentage_B = (count_B / simulation.num_agents) * 100

    if percentage_A >= 90:
        aa_wins += 1
    elif percentage_B >= 90:
        bb_wins += 1

    ResetManager.reset_simulation(simulation)

    print(f"B norm emerged in {bb_wins} simulations")

    return {
        "num_agents" :params['Agent Number'],
        "num_steps" : 3000,
        "topology_type" : 'toroidal',
        "beta" : params['Beta'],
        "trendsetter_percent" : params['Trendsetter %'],
        "weight": [0, 0, 1],
        "epsilon" : params['Epsilon'],
        "A emerge percantage": percentage_A,
        "B emerge percantage": percentage_B,
    }


if __name__ == '__main__':
    input_path = "/Users/beyzasenol/Desktop/Norm-Emergence/MAS/norm-changes-emergence/inputs/b_emergence_check.xlsx"
    output_path = "/Users/beyzasenol/Desktop/Norm-Emergence/MAS/norm-changes-emergence/outputs/b_emergence_check.xlsx"

    input_df = pd.read_excel(input_path)
    results = []

    for _, row in input_df.iterrows():
        params = row.to_dict()
        try:
            result = run_multiple_simulations(params)
            combined = {**params, **result}
            results.append(combined)
        except Exception as e:
            print(f"Error with params {params}: {e}")

    output_df = pd.DataFrame(results)
    output_df.to_excel(output_path, index=False)



