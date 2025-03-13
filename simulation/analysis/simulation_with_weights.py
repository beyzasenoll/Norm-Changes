import numpy as np
from simulation.simulation import Simulation


def generate_weight_combinations(step_size=0.1):
    """Generate weight combinations with a given step size."""
    combinations = []
    num_steps = int(1 / step_size) + 1

    for q in range(num_steps):
        for e in range(num_steps - q):
            o = (num_steps - 1) - q - e
            combinations.append([q * step_size, e * step_size, o * step_size])

    return combinations

def run_simulations_with_weights(num_agents=40, num_steps=2000, step_size=0.1):
    """Run simulations with different weight combinations and collect results."""
    weight_combinations = generate_weight_combinations(step_size)
    results = []

    for weights in weight_combinations:
        simulation = Simulation(
            num_agents=num_agents,
            num_steps=num_steps,
            topology_type="toroidal",
            k=4,
            p=0.2
        )
        # Update agent weights
        for agent in simulation.agents:
            agent.weights = weights

        # Run simulation
        simulation.run_simulation()

        # Calculate percentage of A and B actions
        last_actions = [agent.last_action for agent in simulation.agents]
        percent_A = (last_actions.count('A') / num_agents) * 100
        percent_B = (last_actions.count('B') / num_agents) * 100

        results.append({
            'weights': weights,
            'percent_A': percent_A,
            'percent_B': percent_B
        })

    return results

def save_results_to_txt(results, filename="simulation_results.txt"):
    """Save the simulation results to a text file."""
    with open(filename, 'w') as file:
        file.write("Simulation Results:\n")
        file.write("------------------\n")
        for result in results:
            weights = result['weights']
            percent_A = result['percent_A']
            percent_B = result['percent_B']
            file.write(
                f"With weights [Q: {weights[0]:.2f}, E: {weights[1]:.2f}, O: {weights[2]:.2f}], "
                f"{percent_A:.2f}% of agents chose action A, "
                f"{percent_B:.2f}% of agents chose action B.\n"
            )
        file.write("------------------\n")
        file.write("End of results.\n")

if __name__ == '__main__':
    results = run_simulations_with_weights(num_agents=40, num_steps=1500, step_size=0.1)
    save_results_to_txt(results)