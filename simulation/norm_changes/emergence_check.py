
class SimulationWithEmergence:
    def __init__(self, simulation):
        self.grid_height = simulation.grid_height
        self.grid_width = simulation.grid_width
        self.simulation = simulation

    def run_with_emergence_check(self, drawPlot=True):
        """Run the simulation with norm emergence check and reset if necessary."""
        self.simulation.run_simulation()
        if drawPlot:
            self.simulation.plot_simulation_results()

        if not self.simulation.norm_checker.check_norm_emergence():
            print("Norm couldn't emerge")
            less_action = self.simulation.norm_checker.determine_less_norm_action()
            self.update_agents_q_values(less_action)

            self.simulation.reset_manager.keep_q_values()
            self.simulation.reset_manager.reset_to_final_q_values()

            self.simulation.run_simulation()

            if self.simulation.norm_checker.check_norm_emergence():
                print("When we execute again, norm emerged.")
                if drawPlot:
                    self.simulation.plot_simulation_results()
            else:
                print("When we execute again, norm couldn't emerge.")
                if drawPlot:
                    self.simulation.plot_simulation_results()


    def update_agents_q_values(self, action, trendsetters_ratio=0.5):
        """Update Q-values for agents who chose the less dominant action, focusing on influential agents."""
        agents_choosing_action = [agent for agent in self.agents if agent.last_action == action]

        agents_sorted_by_influence = sorted(
            agents_choosing_action,
            key=lambda agent: self.calculate_influence(agent.agent_id),
            reverse=True
        )

        num_agents_to_update = int(len(agents_sorted_by_influence) * trendsetters_ratio)
        agents_to_update = agents_sorted_by_influence[:num_agents_to_update]

        for agent in agents_to_update:
            if action == 'B':
                agent.q_values['A'] = 1.0
                agent.q_values['B'] = -1.0
            else:
                agent.q_values['B'] = 1.0
                agent.q_values['A'] = -1.0

    def calculate_influence(self, agent_id):
        row = agent_id // self.grid_width
        col = agent_id % self.grid_width
        center_row = self.grid_height // 2
        center_col = self.grid_width // 2

        influence = 1 / (1 + abs(row - center_row) + abs(col - center_col))
        return influence
