from agents.agent import Agent


class ResetManager:
    def __init__(self, agents, num_agents):
        """
        Initialize the ResetManager to manage and reset agent states.
        :param agents: List of Agent objects to be reset.
        :param num_agents: Total number of agents for resetting simulation history.
        """
        self.agents = agents
        self.num_agents = num_agents

    def keep_q_values(self):
        """Store the final Q-values of each agent to revert to after reset."""
        for agent in self.agents:
            agent.final_q_values = agent.q_values.copy()
            agent.fixed_q_values = True

    def reset_to_final_q_values(self):
        """Reset each agent's Q-values to the previously stored final Q-values."""
        for agent in self.agents:
            if agent.final_q_values:
                agent.q_values = agent.final_q_values.copy()

    @staticmethod
    def reset_simulation(self):
        """Reset the simulation to run it again with the same agents."""
        self.scores_history = [{'A': [], 'B': []} for _ in range(self.num_agents)]
        self.action_combinations = {'AA': [], 'BB': [], 'AB': [], 'BA': []}
        self.agents = [Agent(i, self.alpha, self.gamma, self.epsilon, self.temperature) for i in range(self.num_agents)]
