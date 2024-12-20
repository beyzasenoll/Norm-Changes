class NormChecker:
    def __init__(self, agents,num_agents, threshold=0.9):
        """
        Initialize the NormChecker with agents and a threshold for norm emergence.
        :param agents: List of Agent objects participating in the simulation.
        :param threshold: The percentage of agents (e.g., 90%) needed to adopt an action for norm emergence.
        """
        self.agents = agents
        self.num_agents = num_agents
        self.threshold = threshold
        self.norm_changed = False
        self.less_action = None

    def check_norm_emergence(self):
        """
        Check if a norm has emerged by counting the actions of all agents.
        :return: Boolean indicating if norm has emerged (True if >= threshold of agents chooses the same action).
        """
        count_A = sum(1 for agent in self.agents if agent.last_action == 'A')
        count_B = len(self.agents) - count_A

        if count_A >= len(self.agents) * self.threshold or count_B >= len(self.agents) * self.threshold:
            return True
        else:
            return False

    def determine_less_norm_action(self):
        """
        Determine the action with fewer adopters among agents.
        :return: 'A' or 'B', representing the action with fewer adopters.
        """
        count_A = sum(1 for agent in self.agents if agent.last_action == 'A')
        count_B = len(self.agents) - count_A

        return 'B' if count_A > count_B else 'A'

    def calculate_norm_abandonment(self, new_norm_action):
        """
        Calculate the percentage of agents who abandoned the previous norm (e.g., 'A')
        and now choose the new dominant action (e.g., 'B').
        """
        num_adopting_new_norm = sum(
            1 for agent in self.agents if agent.last_action == new_norm_action
        )
        abandonment_percentage = (num_adopting_new_norm / self.num_agents) * 100
        return abandonment_percentage

