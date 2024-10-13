import random
import matplotlib.pyplot as plt
from Agent import Agent  # Agent sınıfını import ediyoruz


class Simulation:
    def __init__(self, num_agents, num_steps, temperature=1.0):
        self.num_agents = num_agents
        self.agents = [Agent(i, temperature) for i in range(num_agents)]
        self.num_steps = num_steps
        self.scores_history = [[] for _ in range(num_agents)]

        self.action_combinations = {
            'AA': [],
            'BB': [],
            'AB': [],
            'BA': []
        }

        self.total_A_count = 0
        self.total_B_count = 0

    def run(self):
        for step in range(self.num_steps):
            agent_indices = list(range(self.num_agents))
            random.shuffle(agent_indices)

            count_AA, count_BB, count_AB, count_BA = 0, 0, 0, 0
            for i in range(0, self.num_agents, 2):
                agent1 = self.agents[agent_indices[i]]
                agent2 = self.agents[agent_indices[i + 1]]

                action1 = agent1.boltzmann()
                action2 = agent2.boltzmann()

                if action1 == 'A':
                    self.total_A_count += 1
                elif action1 == 'B':
                    self.total_B_count += 1

                if action2 == 'A':
                    self.total_A_count += 1
                elif action2 == 'B':
                    self.total_B_count += 1

                if action1 == 'A' and action2 == 'A':
                    count_AA += 1
                elif action1 == 'B' and action2 == 'B':
                    count_BB += 1
                elif action1 == 'A' and action2 == 'B':
                    count_AB += 1
                elif action1 == 'B' and action2 == 'A':
                    count_BA += 1

                if action1 == action2:
                    agent1.update_score(action1, 1)
                    agent2.update_score(action2, 1)
                else:
                    agent1.update_score(action1, -1)
                    agent2.update_score(action2, -1)

                self.scores_history[agent1.agent_id].append(agent1.total_score)
                self.scores_history[agent2.agent_id].append(agent2.total_score)

            # Her iterasyondaki aksiyon kombinasyonlarını kaydet
            self.action_combinations['AA'].append(count_AA)
            self.action_combinations['BB'].append(count_BB)
            self.action_combinations['AB'].append(count_AB)
            self.action_combinations['BA'].append(count_BA)

    def plot_action_combinations(self):
        # Aksiyon kombinasyonları için grafiği çiz
        plt.plot(self.action_combinations['AA'], label='Both A')
        plt.plot(self.action_combinations['BB'], label='Both B')
        plt.plot(self.action_combinations['AB'], label='A vs B')
        plt.plot(self.action_combinations['BA'], label='B vs A')

        plt.xlabel('Timestamp')
        plt.ylabel('Frequency')
        plt.title('Action Combinations Over Time')
        plt.legend()
        plt.show()

    def print_action_counts(self):
        print(f"Total A selections: {self.total_A_count}")
        print(f"Total B selections: {self.total_B_count}")
