import numpy as np


class Agent:
    def __init__(self, agent_id, temperature=1.0):
        self.agent_id = agent_id
        self.actions = ['A', 'B']
        self.scores = {'A': 0, 'B': 0}
        self.total_score = 0
        self.temperature = temperature

    def boltzmann(self):
        """Aksiyon seçimini Boltzmann dağılımına göre yap."""
        exp_scores = np.exp(np.array(list(self.scores.values())) / self.temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        return np.random.choice(self.actions, p=probabilities)

    def update_score(self, action, result):
        """Seçilen aksiyonun puanını güncelle."""
        self.scores[action] += result
        self.total_score += result
