import random
import logging

from networkx import shortest_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrendsetterSelectorByDegree:
    def __init__(self, simulation):
        self.simulation = simulation

    def select_by_degree_toroidal(self, use_random=False):
        num_trendsetters = max(1, int(self.simulation.num_agents * self.simulation.trendsetter_percent / 100))
        selected_trendsetters = []

        if not use_random and self.simulation.topology_type == 'toroidal':
            current_agent_id = random.sample(range(self.simulation.num_agents), 1)[0]
            for _ in range(num_trendsetters):
                selected_trendsetters.append(current_agent_id)
                current_agent_id = (current_agent_id + 1) % self.simulation.num_agents
        else:
            return random.sample(range(self.simulation.num_agents), num_trendsetters)

        return selected_trendsetters
    def select_by_degree(self, distance_type="close"):
        num_trendsetters = max(1, int(self.simulation.num_agents * self.simulation.trendsetter_percent / 100))
        agents_sorted_by_degree = self.get_agents_sorted_by_degree()

        if not agents_sorted_by_degree:
            logger.warning("No agents found by degree. Falling back to random selection.")
            return self.select_by_degree_toroidal(use_random=True)

        high_degree_agents = agents_sorted_by_degree[:len(agents_sorted_by_degree) // 2]
        if not high_degree_agents:
            logger.warning("High-degree agent list is empty. Falling back to random selection.")
            return self.select_by_degree_toroidal(use_random=True)

        trendsetters = []
        distance_list= {}
        first_trendsetter = high_degree_agents[0][0]
        trendsetters.append(first_trendsetter)
        high_degree_agents.remove(high_degree_agents[0])
        for agent in high_degree_agents:
            distance_list[agent[0]] = len(shortest_path(self.simulation.topology.graph,first_trendsetter, agent[0]))

        sorted_distance_list = sorted(distance_list.items(), key=lambda x: x[1], reverse=False)
        if distance_type == "close":
            for agent_id, _ in sorted_distance_list[:num_trendsetters - 1]:  # -1 çünkü ilk zaten eklendi
                trendsetters.append(agent_id)
        else:
            for agent_id, _ in sorted_distance_list[::-1][:num_trendsetters - 1]:
                trendsetters.append(agent_id)
        return trendsetters
    def get_agents_sorted_by_degree(self):
        if self.simulation.topology_type not in ["scale_free", "small_world"] or not self.simulation.topology.graph:
            logger.warning("This method is only applicable for scale-free or small-world topologies.")
            return []

        degree_list = list(self.simulation.topology.graph.degree())
        sorted_agents = sorted(degree_list, key=lambda x: x[1], reverse=True)
        return sorted_agents