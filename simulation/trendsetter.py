import random
import networkx as nx
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrendsetterSelector:
    def __init__(self, simulation):
        self.simulation = simulation

    def select_trendsetters(self, use_random=False):
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

    def select_by_network(self, circle_type="close"):
        num_trendsetters = max(1, int(self.simulation.num_agents * self.simulation.trendsetter_percent / 100))
        agents_sorted_by_degree = self.get_agents_sorted_by_degree()

        if not agents_sorted_by_degree:
            logger.warning("No agents found by degree. Falling back to random selection.")
            return self.select_trendsetters(use_random=True)

        high_degree_agents = agents_sorted_by_degree[:len(agents_sorted_by_degree) // 2]

        if not high_degree_agents:
            logger.warning("High-degree agent list is empty. Falling back to random selection.")
            return self.select_trendsetters(use_random=True)

        trendsetters = []
        first_trendsetter = random.choice(high_degree_agents)[0]
        logger.info(f"Selected first trendsetter: {first_trendsetter}")
        trendsetters.append(first_trendsetter)

        neighbors = list(nx.all_neighbors(self.simulation.topology.graph, first_trendsetter))
        logger.info(f"First-degree neighbors: {neighbors}")

        if neighbors:
            random_neighbor = random.choice(neighbors)
            second_degree_neighbors = list(nx.all_neighbors(self.simulation.topology.graph, random_neighbor))
            logger.info(f"Second-degree neighbors: {second_degree_neighbors}")

            if circle_type == "close":
                combined_neighbors = list(set(neighbors + second_degree_neighbors))
                logger.info(f"Combined close-circle neighbors: {combined_neighbors}")
            elif circle_type == "far":
                third_degree_neighbors = list(nx.all_neighbors(self.simulation.topology.graph, random_neighbor))
                combined_neighbors = third_degree_neighbors

            remaining_count = num_trendsetters - 1
            selected_neighbors = random.sample(combined_neighbors, min(remaining_count, len(combined_neighbors)))
            trendsetters.extend(selected_neighbors)

        return trendsetters

    def get_agents_sorted_by_degree(self):
        if self.simulation.topology_type not in ["scale_free", "small_world"] or not self.simulation.topology.graph:
            logger.warning("This method is only applicable for scale-free or small-world topologies.")
            return []

        degree_list = list(self.simulation.topology.graph.degree())
        sorted_agents = sorted(degree_list, key=lambda x: x[1], reverse=True)
        return sorted_agents
