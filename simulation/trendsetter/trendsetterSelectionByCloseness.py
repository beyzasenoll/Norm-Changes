import random
import logging
import networkx as nx


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrendsetterSelectorByCloseness:
    def __init__(self, simulation):
        self.simulation = simulation


    def get_agents_sorted_by_closeness(self):
        if self.simulation.topology_type not in ["scale_free", "small_world"] or not self.simulation.topology.graph:
            logger.warning("This method is only applicable for scale-free or small-world topologies.")
            return []

        num_trendsetters = max(1, int(self.simulation.num_agents * self.simulation.trendsetter_percent / 100))
        closeness_centrality = nx.closeness_centrality(self.simulation.topology.graph)
        sorted_agents = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)

        trendsetters = [agent_id for agent_id, _ in sorted_agents[:num_trendsetters]]
        logger.info(f"Selected trendsetters: {trendsetters}")
        return trendsetters
