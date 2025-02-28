import random
import networkx as nx


class Topology:
    def __init__(self, num_agents, topology_type="toroidal", k=4, p=0.2):
        """
        Initializes the topology based on the selected type.

        :param num_agents: Number of agents (nodes) in the network.
        :param topology_type: Type of topology ("random", "toroidal", "small_world", "scale_free").
        :param k: Initial number of neighbors per node (used for Small-World topology).
        :param p: Probability of rewiring (used for Small-World topology).
        """
        self.num_agents = 40
        self.topology_type = topology_type
        self.k = k
        self.p = p
        self.grid_height = int(num_agents ** 0.5)  # Used for toroidal topology
        self.grid_width = int(num_agents ** 0.5)

        if topology_type == "small_world":
            self.graph = nx.watts_strogatz_graph(num_agents, k, p)
        elif topology_type == "scale_free":
            self.graph = nx.barabasi_albert_graph(num_agents, k)
        else:
            self.graph = None

    def calculate_beta_distance(self, row, col, grid_height, grid_width, beta):
        """Calculate observable neighbors based on Von Neumann topology and Beta scaling."""
        max_distance = max(grid_height, grid_width)
        observation_range = int(beta * max_distance)

        neighbors = set()
        for d in range(1, observation_range + 1):
            neighbors.update([
                ((row + d) % grid_height, col),
                ((row - d) % grid_height, col),
                (row, (col + d) % grid_width),
                (row, (col - d) % grid_width)
            ])
        return neighbors
    def _get_neighbors_toroidal(self, row, col, degree):
        """
        Get neighbors of a specific degree in a toroidal grid.

        :param row: Row index of the node.
        :param col: Column index of the node.
        :param degree: Degree of neighbors (1st, 2nd, or 3rd).
        :return: Set of neighboring coordinates.
        """
        neighbors = set()

        if degree >= 1:
            neighbors.update([
                ((row + 1) % self.grid_height, col),  # Below
                ((row - 1) % self.grid_height, col),  # Above
                (row, (col + 1) % self.grid_width),  # Right
                (row, (col - 1) % self.grid_width)  # Left
            ])

        if degree >= 2:
            for r, c in neighbors.copy():
                neighbors.update([
                    ((r + 1) % self.grid_height, c),
                    ((r - 1) % self.grid_height, c),
                    (r, (c + 1) % self.grid_width),
                    (r, (c - 1) % self.grid_width)
                ])

        if degree >= 3:
            # 3rd-degree neighbors
            for r, c in neighbors.copy():
                neighbors.update([
                    ((r + 1) % self.grid_height, c),
                    ((r - 1) % self.grid_height, c),
                    (r, (c + 1) % self.grid_width),
                    (r, (c - 1) % self.grid_width)
                ])

        return neighbors

    def _get_neighbors(self, node, degree):
        """
        Get neighbors of a specific degree in a Small-World network.

        :param node: Node index.
        :param degree: Degree of neighbors (1st, 2nd, or 3rd).
        :return: Set of neighboring nodes.
        """
        neighbors = set()
        if degree >= 1:
            first_degree_neighbors = set(self.graph.neighbors(node))
            neighbors.update(first_degree_neighbors)

        if degree >= 2:
            second_degree_neighbors = set()
            for neighbor in first_degree_neighbors:
                second_degree_neighbors.update(self.graph.neighbors(neighbor))
            second_degree_neighbors -= first_degree_neighbors
            second_degree_neighbors.discard(node)
            neighbors.update(second_degree_neighbors)

        if degree >= 3:
            third_degree_neighbors = set()
            for neighbor in second_degree_neighbors:
                third_degree_neighbors.update(self.graph.neighbors(neighbor))
            third_degree_neighbors -= second_degree_neighbors
            third_degree_neighbors -= first_degree_neighbors
            third_degree_neighbors.discard(node)
            neighbors.update(third_degree_neighbors)

        return neighbors


    def get_neighbors(self, agent_id, allowed_circles):
        """
        Get neighbors of an agent based on the topology type and allowed circles.

        :param agent_id: The ID of the agent.
        :param allowed_circles: List of allowed degrees (1st, 2nd, 3rd).
        :return: Set of neighboring agent IDs.
        """
        if self.topology_type == "random":
            # In random topology, all agents are potential neighbors
            return set(range(self.num_agents)) - {agent_id}
        elif self.topology_type == "toroidal":
            row, col = divmod(agent_id, self.grid_width)
            neighbors = set()
            for degree in allowed_circles:
                neighbors.update(self._get_neighbors_toroidal(row, col, degree))
            return {r * self.grid_width + c for r, c in neighbors}
        elif self.topology_type == "small_world":
            neighbors = set()
            for degree in allowed_circles:
                neighbors.update(self._get_neighbors(agent_id, degree))
            return neighbors
        elif self.topology_type == "scale_free":
            neighbors = set()
            for degree in allowed_circles:
                neighbors.update(self._get_neighbors(agent_id, degree))
            return neighbors

    def form_pairs(self, allowed_circles=None):
        """
        Forms agent pairs based on the selected topology.

        :param allowed_circles: Allowed degrees for pairing (e.g., [1, 2, 3] pairs with 1st, 2nd, and 3rd-degree neighbors).
        :return: List of paired agent IDs.
        """
        if allowed_circles is None:
            allowed_circles = [1, 2, 3]

        pairs = []
        used_agents = set()

        for agent_id in range(self.num_agents):
            if agent_id in used_agents:
                continue

            neighbors = self.get_neighbors(agent_id, allowed_circles)

            if neighbors:
                chosen_neighbor = random.choice(list(neighbors))
                pairs.append((agent_id, chosen_neighbor))
                used_agents.add(agent_id)

        return pairs


# **Testing the Implementation**
if __name__ == "__main__":
    num_agents = 40
    topology_types = ["random", "toroidal", "small_world", "scale_free"]

    for topo in topology_types:
        print(f"\n📌 Pairing using {topo} topology:")
        topology = Topology(num_agents, topology_type=topo)
        pairs = topology.form_pairs([1,2])
        print(pairs)