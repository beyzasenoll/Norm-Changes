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
            self.graph = None  # No explicit graph for random and toroidal topologies

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
            # 1st-degree neighbors
            neighbors.update([
                ((row + 1) % self.grid_height, col),  # Below
                ((row - 1) % self.grid_height, col),  # Above
                (row, (col + 1) % self.grid_width),  # Right
                (row, (col - 1) % self.grid_width)  # Left
            ])

        if degree >= 2:
            # 2nd-degree neighbors
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

    def _get_neighbors_small_world(self, node, degree):
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

    def form_pairs(self, allowed_circles = [1, 2, 3]):
        """
        Forms agent pairs based on the selected topology.

        :param allowed_circles: Allowed degrees for pairing (e.g., [2, 3] pairs only with 2nd and 3rd-degree neighbors).
        :return: List of paired agent IDs.
        """
        if self.topology_type == "random":
            return self.form_pairs_randomly()
        elif self.topology_type == "toroidal":
            return self.form_pairs_with_toroidal_topology()
        elif self.topology_type == "small_world":
            return self.form_pairs_with_small_world_topology(allowed_circles)
        elif self.topology_type == "scale_free":
            return self.form_pairs_with_scale_free_topology()

    import random

    def form_pairs_with_toroidal_topology(self):
        """Ensures each agent is paired exactly once using toroidal topology."""
        pairs = []
        unpaired_agents = list(range(self.num_agents))  # List of unpaired agents

        while unpaired_agents:  # Continue until all agents are paired
            agent_id = unpaired_agents.pop(0)  # Pick the first unpaired agent
            row, col = divmod(agent_id, self.grid_width)

            # Get neighbors for different degrees
            first_degree_neighbors = self._get_neighbors_toroidal(row, col, 1)
            second_degree_neighbors = self._get_neighbors_toroidal(row, col, 2) - first_degree_neighbors
            third_degree_neighbors = self._get_neighbors_toroidal(row, col,
                                                                  3) - first_degree_neighbors - second_degree_neighbors

            # Convert neighbors to agent IDs and filter only unpaired ones
            first_degree_ids = [r * self.grid_width + c for r, c in first_degree_neighbors if
                                (r * self.grid_width + c) in unpaired_agents]
            second_degree_ids = [r * self.grid_width + c for r, c in second_degree_neighbors if
                                 (r * self.grid_width + c) in unpaired_agents]
            third_degree_ids = [r * self.grid_width + c for r, c in third_degree_neighbors if
                                (r * self.grid_width + c) in unpaired_agents]

            # If no available neighbors, move the agent to the end of the queue (retry later)
            if not first_degree_ids and not second_degree_ids and not third_degree_ids:
                unpaired_agents.append(agent_id)
                continue  # Skip and retry in the next iteration

            # Circle probabilities
            circle_probabilities = [(1, 0.5), (2, 0.3), (3, 0.2)]
            available_circles = []

            if first_degree_ids:
                available_circles.append(1)
            if second_degree_ids:
                available_circles.append(2)
            if third_degree_ids:
                available_circles.append(3)

            # Normalize probabilities based on available circles
            normalized_probs = [p for circle, p in circle_probabilities if circle in available_circles]
            normalized_probs = [p / sum(normalized_probs) for p in normalized_probs]

            # Randomly select a circle based on adjusted probabilities
            selected_circle = random.choices(available_circles, weights=normalized_probs, k=1)[0]

            # Choose a random unpaired neighbor from the selected circle
            if selected_circle == 1:
                chosen_neighbor = random.choice(first_degree_ids)
            elif selected_circle == 2:
                chosen_neighbor = random.choice(second_degree_ids)
            else:
                chosen_neighbor = random.choice(third_degree_ids)

            # Remove the chosen neighbor from the unpaired list
            unpaired_agents.remove(chosen_neighbor)

            # Store the pair
            pairs.append((agent_id, chosen_neighbor))

        return pairs

    def form_pairs_with_scale_free_topology(self, num_agents, scale_free_graph):
        edges = list(scale_free_graph.edges)

        random.shuffle(edges)

        paired_agents = set()
        pairs = []

        for edge in edges:
            agent1_id, agent2_id = edge
            if agent1_id not in paired_agents and agent2_id not in paired_agents:
                pairs.append((agent1_id, agent2_id))
                paired_agents.add(agent1_id)
                paired_agents.add(agent2_id)

            if len(paired_agents) >= num_agents:
                break

        return pairs

    def form_pairs_randomly(self, num_agents):
        """Pair agents randomly for each timestep, returning agent indices."""
        agent_indices = list(range(num_agents))
        random.shuffle(agent_indices)
        pairs = [(agent_indices[i], agent_indices[i + 1]) for i in range(0, len(agent_indices) - 1, 2)]
        return pairs

    def form_pairs_with_small_world_topology(self, allowed_circles):
        """Form pairs using Small-World topology."""
        pairs = []
        paired_agents = set()

        for node in range(self.num_agents):
            if node in paired_agents:
                continue

            selected_neighbors = set()
            for degree in allowed_circles:
                selected_neighbors.update(self._get_neighbors_small_world(node, degree))

            selected_neighbors = [n for n in selected_neighbors if n not in paired_agents]
            if not selected_neighbors:
                continue

            chosen_neighbor = random.choice(selected_neighbors)
            pairs.append((node, chosen_neighbor))
            paired_agents.add(node)
            paired_agents.add(chosen_neighbor)

        return pairs

    # Todo : Scale free pairs is empty . It is not working correctly.When you use correct this method.
    def form_pairs_with_scale_free_topology(self):
        """Ensures each agent is paired exactly once using Scale-Free topology."""
        edges = list(self.graph.edges)
        random.shuffle(edges)  # Randomize edges to prevent bias

        pairs = []
        unpaired_agents = set(range(self.num_agents))  # Track all unpaired agents

        while len(unpaired_agents) > 1:
            agent1 = unpaired_agents.pop()  # Get an unpaired agent

            # Get only unpaired neighbors
            neighbors = [neighbor for neighbor in self.graph.neighbors(agent1) if neighbor in unpaired_agents]

            # If no available neighbors, retry the agent later instead of skipping it
            if not neighbors:
                unpaired_agents.add(agent1)
                if len(unpaired_agents) == 1:
                    break  # Stop if only one agent remains unpaired
                continue  # Retry with the next agent

            # Randomly select a neighbor from available unpaired agents
            agent2 = random.choice(neighbors)
            unpaired_agents.remove(agent2)  # Mark agent as paired

            pairs.append((agent1, agent2))  # Store the valid pair

        return pairs

    def form_pairs_randomly(self):
        """Pair agents randomly."""
        agent_indices = list(range(self.num_agents))
        random.shuffle(agent_indices)
        return [(agent_indices[i], agent_indices[i + 1]) for i in range(0, len(agent_indices) - 1, 2)]


# **Testing the Implementation**
if __name__ == "__main__":
    num_agents = 40
    topology_types = ["random", "toroidal", "small_world", "scale_free"]

    for topo in topology_types:
        print(f"\n📌 Pairing using {topo} topology:")
        topology = Topology(num_agents, topology_type=topo)
        pairs = topology.form_pairs([1, 2])
        print(pairs)




