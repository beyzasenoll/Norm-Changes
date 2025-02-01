import random


class Topology:
    def _get_neighbors(self, row, col, grid_height, grid_width, degree):
        """Get neighbors of a specific degree (1st, 2nd, 3rd) in a toroidal grid."""
        neighbors = set()

        if degree >= 1:
            # 1st-degree neighbors (direct neighbors)
            neighbors.update([
                ((row + 1) % grid_height, col),  # Below
                ((row - 1) % grid_height, col),  # Above
                (row, (col + 1) % grid_width),  # Right
                (row, (col - 1) % grid_width)  # Left
            ])

        if degree >= 2:
            # 2nd-degree neighbors (neighbors of 1st-degree neighbors)
            for r, c in neighbors.copy():
                neighbors.update([
                    ((r + 1) % grid_height, c),
                    ((r - 1) % grid_height, c),
                    (r, (c + 1) % grid_width),
                    (r, (c - 1) % grid_width)
                ])

        if degree >= 3:
            # 3rd-degree neighbors (neighbors of 2nd-degree neighbors)
            for r, c in neighbors.copy():
                neighbors.update([
                    ((r + 1) % grid_height, c),
                    ((r - 1) % grid_height, c),
                    (r, (c + 1) % grid_width),
                    (r, (c - 1) % grid_width)
                ])

        return neighbors

    def form_pairs_with_toroidal_topology(self, num_agents, grid_height, grid_width):
        """Form pairs with toroidal topology considering 1st, 2nd, and 3rd-degree neighbors with different
        probabilities."""
        pairs = []
        paired_agents = set()

        for agent_id in range(num_agents):
            if agent_id in paired_agents:
                continue

            row, col = divmod(agent_id, grid_width)

            # Get neighbors of different degrees
            first_degree_neighbors = self._get_neighbors(row, col, grid_height, grid_width, 1)
            second_degree_neighbors = self._get_neighbors(row, col, grid_height, grid_width, 2) - first_degree_neighbors
            third_degree_neighbors = self._get_neighbors(row, col, grid_height, grid_width,
                                                         3) - first_degree_neighbors - second_degree_neighbors

            # Convert neighbors to agent IDs
            first_degree_ids = [r * grid_width + c for r, c in first_degree_neighbors if
                                (r * grid_width + c) not in paired_agents]
            second_degree_ids = [r * grid_width + c for r, c in second_degree_neighbors if
                                 (r * grid_width + c) not in paired_agents]
            third_degree_ids = [r * grid_width + c for r, c in third_degree_neighbors if
                                (r * grid_width + c) not in paired_agents]

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

            # Choose a random agent from the selected circle
            if selected_circle == 1:
                chosen_neighbor = random.choice(first_degree_ids)
            elif selected_circle == 2:
                chosen_neighbor = random.choice(second_degree_ids)
            else:
                chosen_neighbor = random.choice(third_degree_ids)

            pairs.append((agent_id, chosen_neighbor))
            paired_agents.add(agent_id)
            paired_agents.add(chosen_neighbor)

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
