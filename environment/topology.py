import random


class Topology:
    def _form_pairs_with_toroidal_topology(self, episode, grid_height, grid_width):
        """Form pairs of agents based on toroidal grid topology."""
        pairs = []

        if episode % 4 == 0:
            # Right neighbor
            for row in range(grid_height):
                for col in range(0, grid_width, 2):
                    agent1_id = row * grid_width + col
                    agent2_id = row * grid_width + (col + 1) % grid_width
                    pairs.append((agent1_id, agent2_id))

        elif episode % 4 == 1:
            # Left neighbor
            for row in range(grid_height):
                for col in range(0, grid_width, 2):
                    agent1_id = row * grid_width + col
                    agent2_id = row * grid_width + (col - 1) % grid_width
                    pairs.append((agent1_id, agent2_id))

        elif episode % 4 == 2:
            # Below neighbor
            for col in range(grid_width):
                for row in range(0, grid_height, 2):
                    agent1_id = row * grid_width + col
                    agent2_id = ((row + 1) % grid_height) * grid_width + col
                    pairs.append((agent1_id, agent2_id))

        else:
            # Above neighbor
            for col in range(grid_width):
                for row in range(0, grid_height, 2):
                    agent1_id = row * grid_width + col
                    agent2_id = ((row - 1) % grid_height) * grid_width + col
                    pairs.append((agent1_id, agent2_id))

        return pairs

    def _form_pairs_with_scale_free_topology(self,num_agents,scale_free_graph):
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

    def _form_pairs_randomly(self,num_agents):
        """Pair agents randomly for each timestep, returning agent indices."""
        agent_indices = list(range(num_agents))
        random.shuffle(agent_indices)
        pairs = [(agent_indices[i], agent_indices[i + 1]) for i in range(0, len(agent_indices) - 1, 2)]
        return pairs
