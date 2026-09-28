import os
import sys
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.graph.graph_builder import build_graph


def create_adjacency_matrix(graph):
    nodes = sorted(graph.keys())

    node_index = {
        node: i
        for i, node in enumerate(nodes)
    }

    n = len(nodes)

    adjacency = np.zeros(
        (n, n),
        dtype=np.float32
    )

    for node in nodes:

        for neighbor in graph[node]:

            if neighbor in node_index:

                i = node_index[node]
                j = node_index[neighbor]

                adjacency[i][j] = 1

    return nodes, adjacency


def create_node_features(local_states):
    """
    Convert local intersection states
    into a node-feature matrix.
    """

    nodes = sorted(local_states.keys())

    features = []

    for node in nodes:

        state = local_states[node]

        node_features = [

            state["vehicle_count"],

            state["queue_length"],

            state["waiting_time"],

            state["average_speed"],

            state["current_phase"],

            state["phase_duration"]

        ]

        features.append(node_features)

    return nodes, np.array(
        features,
        dtype=np.float32
    )


if __name__ == "__main__":

    print("\n======================================")
    print("GNN DATASET")
    print("======================================")

    # Build graph
    graph = build_graph()

    # Create adjacency matrix
    nodes, adjacency = create_adjacency_matrix(
        graph
    )

    print("\nNumber of nodes:", len(nodes))

    print("\nNodes:")

    for node in nodes:
        print("-", node)

    print("\nAdjacency Matrix:")
    print(adjacency)

    # Create example node features
    features = create_node_features(
        vehicle_count=30,
        waiting_time=100,
        average_speed=8,
        queue_length=10,
        number_of_nodes=len(nodes)
    )

    # Convert to PyTorch tensors
    x = torch.tensor(
        features,
        dtype=torch.float32
    )

    adjacency_tensor = torch.tensor(
        adjacency,
        dtype=torch.float32
    )

    print("\n======================================")
    print("TENSOR INFORMATION")
    print("======================================")

    print("Node feature shape:", x.shape)

    print(
        "Adjacency shape:",
        adjacency_tensor.shape
    )

    print("\nGNN dataset preparation completed!")