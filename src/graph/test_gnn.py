import os
import sys
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
from src.graph.graph_builder import create_adjacency_matrix
from src.graph.graph_dataset import create_node_features
from src.graph.gnn import TrafficGNN


print("\n======================================")
print("GNN TEST")
print("======================================")


# 1. Build traffic graph
graph = build_graph()

nodes, adjacency = create_adjacency_matrix(
    graph
)

print("\nNumber of intersections:", len(nodes))


# 2. Create traffic features
local_states = {
    node: {
        "vehicle_count": 30,
        "queue_length": 10,
        "waiting_time": 100,
        "average_speed": 8,
        "current_phase": 0,
        "phase_duration": 30,
    }
    for node in nodes
}

_, features = create_node_features(local_states)


# 3. Convert to tensors
x = torch.tensor(
    features,
    dtype=torch.float32
)

adjacency = torch.tensor(
    adjacency,
    dtype=torch.float32
)


print("Input feature shape:", x.shape)
print("Adjacency shape:", adjacency.shape)


# 4. Create GNN
model = TrafficGNN(
    input_features=6,
    hidden_features=64,
    output_features=32
)


# 5. Forward pass
output = model(
    x,
    adjacency
)


print("\n======================================")
print("GNN OUTPUT")
print("======================================")

print("Output shape:", output.shape)

print("\nFirst intersection representation:")
print(output[0])


print("\n======================================")
print("GNN TEST COMPLETED")
print("======================================")