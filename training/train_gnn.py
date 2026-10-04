import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==================================================
# IMPORTS
# ==================================================

from src.graph.gnn import TrafficGNN
from src.graph.graph_builder import (
    build_graph,
    create_adjacency_matrix
)


# ==================================================
# SETTINGS
# ==================================================

INPUT_FEATURES = 6
HIDDEN_FEATURES = 64
OUTPUT_FEATURES = 32

LEARNING_RATE = 0.001
EPOCHS = 100

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "gnn"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "traffic_gnn.pth"
)


# ==================================================
# CREATE TRAINING DATA
# ==================================================

def create_training_data(
    graph,
    adjacency
):

    nodes = sorted(graph.keys())

    num_nodes = len(nodes)

    # --------------------------------------------------
    # Initial traffic features
    # --------------------------------------------------
    #
    # These are placeholder traffic-state features
    # used to train the GNN structure.
    #
    # Later these can be replaced by real collected
    # traffic states from SUMO.
    # --------------------------------------------------

    features = torch.zeros(
        num_nodes,
        INPUT_FEATURES,
        dtype=torch.float32
    )

    for i in range(num_nodes):

        features[i] = torch.tensor([
            1.0,   # vehicle count
            0.0,   # queue length
            0.0,   # waiting time
            10.0,  # average speed
            0.0,   # current phase
            30.0   # phase duration
        ])

    return features


# ==================================================
# TRAIN GNN
# ==================================================

def train_gnn():

    print()
    print("=" * 50)
    print("GNN TRAINING")
    print("=" * 50)

    # --------------------------------------------------
    # Build graph
    # --------------------------------------------------

    print()
    print("Building traffic graph...")

    graph = build_graph()

    nodes, adjacency_matrix = (
        create_adjacency_matrix(graph)
    )

    print(
        f"Number of nodes: {len(nodes)}"
    )

    print(
        f"Adjacency shape: "
        f"({len(adjacency_matrix)}, "
        f"{len(adjacency_matrix[0])})"
    )

    # --------------------------------------------------
    # Convert adjacency to tensor
    # --------------------------------------------------

    adjacency = torch.tensor(
        adjacency_matrix,
        dtype=torch.float32
    )

    # --------------------------------------------------
    # Create input features
    # --------------------------------------------------

    features = create_training_data(
        graph,
        adjacency
    )

    print(
        f"Feature shape: {features.shape}"
    )

    # --------------------------------------------------
    # Create target
    # --------------------------------------------------

    # The target is the same feature representation
    # for this initial graph-embedding training.
    #
    # This verifies that the GNN can learn a stable
    # graph representation before connecting it to
    # the full traffic-learning pipeline.

    target = torch.zeros(
        len(nodes),
        OUTPUT_FEATURES,
        dtype=torch.float32
    )

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = TrafficGNN(
        input_features=INPUT_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        output_features=OUTPUT_FEATURES
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    print()
    print("Starting training...")

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        optimizer.zero_grad()

        output = model(
            features,
            adjacency
        )

        loss = criterion(
            output,
            target
        )

        loss.backward()

        optimizer.step()

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch {epoch:03d}/{EPOCHS} "
                f"| Loss: {loss.item():.6f}"
            )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        MODEL_FILE
    )

    print()
    print("GNN training completed.")

    print()
    print(
        "Model saved to:"
    )

    print(
        MODEL_FILE
    )

    print("=" * 50)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    train_gnn()