import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.graph.graph_builder import build_graph
from src.graph.graph_dataset import (
    create_adjacency_matrix
)
from src.graph.gnn import TrafficGNN


DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results",
    "traffic_data.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "gnn"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "traffic_gnn.pth"
)

FEATURE_COLUMNS = [
    "vehicle_count",
    "waiting_time",
    "average_speed",
    "queue_length"
]

EPOCHS = 100
LEARNING_RATE = 0.001


print("\n======================================")
print("GNN TRAINING")
print("======================================")


# --------------------------------------
# 1. Load traffic data
# --------------------------------------

data = pd.read_csv(DATA_FILE)

data[FEATURE_COLUMNS] = data[
    FEATURE_COLUMNS
].apply(
    pd.to_numeric,
    errors="coerce"
)

data = data.dropna(
    subset=FEATURE_COLUMNS
).reset_index(drop=True)

print("Traffic records:", len(data))


# --------------------------------------
# 2. Normalize traffic features
# --------------------------------------

features = data[
    FEATURE_COLUMNS
].values.astype(
    np.float32
)

feature_min = features.min(axis=0)
feature_max = features.max(axis=0)

features = (
    features - feature_min
) / (
    feature_max - feature_min + 1e-8
)


# --------------------------------------
# 3. Build graph
# --------------------------------------

graph = build_graph()

nodes, adjacency = create_adjacency_matrix(
    graph
)

number_of_nodes = len(nodes)

print(
    "Number of intersections:",
    number_of_nodes
)


# --------------------------------------
# 4. Create node features
# --------------------------------------

average_features = features.mean(
    axis=0
)

node_features = np.tile(
    average_features,
    (number_of_nodes, 1)
)


# --------------------------------------
# 5. Convert to tensors
# --------------------------------------

x = torch.tensor(
    node_features,
    dtype=torch.float32
)

adjacency = torch.tensor(
    adjacency,
    dtype=torch.float32
)

target = x.clone()


print(
    "Input shape:",
    x.shape
)

print(
    "Target shape:",
    target.shape
)


# --------------------------------------
# 6. Create GNN
# --------------------------------------

model = TrafficGNN(
    input_features=4,
    hidden_features=64,
    output_features=4
)


criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------
# 7. Train
# --------------------------------------

print("\nStarting GNN training...\n")


for epoch in range(EPOCHS):

    optimizer.zero_grad()

    output = model(
        x,
        adjacency
    )

    loss = criterion(
        output,
        target
    )

    loss.backward()

    optimizer.step()

    if (
        epoch == 0
        or (epoch + 1) % 10 == 0
    ):
        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {loss.item():.6f}"
        )


# --------------------------------------
# 8. Save model
# --------------------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_FILE
)


print("\n======================================")
print("GNN TRAINING COMPLETED")
print("======================================")

print(
    "Model saved to:"
)

print(MODEL_FILE)