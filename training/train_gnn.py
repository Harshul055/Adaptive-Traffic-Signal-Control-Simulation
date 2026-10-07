import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import traci

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.graph.gnn import TrafficGNN
from src.graph.graph_builder import build_graph, create_adjacency_matrix
from src.graph.graph_dataset import create_node_features
from src.environment.state import get_all_local_states


INPUT_FEATURES = 6
HIDDEN_FEATURES = 64
OUTPUT_FEATURES = 32

LEARNING_RATE = 0.001
EPOCHS = 50
MAX_STEPS = 500
SUMO_DELAY_MS = 0
SUMO_SEED = 42

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "gnn")
MODEL_FILE = os.path.join(MODEL_DIR, "traffic_gnn.pth")

SUMO_HOME = os.environ.get(
    "SUMO_HOME",
    r"C:\Program Files (x86)\Eclipse\Sumo",
)
SUMO_BINARY = os.path.join(SUMO_HOME, "bin", "sumo.exe")
SUMO_CONFIG = os.path.join(
    PROJECT_ROOT,
    "sumo-rl", "sumo_rl", "nets", "RESCO",
    "grid4x4", "grid4x4.sumocfg",
)


class GraphStateAutoencoder(nn.Module):
    """Train the GNN encoder to reconstruct real local traffic states."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(OUTPUT_FEATURES, INPUT_FEATURES)

    def forward(self, features, adjacency):
        embedding = self.encoder(features, adjacency)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction


def collect_training_snapshots(adjacency):
    """Collect real local intersection states from SUMO."""

    if not os.path.exists(SUMO_BINARY):
        raise FileNotFoundError(
            f"SUMO executable not found: {SUMO_BINARY}. "
            "Set SUMO_HOME to your SUMO installation."
        )
    if not os.path.exists(SUMO_CONFIG):
        raise FileNotFoundError(f"SUMO config not found: {SUMO_CONFIG}")

    snapshots = []

    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--delay", str(SUMO_DELAY_MS), "--seed", str(SUMO_SEED)])
    try:
        for _ in range(MAX_STEPS):
            local_states = get_all_local_states()
            _, features = create_node_features(local_states)
            snapshots.append(
                torch.tensor(features, dtype=torch.float32)
            )
            traci.simulationStep()

            if traci.simulation.getMinExpectedNumber() <= 0:
                break
    finally:
        traci.close()

    if not snapshots:
        raise RuntimeError("No GNN training snapshots were collected.")

    return torch.stack(snapshots)


def train_gnn():
    print("=" * 60)
    print("GNN TRAINING FROM REAL SUMO TRAFFIC STATES")
    print("=" * 60)

    graph = build_graph()
    nodes, adjacency_matrix = create_adjacency_matrix(graph)

    if len(nodes) == 0:
        raise RuntimeError("Traffic graph contains no signalized intersections.")

    adjacency = torch.tensor(adjacency_matrix, dtype=torch.float32)

    print(f"Intersections: {len(nodes)}")
    print("Collecting local traffic snapshots from SUMO...")

    snapshots = collect_training_snapshots(adjacency)
    print(f"Snapshots collected: {len(snapshots)}")
    print(f"Snapshot shape: {tuple(snapshots.shape)}")

    # Feature scales balance waiting-time/queue magnitudes against speed
    # without changing the raw feature representation consumed at inference.
    feature_scale = snapshots.reshape(-1, INPUT_FEATURES).std(dim=0)
    feature_scale = torch.where(
        feature_scale > 1e-6,
        feature_scale,
        torch.ones_like(feature_scale),
    )

    encoder = TrafficGNN(
        input_features=INPUT_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        output_features=OUTPUT_FEATURES,
    )
    model = GraphStateAutoencoder(encoder)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting GNN autoencoder training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        # Every snapshot is a graph with the same 16-node topology.
        for features in snapshots:
            optimizer.zero_grad()

            _, reconstruction = model(features, adjacency)
            error = (reconstruction - features) / feature_scale
            loss = torch.mean(error ** 2)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(snapshots)

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:03d}/{EPOCHS} "
                f"| Reconstruction loss: {epoch_loss:.6f}"
            )

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.encoder.state_dict(), MODEL_FILE)

    print()
    print("GNN training completed.")
    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    train_gnn()
