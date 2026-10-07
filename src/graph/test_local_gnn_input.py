import os
import sys
import traci
import torch

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.environment.state import get_all_local_states
from src.graph.graph_builder import build_graph
from src.graph.graph_builder import create_adjacency_matrix
from src.graph.graph_dataset import create_node_features
from src.graph.gnn import TrafficGNN


SUMO_BINARY = (
    r"C:\Program Files (x86)\Eclipse\Sumo"
    r"\bin\sumo-gui.exe"
)

SUMO_CONFIG = (
    r"C:\Project\Adaptive-Traffic-Control"
    r"\sumo-rl\sumo_rl\nets\RESCO\grid4x4"
    r"\grid4x4.sumocfg"
)


traci.start([
    SUMO_BINARY,
    "-c",
    SUMO_CONFIG
])


# Build graph once
graph = build_graph()

nodes, adjacency = create_adjacency_matrix(
    graph
)

adjacency = torch.tensor(
    adjacency,
    dtype=torch.float32
)


for step in range(100):

    traci.simulationStep()

    local_states = get_all_local_states()

    feature_nodes, features = create_node_features(
        local_states
    )

    x = torch.tensor(
        features,
        dtype=torch.float32
    )

    # GNN
    gnn = TrafficGNN(
        input_features=6,
        hidden_features=64,
        output_features=32
    )

    output = gnn(
        x,
        adjacency
    )

    if step % 20 == 0:

        print("\n==============================")
        print("STEP:", step)
        print("==============================")

        print(
            "Input shape:",
            x.shape
        )

        print(
            "Adjacency shape:",
            adjacency.shape
        )

        print(
            "GNN output shape:",
            output.shape
        )

        print(
            "A0 embedding:",
            output[0]
        )


traci.close()