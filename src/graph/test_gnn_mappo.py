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
from src.agents.mappo import MAPPO


# -----------------------------
# SUMO configuration
# -----------------------------

SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
SUMO_BINARY = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe")

SUMO_CONFIG = os.path.join(
    PROJECT_ROOT, "sumo-rl", "sumo_rl", "nets", "RESCO",
    "grid4x4", "grid4x4.sumocfg"
)


# -----------------------------
# Start SUMO
# -----------------------------

traci.start([
    SUMO_BINARY,
    "-c",
    SUMO_CONFIG
])


# -----------------------------
# Build graph
# -----------------------------

graph = build_graph()

nodes, adjacency = create_adjacency_matrix(graph)

adjacency = torch.tensor(
    adjacency,
    dtype=torch.float32
)


# -----------------------------
# Create ONE GNN
# -----------------------------

gnn = TrafficGNN(
    input_features=6,
    hidden_features=64,
    output_features=32
)


# -----------------------------
# Create 16 MAPPO agents
# -----------------------------

agents = []

for i in range(16):

    agent = MAPPO(
        state_size=32,
        action_size=4
    )

    agents.append(agent)


print()
print("Number of MAPPO agents:", len(agents))
print("Number of intersections:", len(nodes))


# -----------------------------
# Simulation
# -----------------------------

for step in range(100):

    traci.simulationStep()

    # Get local state of all 16 intersections
    local_states = get_all_local_states()

    # Convert states to features
    feature_nodes, features = create_node_features(
        local_states
    )

    x = torch.tensor(
        features,
        dtype=torch.float32
    )

    # -------------------------
    # GNN
    # -------------------------

    gnn_output = gnn(
        x,
        adjacency
    )

    # -------------------------
    # MAPPO
    # -------------------------

    actions = []

    for i in range(16):

        node_state = gnn_output[i]

        global_state = gnn_output.flatten()
        action, log_probability, value = agents[i].select_action(
            node_state,
            global_state
        )

        actions.append(action)

    # -------------------------
    # Print results
    # -------------------------

    if step % 20 == 0:

        print()
        print("STEP:", step)

        print(
            "Local feature shape:",
            x.shape
        )

        print(
            "GNN output shape:",
            gnn_output.shape
        )

        print(
            "Number of actions:",
            len(actions)
        )

        print(
            "Actions:",
            actions
        )

        print(
            "A0 action:",
            actions[0]
        )


# -----------------------------
# Close SUMO
# -----------------------------

traci.close()

print()
print("GNN → MAPPO TEST COMPLETED")