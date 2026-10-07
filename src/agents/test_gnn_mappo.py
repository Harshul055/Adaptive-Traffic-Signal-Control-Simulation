import os
import sys
import torch
import traci

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
from src.environment.action import apply_action


print("\n======================================")
print("GNN → MAPPO → SUMO TEST")
print("======================================")


# --------------------------------------
# 1. Start SUMO
# --------------------------------------

SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
SUMO_BINARY = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe")

SUMO_CONFIG = os.path.join(
    PROJECT_ROOT, "sumo-rl", "sumo_rl", "nets", "RESCO",
    "grid4x4", "grid4x4.sumocfg"
)

traci.start([
    SUMO_BINARY,
    "-c",
    SUMO_CONFIG
])


# --------------------------------------
# 2. Build traffic graph
# --------------------------------------

graph = build_graph()

nodes, adjacency = create_adjacency_matrix(
    graph
)

number_of_nodes = len(nodes)

adjacency = torch.tensor(
    adjacency,
    dtype=torch.float32
)

print(
    "Number of intersections:",
    number_of_nodes
)

print(
    "Adjacency shape:",
    adjacency.shape
)


# --------------------------------------
# 3. Create GNN
# --------------------------------------

gnn = TrafficGNN(
    input_features=6,
    hidden_features=64,
    output_features=32
)


# --------------------------------------
# 4. Create MAPPO agents
# --------------------------------------

agents = []

for i in range(number_of_nodes):

    agent = MAPPO(
        state_size=32,
        action_size=4
    )

    agents.append(agent)


print(
    "MAPPO agents created:",
    len(agents)
)


# --------------------------------------
# 5. Simulation
# --------------------------------------

try:

    for step in range(100):

        # Advance SUMO
        traci.simulationStep()


        # ----------------------------------
        # Get real local traffic states
        # ----------------------------------

        local_states = get_all_local_states()


        # ----------------------------------
        # Convert states to features
        # ----------------------------------

        feature_nodes, features = (
            create_node_features(
                local_states
            )
        )


        x = torch.tensor(
            features,
            dtype=torch.float32
        )


        # ----------------------------------
        # GNN
        # ----------------------------------

        gnn_output = gnn(
            x,
            adjacency
        )


        # ----------------------------------
        # MAPPO
        # ----------------------------------

        actions = []

        for i in range(number_of_nodes):

            node_state = gnn_output[i]

            global_state = gnn_output.flatten()
            action, log_probability, value = (
                agents[i].select_action(
                    node_state,
                    global_state
                )
            )

            actions.append(action)


            # --------------------------------
            # Apply action to SUMO
            # --------------------------------

            tls_id = nodes[i]

            apply_action(
                tls_id,
                action,
                min_phase_duration=10
            )


        # ----------------------------------
        # Display
        # ----------------------------------

        if step % 20 == 0:

            print()
            print("STEP:", step)

            print(
                "GNN input shape:",
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


finally:

    # Close SUMO
    try:
        traci.close()
    except:
        pass


print()
print("======================================")
print("GNN → MAPPO → SUMO TEST COMPLETED")
print("======================================")