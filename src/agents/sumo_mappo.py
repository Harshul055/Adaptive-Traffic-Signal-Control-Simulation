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

from src.graph.graph_builder import build_graph
from src.graph.graph_builder import create_adjacency_matrix
from src.graph.gnn import TrafficGNN
from src.graph.graph_dataset import create_node_features
from src.agents.mappo import MAPPO
from src.environment.state import get_all_local_states


SUMO_HOME = os.environ.get(
    "SUMO_HOME",
    r"C:\Program Files (x86)\Eclipse\Sumo"
)

SUMO_CONFIG = os.path.join(
    PROJECT_ROOT,
    "sumo-rl", "sumo_rl", "nets", "RESCO",
    "grid4x4", "grid4x4.sumocfg"
)

SUMO_BINARY = os.path.join(
    SUMO_HOME,
    "bin",
    "sumo-gui.exe"
)


print("\n======================================")
print("SUMO + GNN + MAPPO")
print("======================================")


# --------------------------------------
# 1. Start SUMO
# --------------------------------------

traci.start([
    SUMO_BINARY,
    "-c",
    SUMO_CONFIG
])

print("SUMO started successfully!")


# --------------------------------------
# 2. Get traffic lights
# --------------------------------------

traffic_lights = (
    traci.trafficlight.getIDList()
)

print(
    "Traffic lights:",
    len(traffic_lights)
)


# --------------------------------------
# 3. Build graph
# --------------------------------------

graph = build_graph()

nodes, adjacency = create_adjacency_matrix(
    graph
)

adjacency = torch.tensor(
    adjacency,
    dtype=torch.float32
)


# --------------------------------------
# 4. Create GNN
# --------------------------------------

gnn = TrafficGNN(
    input_features=6,
    hidden_features=64,
    output_features=32
)


# --------------------------------------
# 5. Create MAPPO agents
# --------------------------------------

agents = []

for _ in range(len(traffic_lights)):

    agents.append(
        MAPPO(
            state_size=32,
            action_size=4
        )
    )


print(
    "MAPPO agents:",
    len(agents)
)


# --------------------------------------
# 6. Run simulation
# --------------------------------------

for step in range(1000):

    traci.simulationStep()


    # ------------------------------
    # Collect local traffic states
    # ------------------------------

    local_states = get_all_local_states()

    _, features = create_node_features(
        local_states
    )

    x = torch.tensor(
        features,
        dtype=torch.float32
    )

    # ------------------------------
    # GNN
    # ------------------------------

    gnn_output = gnn(
        x,
        adjacency
    )

    # Centralized MAPPO critic state
    global_state = gnn_output.flatten()


    # ------------------------------
    # MAPPO actions
    # ------------------------------

    actions = []

    for i in range(
        len(traffic_lights)
    ):

        state = gnn_output[i]

        action, _, _ = (
            agents[i].select_action(
                state,
                global_state
            )
        )

        actions.append(action)


    # ------------------------------
    # Apply actions to SUMO every step
    # ------------------------------

    for i, tls_id in enumerate(traffic_lights):
        phase_count = len(
            traci.trafficlight.getAllProgramLogics(tls_id)[0].getPhases()
        )
        if phase_count > 0:
            phase = actions[i] % phase_count
            traci.trafficlight.setPhase(tls_id, phase)

    if step % 10 == 0:
        print("\nStep:", step)
        for i, tls_id in enumerate(traffic_lights):
            print(tls_id, "→ Phase:", actions[i])


# --------------------------------------
# 7. Close SUMO
# --------------------------------------

traci.close()

print("\n======================================")
print("SUMO + GNN + MAPPO TEST COMPLETED")
print("======================================")