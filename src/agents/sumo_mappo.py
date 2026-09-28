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
from src.graph.graph_dataset import create_adjacency_matrix
from src.graph.gnn import TrafficGNN
from src.agents.mappo import MAPPO


SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"

SUMO_CONFIG = (
    r"C:\Project\Adaptive-Traffic-Control"
    r"\sumo-rl\sumo_rl\nets\RESCO\grid4x4"
    r"\grid4x4.sumocfg"
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
    input_features=4,
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
    # Collect traffic state
    # ------------------------------

    features = []

    for tls_id in traffic_lights:

        vehicle_ids = (
            traci.vehicle.getIDList()
        )

        vehicle_count = len(vehicle_ids)

        waiting_time = 0
        total_speed = 0
        queue = 0

        for vehicle_id in vehicle_ids:

            waiting_time += (
                traci.vehicle
                .getAccumulatedWaitingTime(
                    vehicle_id
                )
            )

            speed = (
                traci.vehicle
                .getSpeed(vehicle_id)
            )

            total_speed += speed

            if speed < 0.1:
                queue += 1

        if vehicle_count > 0:
            average_speed = (
                total_speed / vehicle_count
            )
        else:
            average_speed = 0

        features.append([
            vehicle_count,
            waiting_time,
            average_speed,
            queue
        ])


    # ------------------------------
    # Convert to tensor
    # ------------------------------

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
                state
            )
        )

        actions.append(action)


    # ------------------------------
    # Apply actions to SUMO
    # ------------------------------

    if step % 10 == 0:

        print(
            "\nStep:",
            step
        )

        for i, tls_id in enumerate(
            traffic_lights
        ):

            phase_count = len(
                traci.trafficlight
                .getAllProgramLogics(
                    tls_id
                )[0]
                .getPhases()
            )

            if phase_count > 0:

                phase = (
                    actions[i] %
                    phase_count
                )

                traci.trafficlight.setPhase(
                    tls_id,
                    phase
                )

                print(
                    tls_id,
                    "→ Phase:",
                    phase
                )


# --------------------------------------
# 7. Close SUMO
# --------------------------------------

traci.close()

print("\n======================================")
print("SUMO + GNN + MAPPO TEST COMPLETED")
print("======================================")