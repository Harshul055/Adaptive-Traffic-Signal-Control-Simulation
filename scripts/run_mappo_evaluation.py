import os
import sys
import csv

import torch
import traci


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

from src.graph.graph_builder import (
    build_graph,
    create_adjacency_matrix
)

from src.graph.graph_dataset import (
    create_node_features
)

from src.graph.gnn import TrafficGNN

from src.agents.actor import Actor

from src.environment.state import (
    get_all_local_states
)

from src.environment.action import (
    apply_action
)


# ==================================================
# SUMO
# ==================================================

SUMO_HOME = os.environ.get(
    "SUMO_HOME",
    r"C:\Program Files (x86)\Eclipse\Sumo"
)

SUMO_BINARY = os.path.join(
    SUMO_HOME,
    "bin",
    "sumo-gui.exe"
)

SUMO_CONFIG = os.path.join(
    PROJECT_ROOT,
    "sumo-rl", "sumo_rl", "nets", "RESCO",
    "grid4x4", "grid4x4.sumocfg"
)


# ==================================================
# SETTINGS
# ==================================================

MAX_STEPS = 1000
NUM_AGENTS = 16
ACTION_SIZE = 4
STATE_SIZE = 32

MIN_PHASE_DURATION = 10


# ==================================================
# MODEL PATHS
# ==================================================

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "mappo"
)

GNN_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "gnn",
    "traffic_gnn.pth"
)


# ==================================================
# OUTPUT
# ==================================================

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "mappo_final_results.csv"
)


# ==================================================
# TRAFFIC LIGHT ORDER
# ==================================================

TLS_IDS = [
    "A0", "A1", "A2", "A3",
    "B0", "B1", "B2", "B3",
    "C0", "C1", "C2", "C3",
    "D0", "D1", "D2", "D3"
]


# ==================================================
# LOAD GNN
# ==================================================

def load_gnn():

    graph = build_graph()

    traffic_lights, adjacency = (
        create_adjacency_matrix(
            graph
        )
    )

    adjacency = torch.tensor(
        adjacency,
        dtype=torch.float32
    )

    gnn = TrafficGNN(
        input_features=6,
        hidden_features=64,
        output_features=32
    )

    if not os.path.exists(GNN_MODEL):

        raise FileNotFoundError(
            f"GNN model not found:\n{GNN_MODEL}"
        )

    gnn.load_state_dict(
        torch.load(
            GNN_MODEL,
            map_location="cpu"
        )
    )

    gnn.eval()

    print(
        "GNN model loaded successfully."
    )

    return gnn, adjacency, traffic_lights


# ==================================================
# LOAD ACTORS
# ==================================================

def load_actors():

    actors = []

    print()
    print("Loading MAPPO actors...")

    for i in range(1, NUM_AGENTS + 1):

        actor_file = os.path.join(
            MODEL_DIR,
            f"actor_{i:02d}.pth"
        )

        if not os.path.exists(
            actor_file
        ):

            raise FileNotFoundError(
                f"Actor not found:\n{actor_file}"
            )

        actor = Actor(
            input_size=STATE_SIZE,
            hidden_size=64,
            action_size=ACTION_SIZE
        )

        actor.load_state_dict(
            torch.load(
                actor_file,
                map_location="cpu"
            )
        )

        actor.eval()

        actors.append(
            actor
        )

        print(
            f"Actor {i:02d} loaded."
        )

    return actors


# ==================================================
# RUN EVALUATION
# ==================================================

def evaluate():

    print()
    print("=" * 60)
    print("GNN + MAPPO FINAL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------

    gnn, adjacency, graph_tls = load_gnn()

    actors = load_actors()

    # Use graph order if it contains all 16 signals
    if len(graph_tls) == NUM_AGENTS:

        tls_ids = graph_tls

    else:

        tls_ids = TLS_IDS

    print()
    print(
        "Traffic lights:",
        len(tls_ids)
    )

    # --------------------------------------------------
    # START SUMO
    # --------------------------------------------------

    print()
    print("Starting SUMO-GUI...")

    traci.start([
        SUMO_BINARY,
        "-c",
        SUMO_CONFIG
    ])

    print(
        "SUMO started successfully."
    )

    rows = []

    try:

        for step in range(
            MAX_STEPS
        ):

            # --------------------------------------------------
            # GET LOCAL STATES
            # --------------------------------------------------
            # --------------------------------------------------

            local_states = (
                get_all_local_states()
            )

            # --------------------------------------------------
            # CREATE GNN FEATURES
            # --------------------------------------------------

            _, features = (
                create_node_features(
                    local_states
                )
            )

            x = torch.tensor(
                features,
                dtype=torch.float32
            )

            # --------------------------------------------------
            # GNN
            # --------------------------------------------------

            with torch.no_grad():

                gnn_output = gnn(
                    x,
                    adjacency
                )

            # --------------------------------------------------
            # MAPPO ACTIONS
            # --------------------------------------------------

            actions = []

            with torch.no_grad():

                for agent_index in range(
                    NUM_AGENTS
                ):

                    state = (
                        gnn_output[
                            agent_index
                        ]
                    )

                    probabilities = actors[
                        agent_index
                    ](state.unsqueeze(0))

                    action = torch.argmax(
                        probabilities,
                        dim=-1
                    ).item()

                    actions.append(
                        action
                    )

            # --------------------------------------------------
            # APPLY ACTIONS
            # --------------------------------------------------

            for agent_index in range(
                NUM_AGENTS
            ):

                apply_action(
                    tls_ids[agent_index],
                    actions[agent_index],
                    min_phase_duration=(
                        MIN_PHASE_DURATION
                    )
                )

            # --------------------------------------------------
            # ADVANCE AFTER ACTION
            # --------------------------------------------------

            traci.simulationStep()

            # --------------------------------------------------
            # COLLECT METRICS
            # --------------------------------------------------

            vehicle_ids = (
                traci.vehicle.getIDList()
            )

            vehicle_count = len(
                vehicle_ids
            )

            total_waiting = 0.0
            total_speed = 0.0
            queue_length = 0

            for vehicle_id in vehicle_ids:

                speed = traci.vehicle.getSpeed(
                    vehicle_id
                )

                waiting = (
                    traci.vehicle.getAccumulatedWaitingTime(
                        vehicle_id
                    )
                )

                total_speed += speed
                total_waiting += waiting

                if speed < 0.1:

                    queue_length += 1

            if vehicle_count > 0:

                average_speed = (
                    total_speed
                    / vehicle_count
                )

            else:

                average_speed = 0.0

            # --------------------------------------------------
            # REWARD
            # --------------------------------------------------

            reward = (
                average_speed
                - 0.01 * total_waiting
                - queue_length
            )

            rows.append([
                step,
                vehicle_count,
                total_waiting,
                average_speed,
                queue_length,
                reward
            ])

            if step % 100 == 0:

                print(
                    f"Step {step:04d} | "
                    f"Vehicles: {vehicle_count} | "
                    f"Waiting: {total_waiting:.2f} | "
                    f"Speed: {average_speed:.2f} | "
                    f"Queue: {queue_length}"
                )

    finally:

        traci.close()

    # ==================================================
    # SAVE
    # ==================================================

    with open(
        RESULT_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "step",
            "vehicle_count",
            "waiting_time",
            "average_speed",
            "queue_length",
            "reward"
        ])

        writer.writerows(
            rows
        )

    # ==================================================
    # CALCULATE FINAL METRICS
    # ==================================================

    total_rows = len(rows)

    avg_vehicles = sum(
        row[1] for row in rows
    ) / total_rows

    avg_waiting = sum(
        row[2] for row in rows
    ) / total_rows

    avg_speed = sum(
        row[3] for row in rows
    ) / total_rows

    avg_queue = sum(
        row[4] for row in rows
    ) / total_rows

    total_reward = sum(
        row[5] for row in rows
    )

    # ==================================================
    # RESULTS
    # ==================================================

    print()
    print("=" * 60)
    print("GNN + MAPPO RESULTS")
    print("=" * 60)

    print(
        f"Average vehicles: "
        f"{avg_vehicles:.2f}"
    )

    print(
        f"Average waiting time: "
        f"{avg_waiting:.2f}"
    )

    print(
        f"Average speed: "
        f"{avg_speed:.2f} m/s"
    )

    print(
        f"Average queue length: "
        f"{avg_queue:.2f}"
    )

    print(
        f"Total reward: "
        f"{total_reward:.2f}"
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULT_FILE
    )

    print()
    print("=" * 60)
    print("MAPPO EVALUATION COMPLETE")
    print("=" * 60)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    evaluate()