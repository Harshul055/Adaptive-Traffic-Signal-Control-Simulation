import csv
import os
import sys
from pathlib import Path

import torch
import traci

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.graph_builder import build_graph, create_adjacency_matrix
from src.graph.graph_dataset import create_node_features
from src.graph.gnn import TrafficGNN
from src.agents.actor import Actor
from src.environment.state import get_traffic_state, get_all_local_states
from src.prediction.traffic_predictor import OnlineTrafficPredictor
from src.environment.action import apply_action
from src.emergency.detector import get_emergency_vehicles, get_vip_vehicles
from src.emergency.priority import give_emergency_priority, give_vip_priority


MAX_STEPS = 1000
NUM_AGENTS = 16
ACTION_SIZE = 4
STATE_SIZE = 36
MIN_PHASE_DURATION = 10
SUMO_DELAY_MS = 0
SUMO_SEED = 42

SUMO_HOME = Path(
    os.environ.get(
        "SUMO_HOME",
        r"C:\Program Files (x86)\Eclipse\Sumo"
    )
)

SUMO_BINARY = SUMO_HOME / "bin" / ("sumo-gui.exe" if os.environ.get("SUMO_GUI") == "1" else "sumo.exe")

if not SUMO_BINARY.exists():
    SUMO_BINARY = SUMO_HOME / "bin" / "sumo.exe"

SUMO_CONFIG = (
    PROJECT_ROOT
    / "sumo-rl"
    / "sumo_rl"
    / "nets"
    / "RESCO"
    / "grid4x4"
    / "grid4x4.sumocfg"
)

GRID_DIR = SUMO_CONFIG.parent

GNN_MODEL = (
    PROJECT_ROOT
    / "models"
    / "gnn"
    / "traffic_gnn.pth"
)

MAPPO_DIR = (
    PROJECT_ROOT
    / "models"
    / "mappo"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "tables"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

FIXED_RAW = RESULT_DIR / "paper_fixed_time_metrics.csv"
AI_RAW = RESULT_DIR / "paper_gnn_mappo_emergency_vip_metrics.csv"
FINAL_TABLE = TABLE_DIR / "paper_results.csv"

TLS_FALLBACK = [
    "A0", "A1", "A2", "A3",
    "B0", "B1", "B2", "B3",
    "C0", "C1", "C2", "C3",
    "D0", "D1", "D2", "D3"
]


def check_files():
    required = [
        SUMO_BINARY,
        SUMO_CONFIG,
        GNN_MODEL,
        GRID_DIR / "grid4x4_1.rou.xml",
        PROJECT_ROOT / "scripts" / "merge_traffic_scenario.py"
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    for index in range(1, NUM_AGENTS + 1):
        actor = MAPPO_DIR / f"actor_{index:02d}.pth"
        if not actor.exists():
            raise FileNotFoundError(f"MAPPO actor not found: {actor}")


def prepare_matched_scenario():
    print("\nPreparing matched Emergency/VIP scenario...")

    normal_file = GRID_DIR / "grid4x4_1.rou.xml"
    output_file = GRID_DIR / "test_scenario.rou.xml"

    import random
    import xml.etree.ElementTree as ET

    random.seed(42)

    tree = ET.parse(normal_file)
    root = tree.getroot()

    # Remove an older generated special scenario if it exists.
    for vehicle in list(root.findall("vehicle")):
        vehicle_id = vehicle.get("id", "")
        if vehicle_id.startswith("emergency_") or vehicle_id.startswith("vip_"):
            root.remove(vehicle)

    special = []

    for i in range(1, 6):
        special.append(
            {
                "id": f"emergency_{i}",
                "type": "emergency",
                "depart": random.randint(100, 900)
            }
        )

    for i in range(1, 6):
        special.append(
            {
                "id": f"vip_{i}",
                "type": "vip",
                "depart": random.randint(100, 900)
            }
        )

    route = "left0A0 A0A1 A1A2 A2A3 A3top0"

    for vehicle in special:
        element = ET.Element(
            "vehicle",
            {
                "id": vehicle["id"],
                "type": vehicle["type"],
                "depart": str(vehicle["depart"])
            }
        )

        ET.SubElement(
            element,
            "route",
            {"edges": route}
        )

        root.append(element)

    vehicles = root.findall("vehicle")
    vehicles.sort(
        key=lambda item: float(item.get("depart", "0"))
    )

    for vehicle in list(root.findall("vehicle")):
        root.remove(vehicle)

    for vehicle in vehicles:
        root.append(vehicle)

    tree.write(
        output_file,
        encoding="UTF-8",
        xml_declaration=True
    )

    print("Matched scenario created:")
    print(output_file)
    print("Emergency vehicles: 5")
    print("VIP vehicles      : 5")
    print("All special departures are before step 1000.")


def load_models():
    graph = build_graph()
    graph_tls, adjacency = create_adjacency_matrix(graph)

    adjacency = torch.tensor(
        adjacency,
        dtype=torch.float32
    )

    gnn = TrafficGNN(
        input_features=6,
        hidden_features=64,
        output_features=32
    )

    gnn.load_state_dict(
        torch.load(
            GNN_MODEL,
            map_location="cpu"
        )
    )

    gnn.eval()

    lstm_model = (
        PROJECT_ROOT
        / "models"
        / "lstm"
        / "traffic_lstm.pth"
    )

    if not lstm_model.exists():
        raise FileNotFoundError(
            f"LSTM model not found: {lstm_model}"
        )

    predictor = OnlineTrafficPredictor(
        lstm_model,
        sequence_length=10
    )

    actors = []

    for index in range(1, NUM_AGENTS + 1):
        actor = Actor(
            input_size=STATE_SIZE,
            hidden_size=64,
            action_size=ACTION_SIZE
        )

        actor.load_state_dict(
            torch.load(
                MAPPO_DIR / f"actor_{index:02d}.pth",
                map_location="cpu"
            )
        )

        actor.eval()
        actors.append(actor)

    tls_ids = (
        graph_tls
        if len(graph_tls) == NUM_AGENTS
        else TLS_FALLBACK
    )

    return gnn, adjacency, actors, tls_ids, predictor


def collect_step_metrics(step_length):
    state = get_traffic_state()

    co2_mg_s = 0.0

    for vehicle_id in traci.vehicle.getIDList():
        co2_mg_s += traci.vehicle.getCO2Emission(vehicle_id)

    co2_g_step = (
        co2_mg_s * step_length / 1000.0
    )

    return {
        "vehicle_count": state["vehicle_count"],
        "waiting_time": state["total_waiting_time"],
        "average_speed": state["average_speed"],
        "queue_length": state["queue_length"],
        "co2_g_step": co2_g_step
    }


def write_raw_csv(path, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "step",
                "vehicle_count",
                "waiting_time",
                "average_speed",
                "queue_length",
                "co2_g_step",
                "completed_vehicles"
            ]
        )

        writer.writerows(rows)


def summarize(rows, step_length, detected=0, granted=0, response_times=None):
    if not rows:
        raise RuntimeError("No evaluation rows were collected.")

    response_times = response_times or []

    count = len(rows)

    average_vehicle_count = (
        sum(row[1] for row in rows) / count
    )

    average_waiting = (
        sum(row[2] for row in rows) / count
    )

    average_speed = (
        sum(row[3] for row in rows) / count
    )

    average_queue = (
        sum(row[4] for row in rows) / count
    )

    total_co2 = sum(
        row[5] for row in rows
    )

    completed = rows[-1][6]

    simulation_seconds = count * step_length

    throughput = (
        completed / simulation_seconds * 3600.0
        if simulation_seconds > 0
        else 0.0
    )

    priority_rate = (
        granted / detected * 100.0
        if detected
        else 0.0
    )

    average_priority_response = (
        sum(response_times) / len(response_times)
        if response_times
        else None
    )

    return {
        "vehicle_count": average_vehicle_count,
        "waiting_time": average_waiting,
        "average_speed": average_speed,
        "queue_length": average_queue,
        "co2_g": total_co2,
        "completed_vehicles": completed,
        "throughput_vph": throughput,
        "special_detected": detected,
        "priority_granted": granted,
        "priority_response_rate": priority_rate,
        "average_priority_response_s": average_priority_response
    }


def run_fixed_time():
    print("\n" + "=" * 70)
    print("MATCHED FIXED-TIME EVALUATION")
    print("=" * 70)

    traci.start(
        [
            str(SUMO_BINARY),
            "-c",
            str(SUMO_CONFIG),
            "--delay", str(SUMO_DELAY_MS),
            "--seed", str(SUMO_SEED)
        ]
    )

    rows = []
    completed_ids = set()
    step_length = float(traci.simulation.getDeltaT())

    try:
        for step in range(MAX_STEPS):
            traci.simulationStep()

            metrics = collect_step_metrics(
                step_length
            )

            completed_ids.update(
                traci.simulation.getArrivedIDList()
            )

            rows.append(
                [
                    step,
                    metrics["vehicle_count"],
                    metrics["waiting_time"],
                    metrics["average_speed"],
                    metrics["queue_length"],
                    metrics["co2_g_step"],
                    len(completed_ids)
                ]
            )

    finally:
        traci.close()

    write_raw_csv(FIXED_RAW, rows)

    return summarize(
        rows,
        step_length
    )


def apply_priority(
    detected_times,
    granted_times,
    current_time
):
    emergency_ids = set(
        get_emergency_vehicles()
    )

    vip_ids = set(
        get_vip_vehicles()
    )

    special_ids = (
        emergency_ids | vip_ids
    )

    for vehicle_id in special_ids:
        if vehicle_id not in detected_times:
            detected_times[vehicle_id] = current_time

    # Emergency always has priority.
    for vehicle_id in emergency_ids:
        result = give_emergency_priority(vehicle_id)

        if result is not None and result.get("priority_granted", False):
            if vehicle_id not in granted_times:
                granted_times[vehicle_id] = current_time

    # VIP is considered only when no emergency vehicle is active.
    if not emergency_ids:
        for vehicle_id in vip_ids:
            result = give_vip_priority(
                vehicle_id
            )

            if (
                result is not None
                and result.get("priority_granted", False)
                and vehicle_id not in granted_times
            ):
                granted_times[vehicle_id] = current_time


def run_gnn_mappo():
    print("\n" + "=" * 70)
    print("MATCHED GNN + MAPPO + EMERGENCY/VIP EVALUATION")
    print("=" * 70)

    gnn, adjacency, actors, tls_ids, predictor = load_models()

    traci.start([
        str(SUMO_BINARY), "-c", str(SUMO_CONFIG),
        "--delay", str(SUMO_DELAY_MS),
        "--seed", str(SUMO_SEED),
    ])

    rows = []
    completed_ids = set()
    detected_times = {}
    granted_times = {}

    step_length = float(traci.simulation.getDeltaT())

    try:
        for step in range(MAX_STEPS):
            current_time = float(
                traci.simulation.getTime()
            )

            local_states = get_all_local_states()

            _, features = create_node_features(
                local_states
            )

            x = torch.tensor(
                features,
                dtype=torch.float32
            )

            with torch.no_grad():
                gnn_output = gnn(
                    x,
                    adjacency
                )

                lstm_prediction = predictor.predict_normalized(
                    get_traffic_state()
                )

                lstm_features = (
                    lstm_prediction
                    .unsqueeze(0)
                    .repeat(NUM_AGENTS, 1)
                )

                policy_state = torch.cat(
                    [gnn_output, lstm_features],
                    dim=1
                )

                actions = []

                for index in range(NUM_AGENTS):
                    probabilities = actors[index](
                        policy_state[index].unsqueeze(0)
                    )

                    action = torch.argmax(
                        probabilities,
                        dim=-1
                    ).item()

                    actions.append(action)

            for index in range(NUM_AGENTS):
                apply_action(
                    tls_ids[index],
                    actions[index],
                    min_phase_duration=MIN_PHASE_DURATION
                )

            # Detect and apply priority before the next SUMO step.
            apply_priority(
                detected_times,
                granted_times,
                current_time
            )

            traci.simulationStep()

            metrics = collect_step_metrics(
                step_length
            )

            completed_ids.update(
                traci.simulation.getArrivedIDList()
            )

            rows.append(
                [
                    step,
                    metrics["vehicle_count"],
                    metrics["waiting_time"],
                    metrics["average_speed"],
                    metrics["queue_length"],
                    metrics["co2_g_step"],
                    len(completed_ids)
                ]
            )

    finally:
        traci.close()

    response_times = []

    for vehicle_id, granted_time in granted_times.items():
        if vehicle_id in detected_times:
            response_times.append(
                granted_time - detected_times[vehicle_id]
            )

    write_raw_csv(AI_RAW, rows)

    return summarize(
        rows,
        step_length,
        detected=len(detected_times),
        granted=len(granted_times),
        response_times=response_times
    )


def improvement_lower(baseline, adaptive):
    if baseline == 0:
        return 0.0

    return (
        (baseline - adaptive)
        / baseline
        * 100.0
    )


def improvement_higher(baseline, adaptive):
    if baseline == 0:
        return 0.0

    return (
        (adaptive - baseline)
        / baseline
        * 100.0
    )


def save_final_table(fixed, adaptive):
    metrics = [
        (
            "Delay (s)",
            fixed["waiting_time"],
            adaptive["waiting_time"],
            improvement_lower
        ),
        (
            "Queue (vehicles)",
            fixed["queue_length"],
            adaptive["queue_length"],
            improvement_lower
        ),
        (
            "Average Speed (m/s)",
            fixed["average_speed"],
            adaptive["average_speed"],
            improvement_higher
        ),
        (
            "Vehicle Count",
            fixed["vehicle_count"],
            adaptive["vehicle_count"],
            improvement_lower
        ),
        (
            "CO2 (g)",
            fixed["co2_g"],
            adaptive["co2_g"],
            improvement_lower
        ),
        (
            "Throughput (veh/h)",
            fixed["throughput_vph"],
            adaptive["throughput_vph"],
            improvement_higher
        )
    ]

    with FINAL_TABLE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "Controller",
                "Delay (s)",
                "Queue (vehicles)",
                "Average Speed (m/s)",
                "Vehicle Count",
                "CO2 (g)",
                "Throughput (veh/h)",
                "Emergency/VIP Priority Response (%)"
            ]
        )

        writer.writerow(
            [
                "Fixed-Time",
                f"{fixed['waiting_time']:.2f}",
                f"{fixed['queue_length']:.2f}",
                f"{fixed['average_speed']:.2f}",
                f"{fixed['vehicle_count']:.2f}",
                f"{fixed['co2_g']:.2f}",
                f"{fixed['throughput_vph']:.2f}",
                "No priority"
            ]
        )

        writer.writerow(
            [
                "GNN + MAPPO + Emergency/VIP",
                f"{adaptive['waiting_time']:.2f}",
                f"{adaptive['queue_length']:.2f}",
                f"{adaptive['average_speed']:.2f}",
                f"{adaptive['vehicle_count']:.2f}",
                f"{adaptive['co2_g']:.2f}",
                f"{adaptive['throughput_vph']:.2f}",
                f"{adaptive['priority_response_rate']:.2f}%"
            ]
        )

        writer.writerow([])
        writer.writerow(
            [
                "Improvement (%)",
                f"{improvement_lower(fixed['waiting_time'], adaptive['waiting_time']):.2f}",
                f"{improvement_lower(fixed['queue_length'], adaptive['queue_length']):.2f}",
                f"{improvement_higher(fixed['average_speed'], adaptive['average_speed']):.2f}",
                f"{improvement_lower(fixed['vehicle_count'], adaptive['vehicle_count']):.2f}",
                f"{improvement_lower(fixed['co2_g'], adaptive['co2_g']):.2f}",
                f"{improvement_higher(fixed['throughput_vph'], adaptive['throughput_vph']):.2f}",
                ""
            ]
        )

        writer.writerow([])
        writer.writerow(
            [
                "Completed vehicles",
                fixed["completed_vehicles"],
                adaptive["completed_vehicles"]
            ]
        )

        writer.writerow(
            [
                "Special vehicles detected",
                "N/A",
                adaptive["special_detected"]
            ]
        )

        writer.writerow(
            [
                "Priority grants",
                "N/A",
                adaptive["priority_granted"]
            ]
        )

        response = adaptive["average_priority_response_s"]

        writer.writerow(
            [
                "Average priority response time (s)",
                "N/A",
                "" if response is None else f"{response:.2f}"
            ]
        )

    print("\n" + "=" * 70)
    print("PAPER METRICS")
    print("=" * 70)

    print(
        f"Fixed-Time CO2                 : {fixed['co2_g']:.2f} g"
    )
    print(
        f"GNN + MAPPO + Emergency/VIP CO2: {adaptive['co2_g']:.2f} g"
    )
    print(
        f"Fixed-Time throughput          : {fixed['throughput_vph']:.2f} veh/h"
    )
    print(
        f"GNN + MAPPO + Emergency/VIP    : {adaptive['throughput_vph']:.2f} veh/h"
    )
    print(
        f"Priority response rate         : {adaptive['priority_response_rate']:.2f}%"
    )

    if adaptive["average_priority_response_s"] is not None:
        print(
            "Average priority response time : "
            f"{adaptive['average_priority_response_s']:.2f} s"
        )

    print("\nFinal table:")
    print(FINAL_TABLE)


def main():
    check_files()
    prepare_matched_scenario()

    fixed = run_fixed_time()
    adaptive = run_gnn_mappo()

    save_final_table(
        fixed,
        adaptive
    )


if __name__ == "__main__":
    main()
