import os
import pandas as pd

from src.evaluation.metrics import calculate_metrics


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results"
)

FIXED_TIME_FILE = os.path.join(
    RESULTS_DIR,
    "fixed_time_results.csv"
)

MAPPO_FILE = os.path.join(
    RESULTS_DIR,
    "mappo_results.csv"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_results(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Results file not found:\n{file_path}"
        )

    return pd.read_csv(file_path)


# --------------------------------------------------
# EVALUATE ONE METHOD
# --------------------------------------------------

def evaluate_method(name, data):

    metrics = calculate_metrics(data)

    print()
    print("=" * 50)
    print(f"{name} EVALUATION")
    print("=" * 50)

    print(
        f"Average vehicles: "
        f"{metrics['average_vehicle_count']:.2f}"
    )

    print(
        f"Average waiting time: "
        f"{metrics['average_waiting_time']:.2f}"
    )

    print(
        f"Average speed: "
        f"{metrics['average_speed']:.2f} m/s"
    )

    print(
        f"Average queue length: "
        f"{metrics['average_queue_length']:.2f}"
    )

    if metrics["total_reward"] is not None:
        print(
            f"Total reward: "
            f"{metrics['total_reward']:.2f}"
        )

    print("=" * 50)

    return metrics


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print()
    print("Loading evaluation results...")
    print()

    fixed_time_data = load_results(
        FIXED_TIME_FILE
    )

    mappo_data = load_results(
        MAPPO_FILE
    )

    print(
        f"Fixed-Time rows: "
        f"{len(fixed_time_data)}"
    )

    print(
        f"MAPPO rows: "
        f"{len(mappo_data)}"
    )

    fixed_metrics = evaluate_method(
        "FIXED-TIME",
        fixed_time_data
    )

    mappo_metrics = evaluate_method(
        "GNN + MAPPO",
        mappo_data
    )

    print()
    print("=" * 60)
    print("FIXED-TIME vs GNN + MAPPO")
    print("=" * 60)

    print(
        f"{'Metric':<25}"
        f"{'Fixed-Time':>15}"
        f"{'GNN + MAPPO':>15}"
    )

    print("-" * 60)

    print(
        f"{'Average Vehicles':<25}"
        f"{fixed_metrics['average_vehicle_count']:>15.2f}"
        f"{mappo_metrics['average_vehicle_count']:>15.2f}"
    )

    print(
        f"{'Average Waiting Time':<25}"
        f"{fixed_metrics['average_waiting_time']:>15.2f}"
        f"{mappo_metrics['average_waiting_time']:>15.2f}"
    )

    print(
        f"{'Average Speed':<25}"
        f"{fixed_metrics['average_speed']:>15.2f}"
        f"{mappo_metrics['average_speed']:>15.2f}"
    )

    print(
        f"{'Average Queue Length':<25}"
        f"{fixed_metrics['average_queue_length']:>15.2f}"
        f"{mappo_metrics['average_queue_length']:>15.2f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()