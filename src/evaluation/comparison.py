import os
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results"
)

FIXED_FILE = os.path.join(
    RESULT_DIR,
    "fixed_time_results.csv"
)

MAPPO_FILE = os.path.join(
    RESULT_DIR,
    "mappo_results.csv"
)

OUTPUT_FILE = os.path.join(
    RESULT_DIR,
    "comparison_results.csv"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

fixed = pd.read_csv(
    FIXED_FILE
)

mappo = pd.read_csv(
    MAPPO_FILE
)


# --------------------------------------------------
# FIXED-TIME METRICS
# --------------------------------------------------

fixed_vehicles = fixed["vehicle_count"].mean()

fixed_waiting = fixed[
    "waiting_time"
].mean()

fixed_speed = fixed[
    "average_speed"
].mean()

fixed_queue = fixed[
    "queue_length"
].mean()


# --------------------------------------------------
# MAPPO METRICS
# --------------------------------------------------

mappo_vehicles = mappo[
    "vehicle_count"
].mean()

mappo_waiting = mappo[
    "waiting_time"
].mean()

mappo_speed = mappo[
    "average_speed"
].mean()

mappo_queue = mappo[
    "queue_length"
].mean()


# --------------------------------------------------
# IMPROVEMENT CALCULATION
# --------------------------------------------------

def lower_is_better(
    fixed_value,
    mappo_value
):

    if fixed_value == 0:
        return 0

    return (
        (fixed_value - mappo_value)
        / fixed_value
    ) * 100


def higher_is_better(
    fixed_value,
    mappo_value
):

    if fixed_value == 0:
        return 0

    return (
        (mappo_value - fixed_value)
        / fixed_value
    ) * 100


waiting_improvement = lower_is_better(
    fixed_waiting,
    mappo_waiting
)

queue_improvement = lower_is_better(
    fixed_queue,
    mappo_queue
)

speed_improvement = higher_is_better(
    fixed_speed,
    mappo_speed
)

vehicle_improvement = lower_is_better(
    fixed_vehicles,
    mappo_vehicles
)


# --------------------------------------------------
# COMPARISON TABLE
# --------------------------------------------------

comparison = pd.DataFrame({

    "Metric": [
        "Average Vehicles",
        "Average Waiting Time",
        "Average Speed",
        "Average Queue Length"
    ],

    "Fixed-Time": [
        fixed_vehicles,
        fixed_waiting,
        fixed_speed,
        fixed_queue
    ],

    "GNN + MAPPO": [
        mappo_vehicles,
        mappo_waiting,
        mappo_speed,
        mappo_queue
    ],

    "Improvement (%)": [
        vehicle_improvement,
        waiting_improvement,
        speed_improvement,
        queue_improvement
    ]
})


# --------------------------------------------------
# SAVE RESULT
# --------------------------------------------------

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print()
print(
    "======================================"
)

print(
    "FIXED-TIME vs GNN + MAPPO"
)

print(
    "======================================"
)

print(
    comparison.to_string(
        index=False
    )
)

print()

print(
    "Comparison saved to:"
)

print(
    OUTPUT_FILE
)

print(
    "======================================"
)