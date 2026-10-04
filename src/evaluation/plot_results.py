import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

FIXED_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results",
    "fixed_time_results.csv"
)

MAPPO_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results",
    "mappo_results.csv"
)

LSTM_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results",
    "lstm_predictions.csv"
)

PLOT_DIR = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "plots"
)

os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AUTOMATIC TRAFFIC PERFORMANCE GRAPH")
print("=" * 70)

print("\nLoading datasets...")

if not os.path.exists(FIXED_FILE):
    print("ERROR: Fixed-Time CSV not found.")
    print(FIXED_FILE)
    exit()

if not os.path.exists(MAPPO_FILE):
    print("ERROR: MAPPO CSV not found.")
    print(MAPPO_FILE)
    exit()

if not os.path.exists(LSTM_FILE):
    print("ERROR: LSTM prediction CSV not found.")
    print(LSTM_FILE)
    exit()


fixed = pd.read_csv(FIXED_FILE)
mappo = pd.read_csv(MAPPO_FILE)
lstm = pd.read_csv(LSTM_FILE)


print(f"Fixed-Time rows : {len(fixed)}")
print(f"GNN + MAPPO rows: {len(mappo)}")
print(f"LSTM rows       : {len(lstm)}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

metrics = [
    "vehicle_count",
    "waiting_time",
    "average_speed",
    "queue_length"
]

for metric in metrics:

    if metric not in fixed.columns:
        print(f"ERROR: {metric} missing from Fixed-Time data.")
        exit()

    if metric not in mappo.columns:
        print(f"ERROR: {metric} missing from MAPPO data.")
        exit()

    if metric not in lstm.columns:
        print(f"ERROR: {metric} missing from LSTM data.")
        exit()


# ============================================================
# COMMON STEPS
# ============================================================

common_steps = min(
    len(fixed),
    len(mappo),
    len(lstm)
)

fixed = fixed.iloc[:common_steps].copy()
mappo = mappo.iloc[:common_steps].copy()
lstm = lstm.iloc[:common_steps].copy()

print(
    f"\nUsing {common_steps} common data points."
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_three_series(
    series1,
    series2,
    series3
):

    combined = pd.concat(
        [
            series1,
            series2,
            series3
        ]
    )

    minimum = combined.min()
    maximum = combined.max()

    if maximum == minimum:

        return (
            series1 * 0,
            series2 * 0,
            series3 * 0
        )

    normalized1 = (
        (series1 - minimum)
        / (maximum - minimum)
        * 100
    )

    normalized2 = (
        (series2 - minimum)
        / (maximum - minimum)
        * 100
    )

    normalized3 = (
        (series3 - minimum)
        / (maximum - minimum)
        * 100
    )

    return (
        normalized1,
        normalized2,
        normalized3
    )


# ============================================================
# CREATE GRAPH
# ============================================================

plt.figure(
    figsize=(18, 10)
)


# ============================================================
# METRIC STYLES
# ============================================================

metric_settings = {

    "vehicle_count": {
        "name": "Vehicle Count",
        "style": "-"
    },

    "waiting_time": {
        "name": "Waiting Time",
        "style": "--"
    },

    "average_speed": {
        "name": "Average Speed",
        "style": "-."
    },

    "queue_length": {
        "name": "Queue Length",
        "style": ":"
    }
}


# ============================================================
# PLOT ALL THREE DATASETS
# ============================================================

for metric in metrics:

    fixed_normalized, \
    mappo_normalized, \
    lstm_normalized = normalize_three_series(

        fixed[metric],
        mappo[metric],
        lstm[metric]
    )

    style = metric_settings[metric]["style"]
    name = metric_settings[metric]["name"]


    # Fixed-Time
    plt.plot(
        fixed["step"],
        fixed_normalized,
        linestyle=style,
        linewidth=2,
        label=f"Fixed-Time - {name}"
    )


    # LSTM
    plt.plot(
        lstm["step"],
        lstm_normalized,
        linestyle=style,
        linewidth=2,
        label=f"LSTM - {name}"
    )


    # GNN + MAPPO
    plt.plot(
        mappo["step"],
        mappo_normalized,
        linestyle=style,
        linewidth=2,
        label=f"GNN + MAPPO - {name}"
    )


# ============================================================
# GRAPH INFORMATION
# ============================================================

plt.title(
    "Fixed-Time vs LSTM vs GNN + MAPPO",
    fontsize=20,
    fontweight="bold"
)

plt.xlabel(
    "Simulation Step",
    fontsize=14
)

plt.ylabel(
    "Normalized Metric Value (%)",
    fontsize=14
)

plt.ylim(
    0,
    100
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.35
)

plt.legend(
    loc="upper left",
    fontsize=9,
    ncol=3
)

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    PLOT_DIR,
    "combined_traffic_comparison.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("GRAPH GENERATED SUCCESSFULLY")
print("=" * 70)

print("\nGraph contains:")

print("✓ Fixed-Time")
print("✓ LSTM")
print("✓ GNN + MAPPO")

print("\nMetrics:")

print("✓ Vehicle Count")
print("✓ Waiting Time")
print("✓ Average Speed")
print("✓ Queue Length")

print("\nSaved to:")

print(output_file)

print("=" * 70)