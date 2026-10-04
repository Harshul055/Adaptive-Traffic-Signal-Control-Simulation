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

PLOT_DIR = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "plots"
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("FINAL TRAFFIC CONTROL PERFORMANCE GRAPH")
print("=" * 65)

fixed = pd.read_csv(
    FIXED_FILE
)

mappo = pd.read_csv(
    MAPPO_FILE
)


print(
    f"\nFixed-Time rows : {len(fixed)}"
)

print(
    f"GNN + MAPPO rows: {len(mappo)}"
)


# ============================================================
# COMMON DATA LENGTH
# ============================================================

common_steps = min(
    len(fixed),
    len(mappo)
)

fixed = fixed.iloc[
    :common_steps
].copy()

mappo = mappo.iloc[
    :common_steps
].copy()


# ============================================================
# METRICS
# ============================================================

metrics = {

    "vehicle_count":
        "Vehicle Count",

    "waiting_time":
        "Waiting Time",

    "average_speed":
        "Average Speed",

    "queue_length":
        "Queue Length"
}


# ============================================================
# CALCULATE AVERAGES
# ============================================================

fixed_average = []
mappo_average = []
labels = []


for column, label in metrics.items():

    labels.append(label)

    fixed_average.append(
        fixed[column].mean()
    )

    mappo_average.append(
        mappo[column].mean()
    )


# ============================================================
# CREATE GRAPH
# ============================================================

plt.figure(
    figsize=(14, 8)
)


x = range(
    len(labels)
)

width = 0.35


bars1 = plt.bar(
    [i - width / 2 for i in x],
    fixed_average,
    width,
    label="Fixed-Time"
)


bars2 = plt.bar(
    [i + width / 2 for i in x],
    mappo_average,
    width,
    label="GNN + MAPPO"
)


# ============================================================
# VALUE LABELS
# ============================================================

for bars in [
    bars1,
    bars2
]:

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )


# ============================================================
# GRAPH SETTINGS
# ============================================================

plt.xticks(
    list(x),
    labels
)

plt.ylabel(
    "Average Value",
    fontsize=13
)

plt.xlabel(
    "Traffic Performance Metrics",
    fontsize=13
)

plt.title(
    "Fixed-Time vs GNN + MAPPO",
    fontsize=20,
    fontweight="bold"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.35
)

plt.legend(
    fontsize=11
)

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    PLOT_DIR,
    "final_control_comparison.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nAverage Results:")

for i, label in enumerate(labels):

    print(
        f"{label:<20} "
        f"Fixed-Time = {fixed_average[i]:.2f}   "
        f"GNN + MAPPO = {mappo_average[i]:.2f}"
    )


print("\nGraph saved to:")

print(output_file)

print("\n" + "=" * 65)
print("FINAL CONTROL GRAPH GENERATED")
print("=" * 65)