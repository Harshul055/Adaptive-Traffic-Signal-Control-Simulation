import os
import pandas as pd
import matplotlib.pyplot as plt

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

os.makedirs(PLOT_DIR, exist_ok=True)

fixed = pd.read_csv(FIXED_FILE)
mappo = pd.read_csv(MAPPO_FILE)

# Use equal number of simulation steps
common_steps = min(len(fixed), len(mappo))

fixed = fixed.iloc[:common_steps]
mappo = mappo.iloc[:common_steps]

print("=" * 70)
print("FINAL GNN + MAPPO PERFORMANCE")
print("=" * 70)

print(f"\nComparison steps: {common_steps}")

metrics = {
    "vehicle_count": "Vehicle Count",
    "waiting_time": "Waiting Time",
    "average_speed": "Average Speed",
    "queue_length": "Queue Length"
}

results = []

for column, label in metrics.items():

    fixed_value = fixed[column].mean()
    mappo_value = mappo[column].mean()

    if column == "average_speed":
        improvement = (
            (mappo_value - fixed_value)
            / fixed_value
        ) * 100
    else:
        improvement = (
            (fixed_value - mappo_value)
            / fixed_value
        ) * 100

    results.append({
        "Metric": label,
        "Fixed-Time": fixed_value,
        "GNN + MAPPO": mappo_value,
        "Improvement (%)": improvement
    })

results_df = pd.DataFrame(results)

print("\nFINAL RESULTS")
print("-" * 70)

for _, row in results_df.iterrows():

    print(
        f"{row['Metric']:<20}"
        f" Fixed-Time = {row['Fixed-Time']:.2f}"
        f" | GNN + MAPPO = {row['GNN + MAPPO']:.2f}"
        f" | Improvement = {row['Improvement (%)']:.2f}%"
    )

# Save results table
table_file = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "tables",
    "final_results.csv"
)

os.makedirs(
    os.path.dirname(table_file),
    exist_ok=True
)

results_df.to_csv(
    table_file,
    index=False
)

# Plot percentage improvement
plt.figure(figsize=(12, 7))

bars = plt.bar(
    results_df["Metric"],
    results_df["Improvement (%)"]
)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

plt.axhline(
    0,
    linewidth=1
)

plt.ylabel(
    "Improvement (%)",
    fontsize=13
)

plt.xlabel(
    "Traffic Performance Metrics",
    fontsize=13
)

plt.title(
    "GNN + MAPPO Improvement Over Fixed-Time Control",
    fontsize=18,
    fontweight="bold"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.35
)

plt.tight_layout()

plot_file = os.path.join(
    PLOT_DIR,
    "final_percentage_improvement.png"
)

plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nResults table saved:")
print(table_file)

print("\nGraph saved:")
print(plot_file)

print("\n" + "=" * 70)
print("FINAL RESULTS GENERATED")
print("=" * 70)