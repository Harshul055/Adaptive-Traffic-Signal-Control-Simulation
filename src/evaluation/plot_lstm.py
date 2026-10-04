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

ACTUAL_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results",
    "traffic_data.csv"
)

PREDICTED_FILE = os.path.join(
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

print("=" * 65)
print("LSTM ACTUAL VS PREDICTED GRAPH")
print("=" * 65)

actual = pd.read_csv(ACTUAL_FILE)
predicted = pd.read_csv(PREDICTED_FILE)

# Match actual and predicted data using step
data = pd.merge(
    actual,
    predicted,
    on="step",
    suffixes=("_actual", "_predicted")
)

print(f"\nMatched rows: {len(data)}")

metrics = [
    ("vehicle_count", "Vehicle Count"),
    ("waiting_time", "Waiting Time"),
    ("average_speed", "Average Speed"),
    ("queue_length", "Queue Length")
]

plt.figure(figsize=(16, 10))

for index, (column, label) in enumerate(metrics):

    plt.subplot(2, 2, index + 1)

    plt.plot(
        data["step"],
        data[f"{column}_actual"],
        label="Actual",
        linewidth=2
    )

    plt.plot(
        data["step"],
        data[f"{column}_predicted"],
        label="LSTM Predicted",
        linestyle="--",
        linewidth=2
    )

    plt.title(label)
    plt.xlabel("Simulation Step")
    plt.ylabel(label)
    plt.grid(
        True,
        linestyle="--",
        alpha=0.35
    )
    plt.legend()

plt.suptitle(
    "LSTM Traffic Prediction: Actual vs Predicted",
    fontsize=20,
    fontweight="bold"
)

plt.tight_layout()

output_file = os.path.join(
    PLOT_DIR,
    "lstm_actual_vs_predicted.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nGraph saved to:")
print(output_file)

print("\n" + "=" * 65)
print("LSTM GRAPH GENERATED")
print("=" * 65)