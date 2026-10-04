import pandas as pd
from pathlib import Path

# --------------------------------------------------
# FILES
# --------------------------------------------------

BASE_DIR = Path("C:/Project/Adaptive-Traffic-Control")

FIXED_FILE = BASE_DIR / "evaluation/results/fixed_time_results.csv"
FINAL_FILE = BASE_DIR / "evaluation/results/mappo_final_results.csv"

OUTPUT_FILE = BASE_DIR / "evaluation/results/final_emergency_comparison.csv"

# --------------------------------------------------
# LOAD RESULTS
# --------------------------------------------------

fixed = pd.read_csv(FIXED_FILE)
final = pd.read_csv(FINAL_FILE)

# --------------------------------------------------
# CALCULATE AVERAGES
# --------------------------------------------------

metrics = [
    "vehicle_count",
    "waiting_time",
    "average_speed",
    "queue_length"
]

fixed_avg = fixed[metrics].mean()
final_avg = final[metrics].mean()

# --------------------------------------------------
# CALCULATE IMPROVEMENT
# --------------------------------------------------

comparison = pd.DataFrame({
    "Metric": [
        "Vehicle Count",
        "Waiting Time",
        "Average Speed",
        "Queue Length"
    ],
    "Fixed-Time": [
        fixed_avg["vehicle_count"],
        fixed_avg["waiting_time"],
        fixed_avg["average_speed"],
        fixed_avg["queue_length"]
    ],
    "GNN + MAPPO + Emergency/VIP": [
        final_avg["vehicle_count"],
        final_avg["waiting_time"],
        final_avg["average_speed"],
        final_avg["queue_length"]
    ]
})

# Lower is better for these metrics
comparison["Improvement (%)"] = [
    (fixed_avg["vehicle_count"] - final_avg["vehicle_count"])
    / fixed_avg["vehicle_count"] * 100,

    (fixed_avg["waiting_time"] - final_avg["waiting_time"])
    / fixed_avg["waiting_time"] * 100,

    (final_avg["average_speed"] - fixed_avg["average_speed"])
    / fixed_avg["average_speed"] * 100,

    (fixed_avg["queue_length"] - final_avg["queue_length"])
    / fixed_avg["queue_length"] * 100
]

# --------------------------------------------------
# ROUND VALUES
# --------------------------------------------------

comparison["Fixed-Time"] = comparison["Fixed-Time"].round(2)
comparison["GNN + MAPPO + Emergency/VIP"] = (
    comparison["GNN + MAPPO + Emergency/VIP"].round(2)
)
comparison["Improvement (%)"] = comparison["Improvement (%)"].round(2)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

comparison.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print()
print("==============================================")
print(" FINAL EMERGENCY/VIP COMPARISON")
print("==============================================")
print()

print(comparison.to_string(index=False))

print()
print(f"Saved to:")
print(OUTPUT_FILE)

print()
print("==============================================")
