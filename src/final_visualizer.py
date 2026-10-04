from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FIND PROJECT ROOT
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = None

for folder in [CURRENT_DIR] + list(CURRENT_DIR.parents):

    if (
        (folder / "evaluation").exists()
        and (folder / "src").exists()
    ):
        PROJECT_DIR = folder
        break

if PROJECT_DIR is None:
    raise FileNotFoundError(
        "Could not find Adaptive-Traffic-Control project root."
    )


RESULTS_DIR = PROJECT_DIR / "evaluation" / "results"

PLOTS_DIR = (
    PROJECT_DIR
    / "evaluation"
    / "plots"
    / "final"
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print()
print("====================================================")
print(" FINAL MATPLOTLIB VISUALIZER")
print("====================================================")
print(f"Project folder : {PROJECT_DIR}")
print(f"Results folder : {RESULTS_DIR}")
print(f"Output folder  : {PLOTS_DIR}")
print()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(filename):

    path = RESULTS_DIR / filename

    if not path.exists():

        print(f"[SKIP] {filename} not found")

        return None

    try:

        df = pd.read_csv(path)

        print(
            f"[LOADED] {filename} "
            f"({len(df)} rows)"
        )

        return df

    except Exception as e:

        print(
            f"[ERROR] {filename}: {e}"
        )

        return None


def save_plot(filename):

    path = PLOTS_DIR / filename

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[CREATED] {filename}"
    )


# ============================================================
# LOAD PROJECT DATA
# ============================================================

final_df = load_csv(
    "final_emergency_comparison.csv"
)

traffic_df = load_csv(
    "traffic_data.csv"
)

lstm_df = load_csv(
    "lstm_predictions.csv"
)

fixed_df = load_csv(
    "fixed_time_results.csv"
)

mappo_df = load_csv(
    "mappo_final_results.csv"
)


# ============================================================
# FINAL COMPARISON GRAPHS
# ============================================================

if final_df is not None:

    # --------------------------------------------------------
    # Vehicle Count
    # --------------------------------------------------------

    row = final_df[
        final_df["Metric"] == "Vehicle Count"
    ]

    if not row.empty:

        row = row.iloc[0]

        plt.figure(figsize=(9, 5))

        plt.bar(
            [
                "Fixed-Time",
                "GNN + MAPPO\n+ Emergency/VIP"
            ],
            [
                row["Fixed-Time"],
                row["GNN + MAPPO + Emergency/VIP"]
            ]
        )

        plt.title(
            "Vehicle Count Comparison"
        )

        plt.ylabel(
            "Average Vehicles"
        )

        save_plot(
            "01_vehicle_count.png"
        )


    # --------------------------------------------------------
    # Waiting Time
    # --------------------------------------------------------

    row = final_df[
        final_df["Metric"] == "Waiting Time"
    ]

    if not row.empty:

        row = row.iloc[0]

        plt.figure(figsize=(9, 5))

        plt.bar(
            [
                "Fixed-Time",
                "GNN + MAPPO\n+ Emergency/VIP"
            ],
            [
                row["Fixed-Time"],
                row["GNN + MAPPO + Emergency/VIP"]
            ]
        )

        plt.title(
            "Waiting Time Comparison"
        )

        plt.ylabel(
            "Average Waiting Time"
        )

        save_plot(
            "02_waiting_time.png"
        )


    # --------------------------------------------------------
    # Average Speed
    # --------------------------------------------------------

    row = final_df[
        final_df["Metric"] == "Average Speed"
    ]

    if not row.empty:

        row = row.iloc[0]

        plt.figure(figsize=(9, 5))

        plt.bar(
            [
                "Fixed-Time",
                "GNN + MAPPO\n+ Emergency/VIP"
            ],
            [
                row["Fixed-Time"],
                row["GNN + MAPPO + Emergency/VIP"]
            ]
        )

        plt.title(
            "Average Speed Comparison"
        )

        plt.ylabel(
            "Average Speed (m/s)"
        )

        save_plot(
            "03_average_speed.png"
        )


    # --------------------------------------------------------
    # Queue Length
    # --------------------------------------------------------

    row = final_df[
        final_df["Metric"] == "Queue Length"
    ]

    if not row.empty:

        row = row.iloc[0]

        plt.figure(figsize=(9, 5))

        plt.bar(
            [
                "Fixed-Time",
                "GNN + MAPPO\n+ Emergency/VIP"
            ],
            [
                row["Fixed-Time"],
                row["GNN + MAPPO + Emergency/VIP"]
            ]
        )

        plt.title(
            "Queue Length Comparison"
        )

        plt.ylabel(
            "Average Queue Length"
        )

        save_plot(
            "04_queue_length.png"
        )


    # --------------------------------------------------------
    # Overall Improvement
    # --------------------------------------------------------

    if "Improvement (%)" in final_df.columns:

        plt.figure(figsize=(10, 5))

        plt.bar(
            final_df["Metric"],
            final_df["Improvement (%)"]
        )

        plt.title(
            "Overall Performance Improvement"
        )

        plt.ylabel(
            "Improvement (%)"
        )

        plt.ylim(
            0,
            100
        )

        plt.xticks(
            rotation=15
        )

        save_plot(
            "05_overall_improvement.png"
        )


    # --------------------------------------------------------
    # Horizontal Improvement
    # --------------------------------------------------------

    if "Improvement (%)" in final_df.columns:

        plt.figure(figsize=(9, 5))

        plt.barh(
            final_df["Metric"],
            final_df["Improvement (%)"]
        )

        plt.title(
            "Final Performance Gains"
        )

        plt.xlabel(
            "Improvement (%)"
        )

        plt.xlim(
            0,
            100
        )

        save_plot(
            "06_horizontal_improvement.png"
        )


    # --------------------------------------------------------
    # Normalized Comparison
    # --------------------------------------------------------

    if (
        "Fixed-Time" in final_df.columns
        and
        "GNN + MAPPO + Emergency/VIP"
        in final_df.columns
    ):

        fixed_values = []
        mappo_values = []

        for _, row in final_df.iterrows():

            fixed = float(
                row["Fixed-Time"]
            )

            mappo = float(
                row[
                    "GNN + MAPPO + Emergency/VIP"
                ]
            )

            fixed_values.append(100)

            if fixed != 0:

                mappo_values.append(
                    mappo / fixed * 100
                )

            else:

                mappo_values.append(0)

        x = np.arange(
            len(final_df)
        )

        width = 0.35

        plt.figure(
            figsize=(11, 5)
        )

        plt.bar(
            x - width / 2,
            fixed_values,
            width,
            label="Fixed-Time"
        )

        plt.bar(
            x + width / 2,
            mappo_values,
            width,
            label="GNN + MAPPO + Emergency/VIP"
        )

        plt.xticks(
            x,
            final_df["Metric"],
            rotation=15
        )

        plt.ylabel(
            "Relative Value (% of Fixed-Time)"
        )

        plt.title(
            "Normalized Performance Comparison"
        )

        plt.legend()

        save_plot(
            "07_normalized_comparison.png"
        )


# ============================================================
# TRAFFIC DATA GRAPHS
# ============================================================

if traffic_df is not None:

    if "step" in traffic_df.columns:

        metrics = [

            (
                "vehicle_count",
                "Vehicle Count",
                "08_vehicle_count_over_time.png"
            ),

            (
                "waiting_time",
                "Waiting Time",
                "09_waiting_time_over_time.png"
            ),

            (
                "average_speed",
                "Average Speed",
                "10_average_speed_over_time.png"
            ),

            (
                "queue_length",
                "Queue Length",
                "11_queue_length_over_time.png"
            )

        ]

        for column, title, filename in metrics:

            if column not in traffic_df.columns:

                continue

            plt.figure(
                figsize=(10, 5)
            )

            plt.plot(
                traffic_df["step"],
                traffic_df[column]
            )

            plt.title(
                title
                + " Over Simulation"
            )

            plt.xlabel(
                "Simulation Step"
            )

            plt.ylabel(
                title
            )

            save_plot(
                filename
            )


# ============================================================
# ALL TRAFFIC METRICS
# ============================================================

if traffic_df is not None:

    required = [
        "step",
        "vehicle_count",
        "waiting_time",
        "average_speed",
        "queue_length"
    ]

    if all(
        column in traffic_df.columns
        for column in required
    ):

        plt.figure(
            figsize=(11, 6)
        )

        plt.plot(
            traffic_df["step"],
            traffic_df["vehicle_count"],
            label="Vehicle Count"
        )

        plt.plot(
            traffic_df["step"],
            traffic_df["waiting_time"],
            label="Waiting Time"
        )

        plt.plot(
            traffic_df["step"],
            traffic_df["average_speed"],
            label="Average Speed"
        )

        plt.plot(
            traffic_df["step"],
            traffic_df["queue_length"],
            label="Queue Length"
        )

        plt.title(
            "Traffic Metrics Over Simulation"
        )

        plt.xlabel(
            "Simulation Step"
        )

        plt.ylabel(
            "Metric Value"
        )

        plt.legend()

        save_plot(
            "12_all_traffic_metrics.png"
        )


# ============================================================
# DISTRIBUTION GRAPHS
# ============================================================

if traffic_df is not None:

    distributions = [

        (
            "vehicle_count",
            "Vehicle Count",
            "13_vehicle_distribution.png"
        ),

        (
            "waiting_time",
            "Waiting Time",
            "14_waiting_distribution.png"
        ),

        (
            "average_speed",
            "Average Speed",
            "15_speed_distribution.png"
        ),

        (
            "queue_length",
            "Queue Length",
            "16_queue_distribution.png"
        )

    ]

    for column, title, filename in distributions:

        if column not in traffic_df.columns:

            continue

        plt.figure(
            figsize=(9, 5)
        )

        plt.hist(
            traffic_df[column].dropna(),
            bins=20
        )

        plt.title(
            title
            + " Distribution"
        )

        plt.xlabel(
            title
        )

        plt.ylabel(
            "Frequency"
        )

        save_plot(
            filename
        )


# ============================================================
# CORRELATION HEATMAP
# ============================================================

if traffic_df is not None:

    numeric = traffic_df.select_dtypes(
        include=np.number
    )

    if numeric.shape[1] >= 2:

        correlation = numeric.corr()

        plt.figure(
            figsize=(9, 7)
        )

        plt.imshow(
            correlation
        )

        plt.colorbar()

        plt.xticks(
            range(
                len(
                    correlation.columns
                )
            ),
            correlation.columns,
            rotation=45,
            ha="right"
        )

        plt.yticks(
            range(
                len(
                    correlation.columns
                )
            ),
            correlation.columns
        )

        plt.title(
            "Traffic Metrics Correlation"
        )

        save_plot(
            "17_correlation_heatmap.png"
        )


# ============================================================
# LSTM GRAPHS
# ============================================================

if lstm_df is not None:

    pairs = [

        (
            "vehicle_count_actual",
            "vehicle_count_predicted",
            "Vehicle Count"
        ),

        (
            "waiting_time_actual",
            "waiting_time_predicted",
            "Waiting Time"
        ),

        (
            "average_speed_actual",
            "average_speed_predicted",
            "Average Speed"
        ),

        (
            "queue_length_actual",
            "queue_length_predicted",
            "Queue Length"
        )

    ]

    for actual, predicted, title in pairs:

        if (
            actual not in lstm_df.columns
            or
            predicted not in lstm_df.columns
        ):

            continue

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            lstm_df[actual],
            label="Actual"
        )

        plt.plot(
            lstm_df[predicted],
            label="Predicted"
        )

        plt.title(
            "LSTM - "
            + title
            + " Actual vs Predicted"
        )

        plt.xlabel(
            "Sample"
        )

        plt.ylabel(
            title
        )

        plt.legend()

        save_plot(
            "18_lstm_"
            + title.lower().replace(" ", "_")
            + ".png"
        )


# ============================================================
# FINAL RESULTS TABLE
# ============================================================

if final_df is not None:

    fig, ax = plt.subplots(
        figsize=(13, 4)
    )

    ax.axis(
        "off"
    )

    table = ax.table(
        cellText=final_df.round(2).values,
        colLabels=final_df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(
        False
    )

    table.set_fontsize(
        10
    )

    table.scale(
        1,
        2
    )

    plt.title(
        "Final Adaptive Traffic Control Results",
        pad=20
    )

    save_plot(
        "19_final_results_table.png"
    )


# ============================================================
# FINAL PROJECT GAINS
# ============================================================

if final_df is not None:

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        final_df["Metric"],
        final_df["Improvement (%)"]
    )

    plt.title(
        "Adaptive Traffic Control - Final Gains"
    )

    plt.ylabel(
        "Improvement (%)"
    )

    plt.ylim(
        0,
        100
    )

    plt.xticks(
        rotation=15
    )

    save_plot(
        "20_final_project_gains.png"
    )


# ============================================================
# FINISH
# ============================================================

files = sorted(
    PLOTS_DIR.glob("*.png")
)

print()
print("====================================================")
print(" FINAL MATPLOTLIB VISUALIZATION COMPLETE")
print("====================================================")

print(
    f"Graphs saved in:\n{PLOTS_DIR}"
)

print(
    f"\nTotal graphs generated: {len(files)}"
)

for file in files:

    print(
        " -",
        file.name
    )

print("====================================================")