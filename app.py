"""
Adaptive Traffic Signal Control
Main Application

Run from the project root:

    python app.py

Windows + VS Code / PowerShell
"""

from pathlib import Path
import subprocess
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def header(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def run_file(file_path):
    """
    Runs an existing Python file from the project root.
    """

    file_path = PROJECT_ROOT / file_path

    if not file_path.exists():
        print("\nERROR: File not found:")
        print(file_path)
        input("\nPress Enter to continue...")
        return False

    print("\nRunning:")
    print(file_path)
    print()

    result = subprocess.run(
        [sys.executable, str(file_path)],
        cwd=PROJECT_ROOT
    )

    if result.returncode == 0:
        print("\nCompleted successfully.")
        return True

    print(
        f"\nProcess stopped with exit code: "
        f"{result.returncode}"
    )

    return False


def pause():
    input("\nPress Enter to return to the menu...")


# ============================================================
# INDIVIDUAL OPTIONS
# ============================================================

def run_sumo():

    header("SUMO SIMULATION")

    run_file(
        Path("scripts") / "run_sumo.py"
    )

    pause()


def collect_data():

    header("TRAFFIC DATA COLLECTION")

    run_file(
        Path("scripts") / "collect_data.py"
    )

    print("\nTraffic data location:")
    print(
        PROJECT_ROOT /
        "evaluation" /
        "results" /
        "traffic_data.csv"
    )

    pause()


def train_lstm():

    header("LSTM TRAINING")

    run_file(
        Path("training") /
        "train_lstm.py"
    )

    print("\nLSTM model:")
    print(
        PROJECT_ROOT /
        "models" /
        "lstm" /
        "traffic_lstm.pth"
    )

    pause()


def train_gnn():

    header("GNN TRAINING")

    run_file(
        Path("training") /
        "train_gnn.py"
    )

    print("\nGNN model:")
    print(
        PROJECT_ROOT /
        "models" /
        "gnn" /
        "traffic_gnn.pth"
    )

    pause()


def train_mappo():

    header("MAPPO TRAINING")

    run_file(
        Path("training") /
        "train_mappo.py"
    )

    print("\nMAPPO models:")
    print(
        PROJECT_ROOT /
        "models" /
        "mappo"
    )

    pause()


# ============================================================
# EMERGENCY / VIP
# ============================================================

def generate_emergency():

    header("GENERATE EMERGENCY / VIP SCENARIO")

    run_file(
        Path("scripts") /
        "generate_emergency_scenario.py"
    )

    pause()


def merge_scenario():

    header("MERGE TRAFFIC SCENARIO")

    run_file(
        Path("scripts") /
        "merge_traffic_scenario.py"
    )

    pause()


# ============================================================
# EVALUATION
# ============================================================

def run_mappo_evaluation():

    header("MAPPO EVALUATION")

    run_file(
        Path("scripts") /
        "run_mappo_evaluation.py"
    )

    print("\nResults:")
    print(
        PROJECT_ROOT /
        "evaluation" /
        "results"
    )

    pause()


def run_evaluation():

    header("FINAL EVALUATION")

    run_file(
        Path("scripts") /
        "evaluate.py"
    )

    print("\nResults:")
    print(
        PROJECT_ROOT /
        "evaluation" /
        "results"
    )

    pause()


def final_results():

    header("FINAL RESULT PROCESSING")

    run_file(
        Path("src") /
        "evaluation" /
        "final_results.py"
    )

    print("\nFinal tables:")
    print(
        PROJECT_ROOT /
        "evaluation" /
        "tables"
    )

    pause()


def generate_plots():

    header("FINAL PLOT GENERATION")

    run_file(
        Path("src") /
        "evaluation" /
        "plot_results.py"
    )

    print("\nPlots:")
    print(
        PROJECT_ROOT /
        "evaluation" /
        "plots"
    )

    pause()


# ============================================================
# COMPLETE FINAL RESULT PIPELINE
# ============================================================

def final_result_pipeline():

    header("COMPLETE FINAL RESULT PIPELINE")

    print("""
This uses the existing trained models.

Pipeline:

1. Generate Emergency/VIP scenario
2. Merge traffic scenario
3. Run MAPPO evaluation
4. Run final evaluation
5. Process final results
6. Generate final plots
""")

    confirm = input(
        "Start complete final pipeline? (y/n): "
    ).strip().lower()

    if confirm != "y":
        print("\nPipeline cancelled.")
        pause()
        return

    steps = [

        (
            "Generate Emergency/VIP Scenario",
            Path("scripts") /
            "generate_emergency_scenario.py"
        ),

        (
            "Merge Traffic Scenario",
            Path("scripts") /
            "merge_traffic_scenario.py"
        ),

        (
            "Run MAPPO Evaluation",
            Path("scripts") /
            "run_mappo_evaluation.py"
        ),

        (
            "Run Final Evaluation",
            Path("scripts") /
            "evaluate.py"
        ),

        (
            "Process Final Results",
            Path("src") /
            "evaluation" /
            "final_results.py"
        ),

        (
            "Generate Final Plots",
            Path("src") /
            "evaluation" /
            "plot_results.py"
        )
    ]

    for number, (name, path) in enumerate(
        steps,
        start=1
    ):

        print("\n" + "-" * 65)

        print(
            f"STEP {number}/{len(steps)}: "
            f"{name}"
        )

        print("-" * 65)

        success = run_file(path)

        if not success:

            print(
                f"\nPipeline stopped at:"
                f" {name}"
            )

            pause()
            return

    header(
        "FINAL RESULT PIPELINE COMPLETED"
    )

    print("\nResults:")
    print(
        PROJECT_ROOT /
        "evaluation" /
        "results"
    )

    print("\nTables:")
    print(
        PROJECT_ROOT /
        "evaluation" /
        "tables"
    )

    print("\nPlots:")
    print(
        PROJECT_ROOT /
        "evaluation" /
        "plots"
    )

    pause()


# ============================================================
# FULL TRAINING PIPELINE
# ============================================================

def full_training_pipeline():

    header("FULL TRAINING + EVALUATION PIPELINE")

    print("""
This pipeline performs:

1. Collect traffic data
2. Train LSTM
3. Train GNN
4. Train MAPPO
5. Generate Emergency/VIP scenario
6. Merge traffic scenario
7. Run MAPPO evaluation
8. Run final evaluation
9. Process final results
10. Generate final plots

Training can take a long time.
""")

    confirm = input(
        "Continue? (y/n): "
    ).strip().lower()

    if confirm != "y":
        print("\nCancelled.")
        pause()
        return

    steps = [

        (
            "Collect Traffic Data",
            Path("scripts") /
            "collect_data.py"
        ),

        (
            "Train LSTM",
            Path("training") /
            "train_lstm.py"
        ),

        (
            "Train GNN",
            Path("training") /
            "train_gnn.py"
        ),

        (
            "Train MAPPO",
            Path("training") /
            "train_mappo.py"
        ),

        (
            "Generate Emergency/VIP Scenario",
            Path("scripts") /
            "generate_emergency_scenario.py"
        ),

        (
            "Merge Traffic Scenario",
            Path("scripts") /
            "merge_traffic_scenario.py"
        ),

        (
            "Run MAPPO Evaluation",
            Path("scripts") /
            "run_mappo_evaluation.py"
        ),

        (
            "Run Final Evaluation",
            Path("scripts") /
            "evaluate.py"
        ),

        (
            "Process Final Results",
            Path("src") /
            "evaluation" /
            "final_results.py"
        ),

        (
            "Generate Final Plots",
            Path("src") /
            "evaluation" /
            "plot_results.py"
        )
    ]

    for number, (name, path) in enumerate(
        steps,
        start=1
    ):

        print("\n" + "-" * 65)

        print(
            f"STEP {number}/{len(steps)}: "
            f"{name}"
        )

        print("-" * 65)

        success = run_file(path)

        if not success:

            print(
                f"\nPipeline stopped at:"
                f" {name}"
            )

            pause()
            return

    header(
        "FULL PIPELINE COMPLETED"
    )

    pause()


# ============================================================
# OUTPUT LOCATIONS
# ============================================================

def show_outputs():

    header("PROJECT OUTPUT LOCATIONS")

    locations = {

        "Traffic Data":
            PROJECT_ROOT /
            "evaluation" /
            "results" /
            "traffic_data.csv",

        "LSTM Model":
            PROJECT_ROOT /
            "models" /
            "lstm" /
            "traffic_lstm.pth",

        "GNN Model":
            PROJECT_ROOT /
            "models" /
            "gnn" /
            "traffic_gnn.pth",

        "MAPPO Models":
            PROJECT_ROOT /
            "models" /
            "mappo",

        "CSV Results":
            PROJECT_ROOT /
            "evaluation" /
            "results",

        "Final Tables":
            PROJECT_ROOT /
            "evaluation" /
            "tables",

        "Plots":
            PROJECT_ROOT /
            "evaluation" /
            "plots"
    }

    for name, path in locations.items():

        print(f"\n{name}:")
        print(f"  {path}")

    pause()


# ============================================================
# MAIN MENU
# ============================================================

def main():

    while True:

        header(
            "ADAPTIVE TRAFFIC SIGNAL CONTROL"
        )

        print(
            "\nProject Root:"
        )

        print(
            f"  {PROJECT_ROOT}"
        )

        print("""
---------------------------------------------------------------
 SUMO
---------------------------------------------------------------
  1. Run SUMO simulation
  2. Collect traffic data

---------------------------------------------------------------
 MODEL TRAINING
---------------------------------------------------------------
  3. Train LSTM
  4. Train GNN
  5. Train MAPPO

---------------------------------------------------------------
 EMERGENCY / VIP
---------------------------------------------------------------
  6. Generate Emergency/VIP scenario
  7. Merge traffic scenario

---------------------------------------------------------------
 EVALUATION
---------------------------------------------------------------
  8. Run MAPPO evaluation
  9. Run final evaluation
 10. Process final results
 11. Generate final plots

---------------------------------------------------------------
 COMPLETE PIPELINES
---------------------------------------------------------------
 12. Complete final-result pipeline
 13. Full training + evaluation pipeline

---------------------------------------------------------------
 INFORMATION
---------------------------------------------------------------
 14. Show output locations

---------------------------------------------------------------
  0. Exit
---------------------------------------------------------------
""")

        choice = input(
            "Select an option: "
        ).strip()

        if choice == "1":
            run_sumo()

        elif choice == "2":
            collect_data()

        elif choice == "3":
            train_lstm()

        elif choice == "4":
            train_gnn()

        elif choice == "5":
            train_mappo()

        elif choice == "6":
            generate_emergency()

        elif choice == "7":
            merge_scenario()

        elif choice == "8":
            run_mappo_evaluation()

        elif choice == "9":
            run_evaluation()

        elif choice == "10":
            final_results()

        elif choice == "11":
            generate_plots()

        elif choice == "12":
            final_result_pipeline()

        elif choice == "13":
            full_training_pipeline()

        elif choice == "14":
            show_outputs()

        elif choice == "0":

            print(
                "\nExiting Adaptive Traffic "
                "Signal Control."
            )

            break

        else:

            print(
                "\nInvalid option."
            )

            input(
                "Press Enter to continue..."
            )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    main()