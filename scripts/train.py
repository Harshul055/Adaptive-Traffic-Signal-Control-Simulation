import os
import sys
import subprocess


# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==================================================
# TRAINING SCRIPTS
# ==================================================

TRAINING_SCRIPTS = [
    (
        "LSTM",
        "training/train_lstm.py"
    ),
    (
        "GNN",
        "training/train_gnn.py"
    ),
    (
        "MAPPO",
        "training/train_mappo.py"
    )
]


# ==================================================
# RUN TRAINING
# ==================================================

def run_training():

    print()
    print("=" * 60)
    print("ADAPTIVE TRAFFIC CONTROL")
    print("TRAINING PIPELINE")
    print("=" * 60)

    for name, script in TRAINING_SCRIPTS:

        script_path = os.path.join(
            PROJECT_ROOT,
            script
        )

        print()
        print("-" * 60)
        print(f"Starting {name} training...")
        print("-" * 60)

        if not os.path.exists(script_path):

            print(
                f"Training script not found:\n"
                f"{script_path}"
            )

            continue

        result = subprocess.run(
            [
                sys.executable,
                script_path
            ],
            cwd=PROJECT_ROOT
        )

        if result.returncode != 0:

            print()
            print(
                f"{name} training failed."
            )

            print(
                f"Return code: "
                f"{result.returncode}"
            )

            return

        print()
        print(
            f"{name} training completed."
        )

    print()
    print("=" * 60)
    print("ALL TRAINING COMPLETED")
    print("=" * 60)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    run_training()