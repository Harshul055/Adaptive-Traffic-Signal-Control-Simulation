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
# EVALUATION SCRIPT
# ==================================================

EVALUATION_SCRIPT = os.path.join(
    PROJECT_ROOT,
    "src",
    "evaluation",
    "evaluate.py"
)


# ==================================================
# RUN EVALUATION
# ==================================================

def run_evaluation():

    print()
    print("=" * 60)
    print("ADAPTIVE TRAFFIC CONTROL")
    print("EVALUATION")
    print("=" * 60)

    if not os.path.exists(
        EVALUATION_SCRIPT
    ):

        print()
        print(
            "Evaluation script not found:"
        )

        print(
            EVALUATION_SCRIPT
        )

        return

    print()
    print(
        "Running evaluation..."
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.evaluate"
        ],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:

        print()
        print(
            "Evaluation failed."
        )

        print(
            f"Return code: "
            f"{result.returncode}"
        )

        return

    print()
    print(
        "Evaluation completed successfully."
    )

    print("=" * 60)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    run_evaluation()