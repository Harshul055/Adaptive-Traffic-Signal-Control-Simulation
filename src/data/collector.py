import csv
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.environment.sumo_env import SumoEnvironment


def collect_data(num_steps=1000):

    env = SumoEnvironment(use_gui=True)

    state = env.reset()

    output_file = os.path.join(
        PROJECT_ROOT,
        "evaluation",
        "results",
        "traffic_data.csv"
    )

    with open(output_file, "w", newline="") as file:

        writer = csv.writer(file)

        # CSV header
        writer.writerow([
            "step",
            "vehicle_count",
            "waiting_time",
            "average_speed",
            "queue_length"
        ])

        for step in range(num_steps):

            state, reward, done = env.step("A0")

            writer.writerow([
                step,
                state["vehicle_count"],
                state["total_waiting_time"],
                state["average_speed"],
                state["queue_length"]
            ])

            if step % 100 == 0:
                print(
                    "Step:", step,
                    "| Vehicles:", state["vehicle_count"],
                    "| Waiting:", round(
                        state["total_waiting_time"], 2
                    ),
                    "| Speed:", round(
                        state["average_speed"], 2
                    ),
                    "| Queue:", state["queue_length"]
                )

            if done:
                print("Simulation finished.")
                break

            time.sleep(0.01)

    env.close()

    print("\n======================================")
    print("Data collection completed!")
    print("======================================")
    print("Dataset saved to:")
    print(output_file)


if __name__ == "__main__":
    collect_data(500)