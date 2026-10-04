import os
import csv
import sys
import traci


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
# SUMO CONFIGURATION
# ==================================================

SUMO_BINARY = "sumo"

SUMO_CONFIG = os.path.join(
    PROJECT_ROOT,
    "sumo-rl",
    "sumo_rl",
    "nets",
    "RESCO",
    "grid4x4",
    "grid4x4.sumocfg"
)


# ==================================================
# OUTPUT FILE
# ==================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "traffic_data.csv"
)


# ==================================================
# SETTINGS
# ==================================================

MAX_STEPS = 1000


# ==================================================
# START SUMO
# ==================================================

def start_sumo():

    sumo_command = [
        SUMO_BINARY,
        "-c",
        SUMO_CONFIG
    ]

    traci.start(
        sumo_command
    )


# ==================================================
# COLLECT DATA
# ==================================================

def collect_data():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    start_sumo()

    print()
    print("=" * 50)
    print("TRAFFIC DATA COLLECTION")
    print("=" * 50)

    print()
    print(
        f"Output file:\n{OUTPUT_FILE}"
    )

    rows = []

    try:

        for step in range(
            MAX_STEPS
        ):

            traci.simulationStep()

            vehicle_ids = (
                traci.vehicle.getIDList()
            )

            vehicle_count = len(
                vehicle_ids
            )

            total_waiting_time = 0.0
            total_speed = 0.0
            queue_length = 0

            for vehicle_id in vehicle_ids:

                speed = traci.vehicle.getSpeed(
                    vehicle_id
                )

                waiting_time = traci.vehicle.getAccumulatedWaitingTime(
                    vehicle_id
                )

                total_speed += speed
                total_waiting_time += waiting_time

                if speed < 0.1:
                    queue_length += 1

            if vehicle_count > 0:

                average_speed = (
                    total_speed
                    / vehicle_count
                )

            else:

                average_speed = 0.0

            rows.append([
                step,
                vehicle_count,
                total_waiting_time,
                average_speed,
                queue_length
            ])

            if step % 100 == 0:

                print(
                    f"Step {step:04d} | "
                    f"Vehicles: {vehicle_count} | "
                    f"Waiting: {total_waiting_time:.2f} | "
                    f"Speed: {average_speed:.2f} | "
                    f"Queue: {queue_length}"
                )

    finally:

       
        traci.close()

    # ==================================================
    # SAVE CSV
    # ==================================================

    with open(
        OUTPUT_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "step",
            "vehicle_count",
            "waiting_time",
            "average_speed",
            "queue_length"
        ])

        writer.writerows(rows)

    print()
    print(
        f"Collected records: {len(rows)}"
    )

    print(
        f"Data saved to:\n{OUTPUT_FILE}"
    )

    print()
    print("=" * 50)
    print("DATA COLLECTION COMPLETE")
    print("=" * 50)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    collect_data()