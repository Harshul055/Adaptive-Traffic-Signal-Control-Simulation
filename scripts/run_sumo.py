import sys
import os
import csv
import traci


# ==================================================
# PROJECT PATH
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)


# ==================================================
# IMPORT PROJECT MODULES
# ==================================================

from src.environment.state import get_traffic_state
from src.environment.reward import calculate_local_reward


# ==================================================
# SUMO CONFIGURATION
# ==================================================

SUMO_HOME = os.environ.get(
    "SUMO_HOME",
    r"C:\Program Files (x86)\Eclipse\Sumo"
)

SUMO_CONFIG = os.path.join(
    PROJECT_ROOT,
    "sumo-rl", "sumo_rl", "nets", "RESCO",
    "grid4x4", "grid4x4.sumocfg"
)


# ==================================================
# SUMO BINARY
# ==================================================

SUMO_BINARY = os.path.join(
    SUMO_HOME,
    "bin",
    "sumo-gui.exe"
)


# ==================================================
# RESULT FILE
# ==================================================

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "fixed_time_results.csv"
)


# ==================================================
# SIMULATION SETTINGS
# ==================================================

MAX_STEPS = 1000


# ==================================================
# START SUMO
# ==================================================

print("\n======================================")
print("Starting FIXED-TIME SUMO...")
print("======================================")

traci.start([
    SUMO_BINARY,
    "-c",
    SUMO_CONFIG
])

print("SUMO started successfully!")


# ==================================================
# GET TRAFFIC LIGHTS
# ==================================================

traffic_lights = traci.trafficlight.getIDList()

print("\nTraffic lights found:")
print(
    "Number of traffic lights:",
    len(traffic_lights)
)

for tls_id in traffic_lights:
    print("-", tls_id)


# ==================================================
# METRIC STORAGE
# ==================================================

total_waiting_time_sum = 0.0
total_speed_sum = 0.0
total_queue_sum = 0.0
total_vehicle_count_sum = 0.0
total_reward = 0.0


# ==================================================
# CSV FILE
# ==================================================

with open(
    RESULT_FILE,
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "step",
        "vehicle_count",
        "waiting_time",
        "average_speed",
        "queue_length",
        "reward"
    ])


    # ==================================================
    # SIMULATION LOOP
    # ==================================================

    for step in range(MAX_STEPS):

        # ------------------------------------------------
        # Move simulation forward
        # ------------------------------------------------

        traci.simulationStep()


        # ------------------------------------------------
        # Collect traffic state
        # ------------------------------------------------

        state = get_traffic_state()


        # ------------------------------------------------
        # Calculate reward
        # ------------------------------------------------

        reward = calculate_local_reward(state)


        # ------------------------------------------------
        # Extract metrics
        # ------------------------------------------------

        vehicle_count = state["vehicle_count"]

        waiting_time = state["total_waiting_time"]

        average_speed = state["average_speed"]

        queue_length = state["queue_length"]


        # ------------------------------------------------
        # Accumulate metrics
        # ------------------------------------------------

        total_vehicle_count_sum += vehicle_count

        total_waiting_time_sum += waiting_time

        total_speed_sum += average_speed

        total_queue_sum += queue_length

        total_reward += reward


        # ------------------------------------------------
        # Save step data
        # ------------------------------------------------

        writer.writerow([
            step,
            vehicle_count,
            waiting_time,
            average_speed,
            queue_length,
            reward
        ])


        # ------------------------------------------------
        # Display every 100 steps
        # ------------------------------------------------

        if step % 100 == 0:

            print("\n======================================")
            print(
                "Simulation step:",
                step
            )
            print("======================================")

            print(
                "Vehicles:",
                vehicle_count
            )

            print(
                "Total waiting time:",
                round(
                    waiting_time,
                    2
                )
            )

            print(
                "Average speed:",
                round(
                    average_speed,
                    2
                ),
                "m/s"
            )

            print(
                "Queue length:",
                queue_length
            )

            print(
                "Reward:",
                round(
                    reward,
                    2
                )
            )


# ==================================================
# CALCULATE FINAL METRICS
# ==================================================

average_vehicle_count = (
    total_vehicle_count_sum
    / MAX_STEPS
)

average_waiting_time = (
    total_waiting_time_sum
    / MAX_STEPS
)

average_speed = (
    total_speed_sum
    / MAX_STEPS
)

average_queue_length = (
    total_queue_sum
    / MAX_STEPS
)


# ==================================================
# CLOSE SUMO
# ==================================================

print("\n======================================")
print("Closing SUMO...")
print("======================================")

traci.close()


# ==================================================
# DISPLAY FINAL RESULTS
# ==================================================

print("\n======================================")
print("FIXED-TIME RESULTS")
print("======================================")

print(
    "Average vehicles:",
    round(
        average_vehicle_count,
        2
    )
)

print(
    "Average waiting time:",
    round(
        average_waiting_time,
        2
    )
)

print(
    "Average speed:",
    round(
        average_speed,
        2
    ),
    "m/s"
)

print(
    "Average queue length:",
    round(
        average_queue_length,
        2
    )
)

print(
    "Total reward:",
    round(
        total_reward,
        2
    )
)

print("\nResults saved to:")

print(RESULT_FILE)

print("\n======================================")
print("Simulation finished successfully!")
print("======================================")