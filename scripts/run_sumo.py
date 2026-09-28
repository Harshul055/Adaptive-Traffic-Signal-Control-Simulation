import sys
import os
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
from src.environment.action import change_phase
from src.environment.reward import calculate_reward


# ==================================================
# SUMO CONFIGURATION
# ==================================================

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"

SUMO_CONFIG = (
    r"C:\Project\Adaptive-Traffic-Control"
    r"\sumo-rl\sumo_rl\nets\RESCO\grid4x4"
    r"\grid4x4.sumocfg"
)


# ==================================================
# SELECT SUMO PROGRAM
# ==================================================

SUMO_BINARY = os.path.join(
    SUMO_HOME,
    "bin",
    "sumo-gui.exe"
)

# For normal SUMO without GUI, use:
# SUMO_BINARY = os.path.join(
#     SUMO_HOME,
#     "bin",
#     "sumo.exe"
# )


# ==================================================
# START SUMO
# ==================================================

print("\n======================================")
print("Starting SUMO...")
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
print("Number of traffic lights:", len(traffic_lights))

for tls_id in traffic_lights:
    print("-", tls_id)


# ==================================================
# SIMULATION LOOP
# ==================================================

for step in range(1000):

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

    reward = calculate_reward(state)


    # ------------------------------------------------
    # Display information every 100 steps
    # ------------------------------------------------

    if step % 100 == 0:

        print("\n======================================")
        print("Simulation step:", step)
        print("======================================")

        print(
            "Vehicles:",
            state["vehicle_count"]
        )

        print(
            "Total waiting time:",
            round(
                state["total_waiting_time"],
                2
            )
        )

        print(
            "Average speed:",
            round(
                state["average_speed"],
                2
            ),
            "m/s"
        )

        print(
            "Queue length:",
            state["queue_length"]
        )

        print(
            "Reward:",
            round(
                reward,
                2
            )
        )


        # ------------------------------------------------
        # Display traffic-light information
        # ------------------------------------------------

        print("\nTraffic Light States:")

        for tls_id, tls_data in state[
            "traffic_lights"
        ].items():

            print(
                tls_id,
                "| Phase:",
                tls_data["phase"],
                "| Phase duration:",
                tls_data["phase_duration"]
            )


    # ------------------------------------------------
    # TEST TRAFFIC-LIGHT CONTROL
    # ------------------------------------------------

    if step > 0 and step % 100 == 0:

        print("\nChanging traffic-light phases...")

        for tls_id in traffic_lights:

            change_phase(tls_id)

            print(
                "Changed phase:",
                tls_id
            )


# ==================================================
# CLOSE SUMO
# ==================================================

print("\n======================================")
print("Closing SUMO...")
print("======================================")

traci.close()

print("\nSimulation finished successfully!")