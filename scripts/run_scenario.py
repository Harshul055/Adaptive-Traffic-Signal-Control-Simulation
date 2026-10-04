import os
import sys


# Add project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


from src.environment.sumo_env import SumoEnvironment


from src.emergency.detector import (
    get_emergency_vehicles,
    get_vip_vehicles
)

from src.emergency.route import (
    get_emergency_route_info
)


# Create SUMO environment
env = SumoEnvironment(
    use_gui=True
)


print("Starting SUMO...")

state = env.reset()

print("SUMO started.")
print(
    "Traffic lights:",
    len(state["traffic_lights"])
)

print()
print("Running simulation...")


done = False
step_number = 0


# Remember previous roads
last_emergency_roads = {}
last_vip_roads = {}


while not done:

    # Get traffic lights
    traffic_lights = state["traffic_lights"]

    if not traffic_lights:

        print("No traffic lights found.")
        break


    # Use first traffic light for this test
    tls_id = list(
        traffic_lights.keys()
    )[0]


    # Run one simulation step
    state, reward, done = env.step(
        tls_id
    )

    step_number += 1


    # Detect emergency and VIP vehicles
    emergency = get_emergency_vehicles()
    vip = get_vip_vehicles()


    # =====================================
    # EMERGENCY VEHICLES
    # =====================================

    for vehicle_id in emergency:

        info = get_emergency_route_info(
            vehicle_id
        )

        current_road = info["current_road"]

        previous_road = (
            last_emergency_roads.get(
                vehicle_id
            )
        )


        # Print only when road changes
        if current_road != previous_road:

            print()
            print(
                f"Step {step_number}:"
            )

            print(
                "  EMERGENCY:",
                vehicle_id
            )

            print(
                "  Previous road:",
                previous_road
            )

            print(
                "  Current road:",
                current_road
            )

            print(
                "  Next road:",
                info["next_road"]
            )


            last_emergency_roads[
                vehicle_id
            ] = current_road


    # =====================================
    # VIP VEHICLES
    # =====================================

    for vehicle_id in vip:

        info = get_emergency_route_info(
            vehicle_id
        )

        current_road = info["current_road"]

        previous_road = (
            last_vip_roads.get(
                vehicle_id
            )
        )


        # Print only when road changes
        if current_road != previous_road:

            print()
            print(
                f"Step {step_number}:"
            )

            print(
                "  VIP:",
                vehicle_id
            )

            print(
                "  Previous road:",
                previous_road
            )

            print(
                "  Current road:",
                current_road
            )

            print(
                "  Next road:",
                info["next_road"]
            )


            last_vip_roads[
                vehicle_id
            ] = current_road


    # =====================================
    # TEST LIMIT
    # =====================================

    if step_number >= 1000:

        print()
        print(
            "Test limit reached."
        )

        break


print()
print("Simulation finished.")
print(
    "Total steps:",
    step_number
)


env.close()
