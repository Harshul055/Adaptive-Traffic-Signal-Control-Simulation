import os
import sys
import traci

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = r"C:\Project\Adaptive-Traffic-Control"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------
# SUMO
# --------------------------------------------------

SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"

SUMO_CONFIG = (
    r"C:\Project\Adaptive-Traffic-Control"
    r"\sumo-rl\sumo_rl\nets\RESCO\grid4x4\grid4x4.sumocfg"
)

# --------------------------------------------------
# START SUMO
# --------------------------------------------------

sumo_cmd = [
    SUMO_BINARY,
    "-c",
    SUMO_CONFIG,
    "--start"
]

traci.start(sumo_cmd)

print()
print("==============================================")
print(" EMERGENCY / VIP DETECTION TEST")
print("==============================================")
print()

detected_emergency = set()
detected_vip = set()

# --------------------------------------------------
# RUN 1000 STEPS
# --------------------------------------------------

for step in range(1000):

    traci.simulationStep()

    vehicle_ids = traci.vehicle.getIDList()

    for vehicle_id in vehicle_ids:

        vehicle_type = traci.vehicle.getTypeID(vehicle_id).lower()

        # Emergency
        if vehicle_type in ["emergency", "ambulance", "fire", "police"]:

            if vehicle_id not in detected_emergency:

                detected_emergency.add(vehicle_id)

                print(
                    f"Step {step:4d} | "
                    f"EMERGENCY DETECTED | "
                    f"{vehicle_id}"
                )

        # VIP
        elif vehicle_type == "vip":

            if vehicle_id not in detected_vip:

                detected_vip.add(vehicle_id)

                print(
                    f"Step {step:4d} | "
                    f"VIP DETECTED       | "
                    f"{vehicle_id}"
                )

# --------------------------------------------------
# STOP SUMO
# --------------------------------------------------

traci.close()

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print()
print("==============================================")
print(" DETECTION RESULTS")
print("==============================================")
print()

print(
    f"Emergency detected : "
    f"{len(detected_emergency)} / 5"
)

print(
    f"VIP detected       : "
    f"{len(detected_vip)} / 5"
)

print()

print("Emergency vehicles:")
for vehicle_id in sorted(detected_emergency):
    print("  ", vehicle_id)

print()

print("VIP vehicles:")
for vehicle_id in sorted(detected_vip):
    print("  ", vehicle_id)

print()

if len(detected_emergency) == 5 and len(detected_vip) == 5:
    print("SUCCESS: ALL 10 SPECIAL VEHICLES DETECTED")
else:
    print("WARNING: SOME SPECIAL VEHICLES WERE NOT DETECTED")

print()
print("==============================================")