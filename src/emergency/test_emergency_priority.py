import traci

from src.emergency.detector import (
    get_emergency_vehicles,
    get_vip_vehicles
)

from src.emergency.priority import (
    give_emergency_priority,
    give_vip_priority
)


SUMO_CONFIG = "sumo-rl/sumo_rl/nets/RESCO/grid4x4/grid4x4.sumocfg"


traci.start([
    "sumo",
    "-c",
    SUMO_CONFIG
])

print("SUMO started.")

for step in range(50):

    traci.simulationStep()

    emergency_vehicles = get_emergency_vehicles()
    vip_vehicles = get_vip_vehicles()

    if emergency_vehicles:

        print()
        print("Emergency vehicles detected:")
        print(emergency_vehicles)

        for vehicle_id in emergency_vehicles:

            result = give_emergency_priority(
                vehicle_id
            )

            print(
                "Emergency priority:",
                result
            )

    if vip_vehicles:

        print()
        print("VIP vehicles detected:")
        print(vip_vehicles)

        for vehicle_id in vip_vehicles:

            result = give_vip_priority(
                vehicle_id
            )

            print(
                "VIP priority:",
                result
            )

    if step % 10 == 0:

        print(
            f"Step {step}: "
            f"Emergency={emergency_vehicles}, "
            f"VIP={vip_vehicles}"
        )


traci.close()

print()
print("Emergency + VIP priority test finished.")