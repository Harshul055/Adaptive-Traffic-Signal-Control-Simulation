import traci

from src.emergency.priority import (
    get_vehicle_priority,
    compare_vehicle_priority,
    get_highest_priority_vehicle
)

SUMO_CONFIG = "sumo-rl/sumo_rl/nets/RESCO/grid4x4/grid4x4.sumocfg"

traci.start(["sumo", "-c", SUMO_CONFIG])

print("SUMO started.")

for step in range(15):
    traci.simulationStep()

    vehicles = traci.vehicle.getIDList()

    if vehicles:
        emergency = [
            v for v in vehicles
            if traci.vehicle.getTypeID(v).lower() == "emergency"
        ]

        vip = [
            v for v in vehicles
            if traci.vehicle.getTypeID(v).lower() == "vip"
        ]

        if emergency:
            print("Emergency:", emergency[0],
                  "Priority:", get_vehicle_priority(emergency[0]))

        if vip:
            print("VIP:", vip[0],
                  "Priority:", get_vehicle_priority(vip[0]))

        special_vehicles = emergency + vip

        if len(special_vehicles) >= 2:
            winner = get_highest_priority_vehicle(special_vehicles)

            print("Highest priority vehicle:", winner)

            comparison = compare_vehicle_priority(
                emergency[0],
                vip[0]
            )

            print("Emergency vs VIP:", comparison)

traci.close()

print("Priority hierarchy test finished.")