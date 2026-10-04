import traci

from src.emergency.detector import (
    get_emergency_vehicles,
    get_vip_vehicles
)

from src.emergency.detector import (
    get_emergency_vehicles,
    get_vip_vehicles
)

from src.emergency.priority import (
    give_emergency_priority,
    give_vip_priority
)

SUMO_CONFIG = "sumo-rl/sumo_rl/nets/RESCO/grid4x4/grid4x4.sumocfg"

traci.start(["sumo", "-c", SUMO_CONFIG])

print("SUMO started.")

for step in range(50):

    traci.simulationStep()

    emergency_vehicles = get_emergency_vehicles()
    vip_vehicles = get_vip_vehicles()

    # Emergency gets priority first
    for vehicle_id in emergency_vehicles:

        result = give_emergency_priority(vehicle_id)

        if result:
            print(
                f"Step {step}: "
                f"EMERGENCY {vehicle_id} -> "
                f"TLS={result.get('tls_id')} "
                f"Phase={result.get('phase')} "
                f"Changed={result.get('changed')}"
            )

    # VIP gets priority only when no emergency is present
    for vehicle_id in vip_vehicles:

        result = give_vip_priority(vehicle_id)

        if result:
            print(
                f"Step {step}: "
                f"VIP {vehicle_id} -> "
                f"TLS={result.get('tls_id')} "
                f"Phase={result.get('phase')} "
                f"Changed={result.get('changed')} "
                f"Granted={result.get('priority_granted')}"
            )

traci.close()

print()
print("Signal priority test finished.")