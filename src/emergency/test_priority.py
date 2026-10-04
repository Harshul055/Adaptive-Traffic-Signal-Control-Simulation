import traci

from src.emergency.detector import (
    get_emergency_vehicles,
    get_vip_vehicles
)

from src.emergency.priority import (
    get_vehicle_priority,
    get_highest_priority_vehicle
)


print("Starting priority test...")

print("Emergency vehicles:")
print(get_emergency_vehicles())

print()

print("VIP vehicles:")
print(get_vip_vehicles())

print()

all_vehicles = (
    get_emergency_vehicles()
    + get_vip_vehicles()
)

for vehicle_id in all_vehicles:

    priority = get_vehicle_priority(
        vehicle_id
    )

    print(
        "Vehicle:",
        vehicle_id,
        "Priority:",
        priority
    )

highest = get_highest_priority_vehicle()

print()
print(
    "Highest priority vehicle:",
    highest
)