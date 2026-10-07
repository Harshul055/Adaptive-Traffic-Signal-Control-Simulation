import random


SPECIAL_ROUTE = "left0A0 A0A1 A1A2 A2A3 A3top0"


def generate_special_vehicles(
    simulation_number,
    total_emergency=5,
    total_vip=5
):
    """
    Generate the requested number of emergency and VIP vehicles.
    All special vehicles depart strictly before simulation step 1000.
    """

    random.seed(simulation_number)

    vehicles = []

    # -------------------------
    # Emergency vehicles
    # -------------------------
    for i in range(1, total_emergency + 1):

        vehicles.append({
            "id": f"emergency_{i}",
            "type": "emergency",
            "depart": random.randint(5, 999),
            "route": SPECIAL_ROUTE
        })

    # -------------------------
    # VIP vehicles
    # -------------------------
    for i in range(1, total_vip + 1):

        vehicles.append({
            "id": f"vip_{i}",
            "type": "vip",
            "depart": random.randint(5, 1000),
            "route": SPECIAL_ROUTE
        })

    # Sort by departure time
    vehicles.sort(
        key=lambda vehicle: vehicle["depart"]
    )

    return vehicles