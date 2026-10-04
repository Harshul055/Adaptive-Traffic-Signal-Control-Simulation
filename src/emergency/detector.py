import traci


def get_emergency_vehicles():
    """
    Detect emergency vehicles currently present in SUMO.
    """

    emergency_vehicles = []

    vehicle_ids = traci.vehicle.getIDList()

    for vehicle_id in vehicle_ids:

        vehicle_type = traci.vehicle.getTypeID(
            vehicle_id
        )

        if vehicle_type.lower() in [
            "emergency",
            "ambulance",
            "fire",
            "police"
        ]:
            emergency_vehicles.append(
                vehicle_id
            )

    return emergency_vehicles


def is_emergency_vehicle(vehicle_id):
    """
    Check whether a specific vehicle is an emergency vehicle.
    """

    if vehicle_id not in traci.vehicle.getIDList():
        return False

    vehicle_type = traci.vehicle.getTypeID(
        vehicle_id
    )

    return vehicle_type.lower() in [
        "emergency",
        "ambulance",
        "fire",
        "police"
    ]


def get_emergency_vehicle_info():
    """
    Return information about detected emergency vehicles.
    """

    emergency_vehicles = []

    for vehicle_id in get_emergency_vehicles():

        emergency_vehicles.append({
            "vehicle_id": vehicle_id,
            "type": traci.vehicle.getTypeID(
                vehicle_id
            ),
            "lane": traci.vehicle.getLaneID(
                vehicle_id
            ),
            "road": traci.vehicle.getRoadID(
                vehicle_id
            ),
            "position": traci.vehicle.getLanePosition(
                vehicle_id
            ),
            "speed": traci.vehicle.getSpeed(
                vehicle_id
            )
        })

    return emergency_vehicles
def get_vip_vehicles():
    """
    Detect VIP vehicles currently present in SUMO.
    """

    vip_vehicles = []

    vehicle_ids = traci.vehicle.getIDList()

    for vehicle_id in vehicle_ids:

        vehicle_type = traci.vehicle.getTypeID(
            vehicle_id
        )

        if vehicle_type.lower() == "vip":
            vip_vehicles.append(
                vehicle_id
            )

    return vip_vehicles


def is_vip_vehicle(vehicle_id):
    """
    Check whether a specific vehicle is a VIP vehicle.
    """

    if vehicle_id not in traci.vehicle.getIDList():
        return False

    vehicle_type = traci.vehicle.getTypeID(
        vehicle_id
    )

    return vehicle_type.lower() == "vip"