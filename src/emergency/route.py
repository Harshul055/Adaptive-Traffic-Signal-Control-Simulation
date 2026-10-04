import traci


def get_vehicle_route(vehicle_id):
    """
    Get the complete route of a vehicle.
    """

    if vehicle_id not in traci.vehicle.getIDList():
        return []

    return list(
        traci.vehicle.getRoute(vehicle_id)
    )


def get_current_road(vehicle_id):
    """
    Get the road currently occupied by the vehicle.
    """

    if vehicle_id not in traci.vehicle.getIDList():
        return None

    return traci.vehicle.getRoadID(
        vehicle_id
    )


def get_next_road(vehicle_id):
    """
    Get the next road in the vehicle's route.
    """

    route = get_vehicle_route(
        vehicle_id
    )

    if not route:
        return None

    current_road = get_current_road(
        vehicle_id
    )

    if current_road not in route:
        return None

    current_index = route.index(
        current_road
    )

    if current_index + 1 >= len(route):
        return None

    return route[
        current_index + 1
    ]


def get_emergency_route_info(vehicle_id):
    """
    Get route information for an emergency vehicle.
    """

    return {
        "vehicle_id": vehicle_id,
        "current_road": get_current_road(
            vehicle_id
        ),
        "next_road": get_next_road(
            vehicle_id
        ),
        "route": get_vehicle_route(
            vehicle_id
        )
    }