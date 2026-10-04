import traci

from src.emergency.detector import (
    get_emergency_vehicles,
    get_vip_vehicles
)


# ==================================================
# FIND TRAFFIC LIGHTS CONTROLLING A ROAD
# ==================================================

def get_controlled_traffic_lights(road_id):

    traffic_lights = []

    for tls_id in traci.trafficlight.getIDList():

        controlled_links = traci.trafficlight.getControlledLinks(
            tls_id
        )

        for link_group in controlled_links:

            if not link_group:
                continue

            for link in link_group:

                if len(link) < 1:
                    continue

                incoming_lane = link[0]

                try:
                    lane_edge = traci.lane.getEdgeID(
                        incoming_lane
                    )
                except Exception:
                    continue

                if lane_edge == road_id:

                    if tls_id not in traffic_lights:
                        traffic_lights.append(tls_id)

                    break

            if tls_id in traffic_lights:
                break

    return traffic_lights


# ==================================================
# FIND TRAFFIC LIGHT CONTROLLING A LANE
# ==================================================

def get_lane_traffic_light(lane_id):

    for tls_id in traci.trafficlight.getIDList():

        controlled_links = traci.trafficlight.getControlledLinks(
            tls_id
        )

        for link_index, link_group in enumerate(controlled_links):

            if not link_group:
                continue

            for link in link_group:

                if len(link) < 1:
                    continue

                incoming_lane = link[0]

                if incoming_lane == lane_id:

                    return tls_id, link_index

    return None, None


# ==================================================
# FIND GREEN PHASES FOR A SIGNAL
# ==================================================

def get_green_phases(tls_id, signal_index):

    green_phases = []

    logic = traci.trafficlight.getAllProgramLogics(
        tls_id
    )

    if not logic:
        return green_phases

    phases = logic[0].getPhases()

    for phase_number, phase in enumerate(phases):

        if signal_index >= len(phase.state):
            continue

        signal_state = phase.state[signal_index].upper()

        if signal_state == "G":
            green_phases.append(phase_number)

    return green_phases


# ==================================================
# FIND CLOSEST TRAFFIC LIGHT
# ==================================================

def select_priority_signal(
    vehicle_id,
    traffic_lights
):

    if not traffic_lights:
        return None

    if vehicle_id not in traci.vehicle.getIDList():
        return None

    vehicle_position = traci.vehicle.getPosition(
        vehicle_id
    )

    best_signal = None
    best_distance = float("inf")

    for tls_id in traffic_lights:

        controlled_links = traci.trafficlight.getControlledLinks(
            tls_id
        )

        signal_position = None

        for link_group in controlled_links:

            if not link_group:
                continue

            for link in link_group:

                if len(link) < 1:
                    continue

                incoming_lane = link[0]

                try:
                    lane_shape = traci.lane.getShape(
                        incoming_lane
                    )
                except Exception:
                    continue

                if not lane_shape:
                    continue

                signal_position = lane_shape[-1]

                break

            if signal_position is not None:
                break

        if signal_position is None:
            continue

        dx = (
            vehicle_position[0]
            - signal_position[0]
        )

        dy = (
            vehicle_position[1]
            - signal_position[1]
        )

        distance = (
            dx ** 2 + dy ** 2
        ) ** 0.5

        if distance < best_distance:

            best_distance = distance
            best_signal = tls_id

    return best_signal


# ==================================================
# GET EMERGENCY PRIORITY SIGNAL
# ==================================================

def get_emergency_priority(
    vehicle_id,
    road_id
):

    traffic_lights = get_controlled_traffic_lights(
        road_id
    )

    priority_signal = select_priority_signal(
        vehicle_id,
        traffic_lights
    )

    return priority_signal


# ==================================================
# GIVE EMERGENCY VEHICLE PRIORITY
# ==================================================

def give_emergency_priority(vehicle_id):
    if vehicle_id not in traci.vehicle.getIDList():
        return None

    lane_id = traci.vehicle.getLaneID(vehicle_id)

    if not lane_id:
        return None

    tls_id, signal_index = get_lane_traffic_light(lane_id)

    if tls_id is None:
        return None

    green_phases = get_green_phases(tls_id, signal_index)

    if not green_phases:
        return None

    current_phase = traci.trafficlight.getPhase(tls_id)

    # Already in a suitable green phase
    if current_phase in green_phases:
        return {
            "vehicle_id": vehicle_id,
            "tls_id": tls_id,
            "lane_id": lane_id,
            "phase": current_phase,
            "changed": False
        }

    # Select the first suitable green phase
    target_phase = green_phases[0]

    # Do not reset the signal if it is already on the target phase
    if current_phase != target_phase:
        traci.trafficlight.setPhase(tls_id, target_phase)

        return {
            "vehicle_id": vehicle_id,
            "tls_id": tls_id,
            "lane_id": lane_id,
            "phase": target_phase,
            "changed": True
        }

    return {
        "vehicle_id": vehicle_id,
        "tls_id": tls_id,
        "lane_id": lane_id,
        "phase": current_phase,
        "changed": False
    }



# ==================================================
# GIVE VIP VEHICLE PRIORITY
# ==================================================

def give_vip_priority(vehicle_id):
    if vehicle_id not in traci.vehicle.getIDList():
        return None

    # Emergency vehicles always have higher priority than VIP
    emergency_vehicles = get_emergency_vehicles()

    if emergency_vehicles:
        return {
            "vehicle_id": vehicle_id,
            "priority_granted": False,
            "reason": "Emergency vehicle has higher priority"
        }

    lane_id = traci.vehicle.getLaneID(vehicle_id)

    if not lane_id:
        return None

    tls_id, signal_index = get_lane_traffic_light(lane_id)

    if tls_id is None:
        return None

    green_phases = get_green_phases(tls_id, signal_index)

    if not green_phases:
        return None

    current_phase = traci.trafficlight.getPhase(tls_id)

    if current_phase in green_phases:
        return {
            "vehicle_id": vehicle_id,
            "tls_id": tls_id,
            "lane_id": lane_id,
            "phase": current_phase,
            "changed": False,
            "priority_granted": True
        }

    target_phase = green_phases[0]

    traci.trafficlight.setPhase(tls_id, target_phase)

    return {
        "vehicle_id": vehicle_id,
        "tls_id": tls_id,
        "lane_id": lane_id,
        "phase": target_phase,
        "changed": True,
        "priority_granted": True
    }

def get_vehicle_priority(vehicle_id):
    """
    Returns priority level:
    2 = Emergency
    1 = VIP
    0 = Normal
    """

    if vehicle_id not in traci.vehicle.getIDList():
        return 0

    vehicle_type = traci.vehicle.getTypeID(vehicle_id).lower()

    if vehicle_type in ["emergency", "ambulance", "fire", "police"]:
        return 2

    if vehicle_type == "vip":
        return 1

    return 0


def compare_vehicle_priority(vehicle_id_1, vehicle_id_2):
    """
    Compares two vehicles.

    Returns:
        vehicle_id_1 → if vehicle 1 has higher priority
        vehicle_id_2 → if vehicle 2 has higher priority
        None          → if both have equal priority
    """

    priority_1 = get_vehicle_priority(vehicle_id_1)
    priority_2 = get_vehicle_priority(vehicle_id_2)

    if priority_1 > priority_2:
        return vehicle_id_1

    if priority_2 > priority_1:
        return vehicle_id_2

    return None


def get_highest_priority_vehicle(vehicle_ids):
    """
    Returns the highest-priority vehicle from a list.
    """

    if not vehicle_ids:
        return None

    highest_vehicle = None
    highest_priority = -1

    for vehicle_id in vehicle_ids:
        priority = get_vehicle_priority(vehicle_id)

        if priority > highest_priority:
            highest_priority = priority
            highest_vehicle = vehicle_id

    return highest_vehicle

# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    print(
        "Emergency and VIP priority module loaded successfully."
    )