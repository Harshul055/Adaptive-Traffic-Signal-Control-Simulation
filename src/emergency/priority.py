import traci


def get_controlled_traffic_lights(road_id):
    traffic_lights = []
    for tls_id in traci.trafficlight.getIDList():
        for link_group in traci.trafficlight.getControlledLinks(tls_id):
            if not link_group:
                continue
            for link in link_group:
                if not link:
                    continue
                try:
                    lane_edge = traci.lane.getEdgeID(link[0])
                except traci.TraCIException:
                    continue
                if lane_edge == road_id:
                    traffic_lights.append(tls_id)
                    break
            if tls_id in traffic_lights:
                break
    return traffic_lights


def get_lane_traffic_light(lane_id):
    for tls_id in traci.trafficlight.getIDList():
        for link_index, link_group in enumerate(
            traci.trafficlight.getControlledLinks(tls_id)
        ):
            if not link_group:
                continue
            for link in link_group:
                if link and link[0] == lane_id:
                    return tls_id, link_index
    return None, None


def get_green_phases(tls_id, signal_index):
    green_phases = []
    logic = traci.trafficlight.getAllProgramLogics(tls_id)
    if not logic:
        return green_phases

    for phase_number, phase in enumerate(logic[0].getPhases()):
        if signal_index < len(phase.state) and phase.state[signal_index].upper() == "G":
            green_phases.append(phase_number)
    return green_phases


def _is_green_phase(tls_id, phase_index):
    logic = traci.trafficlight.getAllProgramLogics(tls_id)
    if not logic:
        return False
    phases = logic[0].getPhases()
    return (
        0 <= phase_index < len(phases)
        and "G" in phases[phase_index].state.upper()
    )


def _next_green_phase(tls_id, current_phase):
    logic = traci.trafficlight.getAllProgramLogics(tls_id)
    if not logic:
        return None

    phases = logic[0].getPhases()
    if not phases:
        return None

    count = len(phases)
    for offset in range(1, count + 1):
        candidate = (current_phase + offset) % count
        if _is_green_phase(tls_id, candidate):
            return candidate
    return None


def _phase_timing(tls_id):
    total = float(traci.trafficlight.getPhaseDuration(tls_id))
    now = float(traci.simulation.getTime())
    remaining = max(0.0, float(traci.trafficlight.getNextSwitch(tls_id)) - now)
    elapsed = max(0.0, total - remaining)
    return elapsed, remaining


def _request_next_green(tls_id, minimum_green=10.0):
    """End the current green safely; SUMO runs its configured yellow phase."""
    phase = traci.trafficlight.getPhase(tls_id)
    if not _is_green_phase(tls_id, phase):
        return False

    elapsed, _ = _phase_timing(tls_id)
    remaining = max(0.0, minimum_green - elapsed)
    traci.trafficlight.setPhaseDuration(tls_id, remaining)
    return True


def select_priority_signal(vehicle_id, traffic_lights):
    if not traffic_lights or vehicle_id not in traci.vehicle.getIDList():
        return None

    vehicle_position = traci.vehicle.getPosition(vehicle_id)
    best_signal = None
    best_distance = float("inf")

    for tls_id in traffic_lights:
        signal_position = None
        for link_group in traci.trafficlight.getControlledLinks(tls_id):
            if not link_group:
                continue
            for link in link_group:
                if not link:
                    continue
                try:
                    shape = traci.lane.getShape(link[0])
                except traci.TraCIException:
                    continue
                if shape:
                    signal_position = shape[-1]
                    break
            if signal_position is not None:
                break

        if signal_position is None:
            continue

        dx = vehicle_position[0] - signal_position[0]
        dy = vehicle_position[1] - signal_position[1]
        distance = (dx * dx + dy * dy) ** 0.5

        if distance < best_distance:
            best_distance = distance
            best_signal = tls_id

    return best_signal


def get_emergency_priority(vehicle_id, road_id):
    return select_priority_signal(
        vehicle_id,
        get_controlled_traffic_lights(road_id),
    )


def _priority_result(vehicle_id, tls_id, lane_id, phase, changed, granted):
    return {
        "vehicle_id": vehicle_id,
        "tls_id": tls_id,
        "lane_id": lane_id,
        "phase": phase,
        "changed": changed,
        "priority_granted": granted,
    }


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

    if current_phase in green_phases:
        return _priority_result(
            vehicle_id, tls_id, lane_id, current_phase, False, True
        )

    # Never jump directly into an arbitrary green. Let the existing
    # program finish its current yellow/clearance phase first.
    next_green = _next_green_phase(tls_id, current_phase)
    if next_green is None or next_green not in green_phases:
        return None

    if _is_green_phase(tls_id, current_phase):
        changed = _request_next_green(tls_id)
        return _priority_result(
            vehicle_id, tls_id, lane_id, current_phase, changed, changed
        )

    # During yellow/transition, do not interrupt the safety clearance.
    return _priority_result(
        vehicle_id, tls_id, lane_id, current_phase, False, False
    )


def give_vip_priority(vehicle_id):
    if vehicle_id not in traci.vehicle.getIDList():
        return None

    from src.emergency.detector import get_emergency_vehicles

    if get_emergency_vehicles():
        return {
            "vehicle_id": vehicle_id,
            "priority_granted": False,
            "reason": "Emergency vehicle has higher priority",
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
        return _priority_result(
            vehicle_id, tls_id, lane_id, current_phase, False, True
        )

    next_green = _next_green_phase(tls_id, current_phase)
    if next_green is None or next_green not in green_phases:
        return None

    if _is_green_phase(tls_id, current_phase):
        changed = _request_next_green(tls_id)
        return _priority_result(
            vehicle_id, tls_id, lane_id, current_phase, changed, changed
        )

    return _priority_result(
        vehicle_id, tls_id, lane_id, current_phase, False, False
    )


def get_vehicle_priority(vehicle_id):
    if vehicle_id not in traci.vehicle.getIDList():
        return 0

    vehicle_type = traci.vehicle.getTypeID(vehicle_id).lower()

    if vehicle_type in {"emergency", "ambulance", "fire", "police"}:
        return 2
    if vehicle_type == "vip":
        return 1
    return 0


def compare_vehicle_priority(vehicle_id_1, vehicle_id_2):
    priority_1 = get_vehicle_priority(vehicle_id_1)
    priority_2 = get_vehicle_priority(vehicle_id_2)

    if priority_1 > priority_2:
        return vehicle_id_1
    if priority_2 > priority_1:
        return vehicle_id_2
    return None


def get_highest_priority_vehicle(vehicle_ids):
    if not vehicle_ids:
        return None

    return max(
        vehicle_ids,
        key=get_vehicle_priority,
    )


if __name__ == "__main__":
    print("Emergency and VIP priority module loaded successfully.")
