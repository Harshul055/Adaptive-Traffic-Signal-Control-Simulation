import traci


def get_traffic_state():
    """Collect global traffic information from SUMO."""

    state = {}

    vehicle_ids = traci.vehicle.getIDList()

    state["vehicle_count"] = len(vehicle_ids)

    total_waiting_time = 0
    total_speed = 0

    for vehicle_id in vehicle_ids:
        total_waiting_time += (
            traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
        )

        total_speed += traci.vehicle.getSpeed(vehicle_id)

    state["total_waiting_time"] = total_waiting_time

    if len(vehicle_ids) > 0:
        state["average_speed"] = (
            total_speed / len(vehicle_ids)
        )
    else:
        state["average_speed"] = 0

    total_queue = 0

    for vehicle_id in vehicle_ids:

        speed = traci.vehicle.getSpeed(vehicle_id)

        if speed < 0.1:
            total_queue += 1

    state["queue_length"] = total_queue

    traffic_lights = traci.trafficlight.getIDList()

    state["traffic_lights"] = {}

    for tls_id in traffic_lights:

        state["traffic_lights"][tls_id] = {

            "phase": traci.trafficlight.getPhase(
                tls_id
            ),

            "phase_duration": traci.trafficlight.getPhaseDuration(
                tls_id
            )
        }

    return state


# ==========================================================
# LOCAL INTERSECTION STATE
# ==========================================================

def get_local_traffic_state(tls_id):
    """Collect traffic information around one intersection."""

    controlled_links = traci.trafficlight.getControlledLinks(
        tls_id
    )

    lane_ids = set()

    for link_group in controlled_links:

        if link_group is None:
            continue

        for link in link_group:

            if link is None:
                continue

            incoming_lane = link[0]

            if incoming_lane:
                lane_ids.add(incoming_lane)

    total_queue = 0
    total_waiting = 0
    total_vehicles = 0
    total_speed = 0

    for lane_id in lane_ids:

        try:

            vehicle_ids = traci.lane.getLastStepVehicleIDs(
                lane_id
            )

            total_vehicles += len(vehicle_ids)

            total_queue += traci.lane.getLastStepHaltingNumber(
                lane_id
            )

            total_waiting += traci.lane.getWaitingTime(
                lane_id
            )

            total_speed += (
                traci.lane.getLastStepMeanSpeed(
                    lane_id
                )
            )

        except traci.TraCIException:
            continue

    if len(lane_ids) > 0:

        average_speed = (
            total_speed / len(lane_ids)
        )

    else:

        average_speed = 0

    state = {

        "tls_id": tls_id,

        "vehicle_count": total_vehicles,

        "queue_length": total_queue,

        "waiting_time": total_waiting,

        "average_speed": average_speed,

        "current_phase": (
            traci.trafficlight.getPhase(
                tls_id
            )
        ),

        "phase_duration": (
            traci.trafficlight.getPhaseDuration(
                tls_id
            )
        ),

        "lane_count": len(lane_ids)
    }

    return state


def get_all_local_states():
    """Collect local state for all intersections."""

    states = {}

    traffic_lights = (
        traci.trafficlight.getIDList()
    )

    for tls_id in traffic_lights:

        states[tls_id] = (
            get_local_traffic_state(
                tls_id
            )
        )

    return states