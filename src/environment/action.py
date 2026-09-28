import traci


def get_traffic_lights():
    return traci.trafficlight.getIDList()


def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)


def get_phase_count(tls_id):
    logic = traci.trafficlight.getAllProgramLogics(tls_id)

    if len(logic) == 0:
        return 0

    phases = logic[0].getPhases()

    return len(phases)


def set_phase(tls_id, phase):
    traci.trafficlight.setPhase(
        tls_id,
        phase
    )


def apply_action(
    tls_id,
    action,
    min_phase_duration=10
):
    """
    Apply one MAPPO action to one traffic light.

    Actions:
    0 = keep current phase
    1 = next phase
    2 = previous phase
    3 = jump to next phase
    """

    current_phase = get_current_phase(tls_id)

    phase_count = get_phase_count(tls_id)

    if phase_count == 0:
        return current_phase

    # Current phase duration
    phase_duration = traci.trafficlight.getPhaseDuration(
        tls_id
    )

    # Do not change phase too frequently
    if phase_duration < min_phase_duration:
        return current_phase

    # Action 0
    if action == 0:
        return current_phase

    # Action 1
    elif action == 1:

        next_phase = (
            current_phase + 1
        ) % phase_count

        set_phase(
            tls_id,
            next_phase
        )

        return next_phase

    # Action 2
    elif action == 2:

        previous_phase = (
            current_phase - 1
        ) % phase_count

        set_phase(
            tls_id,
            previous_phase
        )

        return previous_phase

    # Action 3
    elif action == 3:

        next_phase = (
            current_phase + 1
        ) % phase_count

        set_phase(
            tls_id,
            next_phase
        )

        return next_phase

    return current_phase