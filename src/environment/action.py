import traci


# ==================================================
# GET TRAFFIC LIGHTS
# ==================================================

def get_traffic_lights():

    return traci.trafficlight.getIDList()


# ==================================================
# GET CURRENT PHASE
# ==================================================

def get_current_phase(tls_id):

    return traci.trafficlight.getPhase(
        tls_id
    )


# ==================================================
# GET NUMBER OF PHASES
# ==================================================

def get_phase_count(tls_id):

    logic = traci.trafficlight.getAllProgramLogics(
        tls_id
    )

    if len(logic) == 0:
        return 0

    phases = logic[0].getPhases()

    return len(phases)


# ==================================================
# SET PHASE
# ==================================================

def set_phase(
    tls_id,
    phase
):

    traci.trafficlight.setPhase(
        tls_id,
        phase
    )


# ==================================================
# APPLY ACTION
# ==================================================

def apply_action(
    tls_id,
    action,
    min_phase_duration=10
):

    """
    Actions:

    0 = Keep current phase
    1 = Move to next phase
    2 = Move to previous phase
    3 = Skip one phase
    """

    current_phase = get_current_phase(
        tls_id
    )

    phase_count = get_phase_count(
        tls_id
    )

    if phase_count == 0:

        return current_phase

    phase_duration = (
        traci.trafficlight.getPhaseDuration(
            tls_id
        )
    )

    # ----------------------------------------------
    # Minimum phase duration protection
    # ----------------------------------------------

    if phase_duration < min_phase_duration:

        return current_phase

    # ----------------------------------------------
    # Action 0
    # Keep current phase
    # ----------------------------------------------

    if action == 0:

        return current_phase

    # ----------------------------------------------
    # Action 1
    # Next phase
    # ----------------------------------------------

    elif action == 1:

        next_phase = (
            current_phase + 1
        ) % phase_count

        set_phase(
            tls_id,
            next_phase
        )

        return next_phase

    # ----------------------------------------------
    # Action 2
    # Previous phase
    # ----------------------------------------------

    elif action == 2:

        previous_phase = (
            current_phase - 1
        ) % phase_count

        set_phase(
            tls_id,
            previous_phase
        )

        return previous_phase

    # ----------------------------------------------
    # Action 3
    # Skip one phase
    # ----------------------------------------------

    elif action == 3:

        skip_phase = (
            current_phase + 2
        ) % phase_count

        set_phase(
            tls_id,
            skip_phase
        )

        return skip_phase

    # ----------------------------------------------
    # Invalid action
    # ----------------------------------------------

    return current_phase
def change_phase(tls_id):
    """
    Change the traffic light to the next phase.
    """

    import traci

    if tls_id not in traci.trafficlight.getIDList():
        return False

    current_phase = traci.trafficlight.getPhase(tls_id)
    logics = traci.trafficlight.getAllProgramLogics(tls_id)
    if not logics:
        return False

    phase_count = len(logics[0].getPhases())
    if phase_count == 0:
        return False

    next_phase = (current_phase + 1) % phase_count

    traci.trafficlight.setPhase(tls_id, next_phase)

    return True