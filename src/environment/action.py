"""Safe RL traffic-light actions for the 4x4 SUMO signal programs.

The network alternates 10-second green phases with 3-second yellow
transition phases. The RL action therefore changes green duration rather
than directly jumping between arbitrary phase indices.
"""

import traci


EXTEND_SMALL = 5.0
EXTEND_LARGE = 10.0
DEFAULT_MAX_GREEN = 60.0


def get_traffic_lights():
    return traci.trafficlight.getIDList()


def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)


def get_phase_count(tls_id):
    logic = traci.trafficlight.getAllProgramLogics(tls_id)
    if not logic:
        return 0
    return len(logic[0].getPhases())


def get_phase_state(tls_id, phase_index=None):
    """Return the signal-state string for a phase."""
    if phase_index is None:
        phase_index = get_current_phase(tls_id)

    logic = traci.trafficlight.getAllProgramLogics(tls_id)
    if not logic:
        return ""

    phases = logic[0].getPhases()
    if phase_index < 0 or phase_index >= len(phases):
        return ""

    return phases[phase_index].state


def is_green_phase(tls_id, phase_index=None):
    state = get_phase_state(tls_id, phase_index)
    return "G" in state


def get_phase_timing(tls_id):
    """Return (configured duration, elapsed, remaining) for current phase."""
    total = float(traci.trafficlight.getPhaseDuration(tls_id))
    now = float(traci.simulation.getTime())
    next_switch = float(traci.trafficlight.getNextSwitch(tls_id))
    remaining = max(0.0, next_switch - now)
    elapsed = max(0.0, total - remaining)
    return total, elapsed, remaining


def set_phase_duration(tls_id, remaining_seconds):
    traci.trafficlight.setPhaseDuration(
        tls_id,
        max(0.0, float(remaining_seconds))
    )


def apply_action(
    tls_id,
    action,
    min_phase_duration=10.0,
    max_phase_duration=DEFAULT_MAX_GREEN,
):
    """Apply one safe signal-control action.

    Action 0: hold the current phase.
    Action 1: extend the current green by 5 s.
    Action 2: terminate the current green as soon as the minimum green is met.
    Action 3: extend the current green by 10 s.

    Yellow phases are never selected directly by the RL policy. SUMO's
    existing yellow transition is allowed to run normally.
    """
    if tls_id not in traci.trafficlight.getIDList():
        return None

    action = int(action)
    current_phase = get_current_phase(tls_id)

    if not is_green_phase(tls_id, current_phase):
        return current_phase

    _, elapsed, remaining = get_phase_timing(tls_id)
    minimum = max(0.0, float(min_phase_duration))
    maximum = max(minimum, float(max_phase_duration))

    if action == 0:
        return current_phase

    if action == 1:
        target_remaining = min(
            remaining + EXTEND_SMALL,
            max(0.0, maximum - elapsed),
        )
        set_phase_duration(tls_id, target_remaining)
        return current_phase

    if action == 2:
        if elapsed < minimum:
            set_phase_duration(tls_id, minimum - elapsed)
        else:
            set_phase_duration(tls_id, 0.0)
        return current_phase

    if action == 3:
        target_remaining = min(
            remaining + EXTEND_LARGE,
            max(0.0, maximum - elapsed),
        )
        set_phase_duration(tls_id, target_remaining)
        return current_phase

    return current_phase


def change_phase(tls_id):
    """Request a normal transition to the next phase without skipping yellow."""
    if tls_id not in traci.trafficlight.getIDList():
        return False

    current_phase = get_current_phase(tls_id)

    if not is_green_phase(tls_id, current_phase):
        return False

    _, elapsed, _ = get_phase_timing(tls_id)
    if elapsed < 10.0:
        set_phase_duration(tls_id, 10.0 - elapsed)
    else:
        set_phase_duration(tls_id, 0.0)

    return True
