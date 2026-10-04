# ==================================================
# CALCULATE REWARD FOR ONE TRAFFIC LIGHT
# ==================================================

def calculate_local_reward(state):

    waiting_time = state.get(
        "waiting_time",
        0
    )

    queue_length = state.get(
        "queue_length",
        0
    )

    average_speed = state.get(
        "average_speed",
        0
    )

    # ----------------------------------------------
    # Waiting time penalty
    # ----------------------------------------------

    waiting_penalty = (
        -0.01 * waiting_time
    )

    # ----------------------------------------------
    # Queue length penalty
    # ----------------------------------------------

    queue_penalty = (
        -0.5 * queue_length
    )

    # ----------------------------------------------
    # Speed reward
    # ----------------------------------------------

    speed_reward = (
        0.1 * average_speed
    )

    # ----------------------------------------------
    # Total reward
    # ----------------------------------------------

    reward = (
        waiting_penalty
        + queue_penalty
        + speed_reward
    )

    return float(reward)


# ==================================================
# CALCULATE REWARD FOR ALL TRAFFIC LIGHTS
# ==================================================

def calculate_all_local_rewards(
    local_states
):

    rewards = {}

    for tls_id, state in local_states.items():

        rewards[tls_id] = (
            calculate_local_reward(
                state
            )
        )

    return rewards


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    test_state = {
        "vehicle_count": 20,
        "waiting_time": 50,
        "average_speed": 10,
        "queue_length": 5
    }

    reward = calculate_local_reward(
        test_state
    )

    print(
        f"Test reward: {reward:.2f}"
    )
def calculate_reward(state):
    """
    Calculate reward from the current traffic state.
    Lower waiting time and queue length gives a better reward.
    """

    if not state:
        return 0.0

    total_waiting = 0.0
    total_queue = 0.0

    for tls_id, data in state.items():
        if isinstance(data, dict):
            total_waiting += data.get("waiting_time", 0.0)
            total_queue += data.get("queue_length", 0.0)

    reward = -(total_waiting + total_queue)

    return reward