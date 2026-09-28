def calculate_local_reward(state):
    """
    Calculate reward for one intersection.
    """

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

    # Penalties
    waiting_penalty = (
        -0.01 * waiting_time
    )

    queue_penalty = (
        -0.5 * queue_length
    )

    # Speed reward
    speed_reward = (
        0.1 * average_speed
    )

    reward = (
        waiting_penalty
        + queue_penalty
        + speed_reward
    )

    return float(reward)


def calculate_all_local_rewards(
    local_states
):
    """
    Calculate one reward for every
    traffic-light agent.
    """

    rewards = {}

    for tls_id, state in local_states.items():

        rewards[tls_id] = calculate_local_reward(
            state
        )

    return rewards