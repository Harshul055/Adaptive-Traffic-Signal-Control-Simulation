"""Traffic-control reward functions.

The same reward definition is used for local intersection states and
the global environment state so environment steps do not silently
produce zero rewards.
"""

def calculate_local_reward(state):
    """Calculate reward for one intersection."""
    waiting_time = float(state.get("waiting_time", 0.0))
    queue_length = float(state.get("queue_length", 0.0))
    average_speed = float(state.get("average_speed", 0.0))
    return float(-0.01 * waiting_time - 0.5 * queue_length + 0.1 * average_speed)


def calculate_all_local_rewards(local_states):
    """Return one reward per traffic-light intersection."""
    return {tls_id: calculate_local_reward(state) for tls_id, state in local_states.items()}


def calculate_reward(state):
    """Calculate reward from global, local, or per-intersection state."""
    if not state:
        return 0.0

    if "total_waiting_time" in state:
        return calculate_local_reward({
            "waiting_time": state.get("total_waiting_time", 0.0),
            "queue_length": state.get("queue_length", 0.0),
            "average_speed": state.get("average_speed", 0.0),
        })

    if "waiting_time" in state or "queue_length" in state:
        return calculate_local_reward(state)

    if all(isinstance(value, dict) for value in state.values()):
        return float(sum(calculate_local_reward(value) for value in state.values()))

    return 0.0


if __name__ == "__main__":
    test_state = {"total_waiting_time": 50, "average_speed": 10, "queue_length": 5}
    print(f"Test reward: {calculate_reward(test_state):.2f}")
