from src.environment.reward import (
    calculate_all_local_rewards
)


local_states = {

    "A0": {
        "waiting_time": 20,
        "queue_length": 5,
        "average_speed": 10
    },

    "A1": {
        "waiting_time": 10,
        "queue_length": 2,
        "average_speed": 12
    },

    "A2": {
        "waiting_time": 50,
        "queue_length": 10,
        "average_speed": 5
    }
}


rewards = calculate_all_local_rewards(
    local_states
)


print("Local rewards:")

for tls_id, reward in rewards.items():

    print(
        tls_id,
        "→",
        reward
    )