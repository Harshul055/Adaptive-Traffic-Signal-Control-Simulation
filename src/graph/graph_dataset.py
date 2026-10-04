import numpy as np


# ==================================================
# CREATE NODE FEATURES
# ==================================================

def create_node_features(
    local_states
):

    nodes = sorted(
        local_states.keys()
    )

    features = []

    for node in nodes:

        state = local_states[node]

        node_features = [

            state.get(
                "vehicle_count",
                0
            ),

            state.get(
                "queue_length",
                0
            ),

            state.get(
                "waiting_time",
                0
            ),

            state.get(
                "average_speed",
                0
            ),

            state.get(
                "current_phase",
                0
            ),

            state.get(
                "phase_duration",
                0
            )
        ]

        features.append(
            node_features
        )

    return (
        nodes,
        np.array(
            features,
            dtype=np.float32
        )
    )


# ==================================================
# CREATE GRAPH DATASET
# ==================================================

def create_graph_dataset(
    local_states
):

    nodes, features = (
        create_node_features(
            local_states
        )
    )

    return {
        "nodes": nodes,
        "features": features
    }


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    test_states = {

        "A0": {
            "vehicle_count": 10,
            "queue_length": 2,
            "waiting_time": 20,
            "average_speed": 8,
            "current_phase": 0,
            "phase_duration": 30
        },

        "A1": {
            "vehicle_count": 15,
            "queue_length": 4,
            "waiting_time": 30,
            "average_speed": 7,
            "current_phase": 1,
            "phase_duration": 30
        }
    }

    nodes, features = (
        create_node_features(
            test_states
        )
    )

    print()
    print("=" * 50)
    print("GRAPH DATASET TEST")
    print("=" * 50)

    print(
        f"Nodes: {nodes}"
    )

    print(
        f"Feature shape: {features.shape}"
    )

    print()
    print("Feature matrix:")

    print(features)

    print("=" * 50)