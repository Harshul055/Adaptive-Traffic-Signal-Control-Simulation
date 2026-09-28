import os
import sys
import torch
import traci

# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

from src.graph.graph_builder import (
    build_graph,
    create_adjacency_matrix
)

from src.graph.graph_dataset import (
    create_node_features
)

from src.graph.gnn import TrafficGNN

from src.agents.mappo import MAPPO

from src.environment.state import (
    get_all_local_states
)

from src.environment.reward import (
    calculate_all_local_rewards
)

from src.environment.action import (
    apply_action
)


# --------------------------------------------------
# SUMO SETTINGS
# --------------------------------------------------

SUMO_HOME = (
    r"C:\Program Files (x86)\Eclipse\Sumo"
)

SUMO_BINARY = os.path.join(
    SUMO_HOME,
    "bin",
    "sumo.exe"
)

SUMO_CONFIG = (
    r"C:\Project\Adaptive-Traffic-Control"
    r"\sumo-rl\sumo_rl\nets\RESCO\grid4x4"
    r"\grid4x4.sumocfg"
)


# --------------------------------------------------
# TRAINING SETTINGS
# --------------------------------------------------

NUM_EPISODES = 5

MAX_STEPS = 500

NUM_AGENTS = 16

STATE_SIZE = 32

ACTION_SIZE = 4

MIN_PHASE_DURATION = 10


# --------------------------------------------------
# TRAINING FUNCTION
# --------------------------------------------------

def train():

    # ==================================================
    # 1. BUILD TRAFFIC GRAPH
    # ==================================================

    graph = build_graph()

    traffic_lights, adjacency = (
        create_adjacency_matrix(graph)
    )

    adjacency = torch.tensor(
        adjacency,
        dtype=torch.float32
    )

    print(
        "Traffic lights found:",
        len(traffic_lights)
    )


    # ==================================================
    # 2. CREATE GNN
    # ==================================================

    gnn = TrafficGNN(
        input_features=6,
        hidden_features=64,
        output_features=32
    )

    print(
        "GNN input features: 6"
    )

    print(
        "GNN output features: 32"
    )


    # ==================================================
    # 3. CREATE 16 MAPPO AGENTS
    # ==================================================

    agents = []

    for _ in range(NUM_AGENTS):

        agent = MAPPO(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            hidden_size=64,
            learning_rate=0.0003,
            gamma=0.99,
            clip_epsilon=0.2
        )

        agents.append(agent)


    print()
    print("======================================")
    print("GNN + MAPPO TRAINING")
    print("======================================")

    print(
        "Agents:",
        len(agents)
    )

    print()


    # ==================================================
    # 4. TRAINING EPISODES
    # ==================================================

    for episode in range(NUM_EPISODES):

        print(
            f"Episode {episode + 1}/{NUM_EPISODES}"
        )


        # ==================================================
        # START SUMO
        # ==================================================

        try:
            traci.close()
        except:
            pass


        traci.start([
            SUMO_BINARY,
            "-c",
            SUMO_CONFIG
        ])


        # ==================================================
        # ROLLOUT STORAGE
        # ==================================================

        agent_states = [
            [] for _ in range(NUM_AGENTS)
        ]

        agent_actions = [
            [] for _ in range(NUM_AGENTS)
        ]

        agent_log_probs = [
            [] for _ in range(NUM_AGENTS)
        ]

        agent_values = [
            [] for _ in range(NUM_AGENTS)
        ]

        agent_rewards = [
            [] for _ in range(NUM_AGENTS)
        ]


        episode_reward = 0.0


        # ==================================================
        # SIMULATION LOOP
        # ==================================================

        for step in range(MAX_STEPS):


            # --------------------------------------------------
            # 1. ADVANCE SUMO
            # --------------------------------------------------

            traci.simulationStep()


            # --------------------------------------------------
            # 2. GET LOCAL STATES
            # --------------------------------------------------

            local_states = (
                get_all_local_states()
            )


            # --------------------------------------------------
            # 3. CREATE NODE FEATURES
            # --------------------------------------------------

            feature_nodes, features = (
                create_node_features(
                    local_states
                )
            )


            x = torch.tensor(
                features,
                dtype=torch.float32
            )


            # --------------------------------------------------
            # 4. GNN
            # --------------------------------------------------

            with torch.no_grad():

                gnn_output = gnn(
                    x,
                    adjacency
                )


            # --------------------------------------------------
            # 5. MAPPO ACTIONS
            # --------------------------------------------------

            actions = []

            log_probs = []

            values = []


            for agent_index in range(
                NUM_AGENTS
            ):

                agent = agents[
                    agent_index
                ]


                # GNN embedding for this
                # intersection

                agent_state = (
                    gnn_output[
                        agent_index
                    ]
                )


                # MAPPO action

                action, log_prob, value = (
                    agent.select_action(
                        agent_state
                    )
                )


                actions.append(
                    action
                )

                log_probs.append(
                    log_prob
                )

                values.append(
                    value
                )


            # --------------------------------------------------
            # 6. APPLY ACTIONS TO SUMO
            # --------------------------------------------------

            for agent_index in range(
                NUM_AGENTS
            ):

                tls_id = (
                    traffic_lights[
                        agent_index
                    ]
                )

                apply_action(
                    tls_id,
                    actions[
                        agent_index
                    ],
                    min_phase_duration=(
                        MIN_PHASE_DURATION
                    )
                )


            # --------------------------------------------------
            # 7. ADVANCE SUMO AFTER ACTION
            # --------------------------------------------------

            traci.simulationStep()


            # --------------------------------------------------
            # 8. GET NEW LOCAL STATES
            # --------------------------------------------------

            next_local_states = (
                get_all_local_states()
            )


            # --------------------------------------------------
            # 9. CALCULATE LOCAL REWARDS
            # --------------------------------------------------

            reward_dict = (
                calculate_all_local_rewards(
                    next_local_states
                )
            )


            # --------------------------------------------------
            # 10. STORE EXPERIENCE
            # --------------------------------------------------

            for agent_index in range(
                NUM_AGENTS
            ):

                tls_id = (
                    traffic_lights[
                        agent_index
                    ]
                )


                reward = reward_dict[
                    tls_id
                ]


                # State

                agent_states[
                    agent_index
                ].append(
                    gnn_output[
                        agent_index
                    ].detach()
                )


                # Action

                agent_actions[
                    agent_index
                ].append(
                    actions[
                        agent_index
                    ]
                )


                # Log probability

                agent_log_probs[
                    agent_index
                ].append(
                    log_probs[
                        agent_index
                    ]
                )


                # Value

                agent_values[
                    agent_index
                ].append(
                    values[
                        agent_index
                    ]
                )


                # Local reward

                agent_rewards[
                    agent_index
                ].append(
                    reward
                )


                episode_reward += reward


            # --------------------------------------------------
            # 11. DISPLAY PROGRESS
            # --------------------------------------------------

            if step % 100 == 0:

                total_vehicles = sum(
                state["vehicle_count"]
                for state in next_local_states.values()
                )

                print(
                    f"Step {step}: "
                    f"Vehicles={total_vehicles}"
                )


        # ==================================================
        # 5. UPDATE MAPPO AGENTS
        # ==================================================

        print()
        print(
            "Updating MAPPO agents..."
        )


        for agent_index in range(
            NUM_AGENTS
        ):

            agent = agents[
                agent_index
            ]


            actor_loss, critic_loss = (
                agent.update(
                    agent_states[
                        agent_index
                    ],

                    agent_actions[
                        agent_index
                    ],

                    agent_log_probs[
                        agent_index
                    ],

                    agent_rewards[
                        agent_index
                    ],

                    agent_values[
                        agent_index
                    ]
                )
            )


            print(
                f"Agent {agent_index + 1}: "
                f"Actor Loss={actor_loss:.4f}, "
                f"Critic Loss={critic_loss:.4f}"
            )


        # ==================================================
        # 6. EPISODE RESULT
        # ==================================================

        print()

        print(
            f"Episode reward: "
            f"{episode_reward:.2f}"
        )

        print()


        # ==================================================
        # CLOSE SUMO
        # ==================================================

        try:
            traci.close()
        except:
            pass


    # ==================================================
    # TRAINING COMPLETE
    # ==================================================

    print()
    print("======================================")
    print("MAPPO TRAINING COMPLETED")
    print("======================================")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    train()