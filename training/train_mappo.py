import os
import sys
import csv

import torch
import traci


# ==================================================
# PROJECT PATH
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==================================================
# IMPORTS
# ==================================================

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


# ==================================================
# EMERGENCY SYSTEM
# ==================================================

from src.emergency.detector import (
    get_emergency_vehicles
)

from src.emergency.route import (
    get_current_road,
    get_next_road
)

from src.emergency.priority import (
    get_controlled_traffic_lights,
    select_priority_signal,
    give_emergency_priority
)


# ==================================================
# SUMO SETTINGS
# ==================================================

SUMO_HOME = os.environ.get(
    "SUMO_HOME",
    r"C:\Program Files (x86)\Eclipse\Sumo"
)

SUMO_BINARY = os.path.join(
    SUMO_HOME,
    "bin",
    "sumo-gui.exe"
)

SUMO_CONFIG = os.path.join(
    PROJECT_ROOT,
    "sumo-rl", "sumo_rl", "nets", "RESCO",
    "grid4x4", "grid4x4.sumocfg"
)


# ==================================================
# RESULT SETTINGS
# ==================================================

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "mappo_final_results.csv"
)


# ==================================================
# TRAINING SETTINGS
# ==================================================

NUM_EPISODES = 2
MAX_STEPS = 500

NUM_AGENTS = 16

STATE_SIZE = 32
ACTION_SIZE = 4

MIN_PHASE_DURATION = 10


# ==================================================
# EMERGENCY VEHICLE HANDLER
# ==================================================

def handle_emergency_vehicles():

    emergency_vehicles = (
        get_emergency_vehicles()
    )

    priority_signals = {}

    for vehicle_id in emergency_vehicles:

        # ------------------------------------------
        # CURRENT ROAD
        # ------------------------------------------

        current_road = get_current_road(
            vehicle_id
        )

        # ------------------------------------------
        # NEXT ROAD
        # ------------------------------------------

        next_road = get_next_road(
            vehicle_id
        )

        if current_road is None:
            continue

        # ------------------------------------------
        # FIND TRAFFIC LIGHTS
        # ------------------------------------------

        traffic_lights = (
            get_controlled_traffic_lights(
                current_road
            )
        )

        # ------------------------------------------
        # SELECT PRIORITY SIGNAL
        # ------------------------------------------

        priority_signal = (
            select_priority_signal(
                vehicle_id,
                traffic_lights
            )
        )

        if priority_signal is not None:

            priority_signals[
                vehicle_id
            ] = {
                "current_road": current_road,
                "next_road": next_road,
                "traffic_light": priority_signal
            }

    return priority_signals


# ==================================================
# TRAINING FUNCTION
# ==================================================

def train():

    # ==================================================
    # 1. BUILD TRAFFIC GRAPH
    # ==================================================

    graph = build_graph()

    traffic_lights, adjacency = (
        create_adjacency_matrix(
            graph
        )
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


    # ==================================================
    # LOAD TRAINED GNN
    # ==================================================

    GNN_MODEL_PATH = os.path.join(
        PROJECT_ROOT,
        "models",
        "gnn",
        "traffic_gnn.pth"
    )

    if os.path.exists(
        GNN_MODEL_PATH
    ):

        gnn.load_state_dict(
            torch.load(
                GNN_MODEL_PATH,
                map_location=torch.device("cpu")
            )
        )

        print()
        print(
            "Trained GNN model loaded:"
        )
        print(
            GNN_MODEL_PATH
        )

    else:

        print()
        print(
            "WARNING: Trained GNN model not found."
        )
        print(
            GNN_MODEL_PATH
        )

    gnn.eval()

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
            clip_epsilon=0.2,
            ppo_epochs=5
        )

        agents.append(
            agent
        )


    print()
    print(
        "======================================"
    )
    print(
        "GNN + MAPPO TRAINING"
    )
    print(
        "======================================"
    )
    print(
        "Agents:",
        len(agents)
    )
    print()


    # ==================================================
    # CREATE CSV FILE
    # ==================================================

    with open(
        RESULT_FILE,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow([
            "episode",
            "step",
            "vehicle_count",
            "waiting_time",
            "average_speed",
            "queue_length",
            "reward"
        ])


        # ==================================================
        # 4. TRAINING EPISODES
        # ==================================================

        for episode in range(
            NUM_EPISODES
        ):

            print(
                f"Episode {episode + 1}/{NUM_EPISODES}"
            )


            # ==================================================
            # START SUMO-GUI
            # ==================================================

            try:
                traci.close()
            except:
                pass

            print(
                "Starting SUMO-GUI..."
            )

            traci.start([
                SUMO_BINARY,
                "-c",
                SUMO_CONFIG
            ])

            print(
                "SUMO-GUI connected successfully."
            )


            # ==================================================
            # ROLLOUT STORAGE
            # ==================================================

            agent_states = [
                [] for _ in range(NUM_AGENTS)
            ]

            agent_global_states = [
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

            agent_next_values = [
                [] for _ in range(NUM_AGENTS)
            ]

            agent_dones = [
                [] for _ in range(NUM_AGENTS)
            ]


            # ==================================================
            # EPISODE METRIC STORAGE
            # ==================================================

            episode_reward = 0.0

            episode_waiting_time = 0.0

            episode_speed = 0.0

            episode_queue = 0.0

            episode_vehicle_count = 0.0


            # ==================================================
            # SIMULATION LOOP
            # ==================================================

            for step in range(
                MAX_STEPS
            ):

                # ==================================================
                # 1. ADVANCE SUMO
                # ==================================================

                traci.simulationStep()


                # ==================================================
                # 2. DETECT EMERGENCY VEHICLES
                # ==================================================

                emergency_info = (
                    handle_emergency_vehicles()
                )


                if emergency_info:

                    print()

                    print(
                        "Emergency vehicle detected:"
                    )

                    for (
                        vehicle_id,
                        info
                    ) in emergency_info.items():

                        print(
                            f"Vehicle={vehicle_id} | "
                            f"Current Road="
                            f"{info['current_road']} | "
                            f"Next Road="
                            f"{info['next_road']} | "
                            f"Priority TLS="
                            f"{info['traffic_light']}"
                        )


                # ==================================================
                # 3. GET LOCAL STATES
                # ==================================================

                local_states = (
                    get_all_local_states()
                )


                # ==================================================
                # 4. CREATE NODE FEATURES
                # ==================================================

                feature_nodes, features = (
                    create_node_features(
                        local_states
                    )
                )

                x = torch.tensor(
                    features,
                    dtype=torch.float32
                )


                # ==================================================
                # 5. GNN
                # ==================================================

                with torch.no_grad():

                    gnn_output = gnn(
                        x,
                        adjacency
                    )


                # ==================================================
                # 6. CREATE GLOBAL STATE
                # ==================================================

                global_state = (
                    gnn_output.flatten()
                )


                # ==================================================
                # 7. MAPPO ACTIONS
                # ==================================================

                actions = []

                log_probs = []

                values = []


                for agent_index in range(
                    NUM_AGENTS
                ):

                    # ------------------------------------------
                    # LOCAL STATE FOR ACTOR
                    # ------------------------------------------

                    agent_state = (
                        gnn_output[
                            agent_index
                        ]
                    )


                    # ------------------------------------------
                    # SELECT ACTION
                    # ------------------------------------------

                    action, log_prob, value = (
                        agents[
                            agent_index
                        ].select_action(
                            agent_state,
                            global_state
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


                # ==================================================
                # 8. APPLY MAPPO ACTIONS
                # ==================================================

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


                # ==================================================
                # 9. EMERGENCY PRIORITY OVERRIDE
                # ==================================================

                emergency_vehicles = (
                    get_emergency_vehicles()
                )


                for vehicle_id in (
                    emergency_vehicles
                ):

                    priority_result = (
                        give_emergency_priority(
                            vehicle_id
                        )
                    )


                    if (
                        priority_result
                        and
                        priority_result.get(
                            "changed",
                            False
                        )
                    ):

                        print()

                        print(
                            "======================================"
                        )

                        print(
                            "EMERGENCY PRIORITY ACTIVATED"
                        )

                        print(
                            f"Vehicle: "
                            f"{vehicle_id}"
                        )

                        print(
                            f"Traffic Light: "
                            f"{priority_result['tls_id']}"
                        )

                        print(
                            f"Lane: "
                            f"{priority_result['lane_id']}"
                        )

                        print(
                            f"Green Phase: "
                            f"{priority_result['phase']}"
                        )

                        print(
                            "======================================"
                        )


                # ==================================================
                # 10. ADVANCE SUMO AFTER ACTION
                # ==================================================

                traci.simulationStep()


                # ==================================================
                # 11. GET NEW LOCAL STATES
                # ==================================================

                next_local_states = (
                    get_all_local_states()
                )


                # ==================================================
                # 12. CALCULATE LOCAL REWARDS
                # ==================================================

                reward_dict = (
                    calculate_all_local_rewards(
                        next_local_states
                    )
                )


                # ==================================================
                # 13. CHECK DONE
                # ==================================================

                done = (
                    step == MAX_STEPS - 1
                )


                # ==================================================
                # 14. CREATE NEXT GNN STATE
                # ==================================================

                next_feature_nodes, next_features = (
                    create_node_features(
                        next_local_states
                    )
                )

                next_x = torch.tensor(
                    next_features,
                    dtype=torch.float32
                )


                with torch.no_grad():

                    next_gnn_output = gnn(
                        next_x,
                        adjacency
                    )


                # ==================================================
                # 15. CREATE NEXT GLOBAL STATE
                # ==================================================

                next_global_state = (
                    next_gnn_output.flatten()
                )


                # ==================================================
                # 16. STORE EXPERIENCE
                # ==================================================

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


                    # ------------------------------------------
                    # CURRENT LOCAL STATE
                    # ------------------------------------------

                    agent_states[
                        agent_index
                    ].append(
                        gnn_output[
                            agent_index
                        ].detach()
                    )


                    # ------------------------------------------
                    # CURRENT GLOBAL STATE
                    # ------------------------------------------

                    agent_global_states[
                        agent_index
                    ].append(
                        global_state.detach()
                    )


                    # ------------------------------------------
                    # ACTION
                    # ------------------------------------------

                    agent_actions[
                        agent_index
                    ].append(
                        actions[
                            agent_index
                        ]
                    )


                    # ------------------------------------------
                    # LOG PROBABILITY
                    # ------------------------------------------

                    agent_log_probs[
                        agent_index
                    ].append(
                        log_probs[
                            agent_index
                        ]
                    )


                    # ------------------------------------------
                    # CURRENT VALUE
                    # ------------------------------------------

                    agent_values[
                        agent_index
                    ].append(
                        values[
                            agent_index
                        ]
                    )


                    # ------------------------------------------
                    # REWARD
                    # ------------------------------------------

                    agent_rewards[
                        agent_index
                    ].append(
                        reward
                    )


                    # ------------------------------------------
                    # NEXT VALUE
                    # ------------------------------------------

                    if done:

                        next_value = 0.0

                    else:

                        with torch.no_grad():

                            next_value = agents[
                                agent_index
                            ].critic(
                                next_global_state.unsqueeze(
                                    0
                                )
                            ).item()


                    agent_next_values[
                        agent_index
                    ].append(
                        next_value
                    )


                    # ------------------------------------------
                    # DONE FLAG
                    # ------------------------------------------

                    agent_dones[
                        agent_index
                    ].append(
                        done
                    )


                    episode_reward += reward


                # ==================================================
                # 17. CALCULATE EPISODE METRICS
                # ==================================================

                total_vehicles = sum(
                    state["vehicle_count"]
                    for state in (
                        next_local_states.values()
                    )
                )


                total_waiting = sum(
                    state["waiting_time"]
                    for state in (
                        next_local_states.values()
                    )
                )


                total_queue = sum(
                    state["queue_length"]
                    for state in (
                        next_local_states.values()
                    )
                )


                total_speed = sum(
                    state["average_speed"]
                    for state in (
                        next_local_states.values()
                    )
                )


                average_speed_step = (
                    total_speed
                    / NUM_AGENTS
                )


                episode_vehicle_count += (
                    total_vehicles
                )

                episode_waiting_time += (
                    total_waiting
                )

                episode_queue += (
                    total_queue
                )

                episode_speed += (
                    average_speed_step
                )


                # ==================================================
                # 18. SAVE STEP DATA
                # ==================================================

                writer.writerow([
                    episode + 1,
                    step,
                    total_vehicles,
                    total_waiting,
                    average_speed_step,
                    total_queue,
                    episode_reward
                ])


                # ==================================================
                # 19. DISPLAY PROGRESS
                # ==================================================

                if step % 100 == 0:

                    print(
                        f"Step {step}: "
                        f"Vehicles={total_vehicles}"
                    )


            # ==================================================
            # 20. UPDATE MAPPO AGENTS
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

                        agent_global_states[
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
                        ],

                        agent_next_values[
                            agent_index
                        ],

                        agent_dones[
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
            # 21. EPISODE RESULT
            # ==================================================

            average_episode_vehicles = (
                episode_vehicle_count
                / MAX_STEPS
            )


            average_episode_waiting = (
                episode_waiting_time
                / MAX_STEPS
            )


            average_episode_speed = (
                episode_speed
                / MAX_STEPS
            )


            average_episode_queue = (
                episode_queue
                / MAX_STEPS
            )


            print()

            print(
                "======================================"
            )

            print(
                f"Episode {episode + 1} Results"
            )

            print(
                "======================================"
            )

            print(
                f"Average vehicles: "
                f"{average_episode_vehicles:.2f}"
            )

            print(
                f"Average waiting time: "
                f"{average_episode_waiting:.2f}"
            )

            print(
                f"Average speed: "
                f"{average_episode_speed:.2f} m/s"
            )

            print(
                f"Average queue length: "
                f"{average_episode_queue:.2f}"
            )

            print(
                f"Episode reward: "
                f"{episode_reward:.2f}"
            )

            print()


            # ==================================================
            # CLOSE SUMO-GUI
            # ==================================================

            try:
                traci.close()
            except:
                pass


    # ==================================================
    # SAVE TRAINED MAPPO MODELS
    # ==================================================

    MODEL_DIR = os.path.join(
        PROJECT_ROOT,
        "models",
        "mappo"
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


    print()

    print(
        "Saving trained MAPPO models..."
    )


    for agent_index, agent in enumerate(
        agents
    ):

        actor_path = os.path.join(
            MODEL_DIR,
            f"actor_{agent_index + 1:02d}.pth"
        )

        critic_path = os.path.join(
            MODEL_DIR,
            f"critic_{agent_index + 1:02d}.pth"
        )


        torch.save(
            agent.actor.state_dict(),
            actor_path
        )


        torch.save(
            agent.critic.state_dict(),
            critic_path
        )


        print(
            f"Agent {agent_index + 1}: "
            f"Actor and Critic saved."
        )


    print()

    print(
        "MAPPO models saved to:"
    )

    print(
        MODEL_DIR
    )


    # ==================================================
    # TRAINING COMPLETE
    # ==================================================

    print()

    print(
        "======================================"
    )

    print(
        "MAPPO TRAINING COMPLETED"
    )

    print(
        "======================================"
    )

    print()

    print(
        "MAPPO results saved to:"
    )

    print(
        RESULT_FILE
    )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    train()