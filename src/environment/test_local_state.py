import traci

SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"

SUMO_CONFIG = (
    r"C:\Project\Adaptive-Traffic-Control"
    r"\sumo-rl\sumo_rl\nets\RESCO\grid4x4"
    r"\grid4x4.sumocfg"
)

traci.start([
    SUMO_BINARY,
    "-c",
    SUMO_CONFIG
])

from src.environment.state import get_all_local_states

# Run simulation for 100 steps
for step in range(100):

    traci.simulationStep()

    states = get_all_local_states()

    if step % 20 == 0:

        print("\n" + "=" * 50)
        print("STEP:", step)
        print("=" * 50)

        for tls_id, state in states.items():

            print(
                tls_id,
                "| Vehicles:",
                state["vehicle_count"],
                "| Queue:",
                state["queue_length"],
                "| Waiting:",
                round(state["waiting_time"], 2),
                "| Speed:",
                round(state["average_speed"], 2),
                "| Phase:",
                state["current_phase"]
            )

traci.close()