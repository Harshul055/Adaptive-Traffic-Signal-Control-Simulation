import traci


print("Starting SUMO...")

traci.start([
    r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe",
    "-c",
    r"C:\Project\Adaptive-Traffic-Control\sumo-rl\sumo_rl\nets\RESCO\grid4x4\grid4x4.sumocfg"
])


print()
print("Traffic lights:")

tls_ids = traci.trafficlight.getIDList()

for tls_id in tls_ids:

    current_phase = traci.trafficlight.getPhase(
        tls_id
    )

    logic = traci.trafficlight.getAllProgramLogics(
        tls_id
    )[0]

    phase_count = len(
        logic.phases
    )

    print(
        f"{tls_id}: "
        f"current phase={current_phase}, "
        f"phases={phase_count}"
    )


traci.close()

print()
print("TLS test finished.")