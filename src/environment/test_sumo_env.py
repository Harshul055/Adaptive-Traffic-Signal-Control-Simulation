import traci

from src.environment.sumo_env import SumoEnvironment
from src.emergency.detector import (
    get_emergency_vehicles,
    get_vip_vehicles
)


env = SumoEnvironment(use_gui=True)

print("Starting environment...")

state = env.reset()

print("Environment started")
print("Initial state:", state)

for step in range(1000):

    state, reward, done = env.step("A0")

    emergency = get_emergency_vehicles()
    vip = get_vip_vehicles()

    print(
        f"Step {step + 1}: "
        f"Reward={reward}, "
        f"Emergency={emergency}, "
        f"VIP={vip}, "
        f"Done={done}"
    )

    if done:
        break

env.close()

print("Environment test finished.")