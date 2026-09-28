import os
import sys
import time

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.environment.sumo_env import SumoEnvironment


env = SumoEnvironment(use_gui=True)

state = env.reset()

print("\n======================================")
print("Environment started!")
print("======================================")

for step in range(100):

    state, reward, done = env.step("A0")

    print(
        "Step:", step,
        "| Vehicles:", state["vehicle_count"],
        "| Queue:", state["queue_length"],
        "| Reward:", round(reward, 2)
    )

    time.sleep(0.1)

    if done:
        print("Simulation finished.")
        break

env.close()

print("\nEnvironment closed successfully!")