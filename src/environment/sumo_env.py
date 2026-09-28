import os
import sys
import traci


# --------------------------------------------------
# Make project root available
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

sys.path.append(PROJECT_ROOT)


# --------------------------------------------------
# Import our modules
# --------------------------------------------------

from src.environment.state import get_traffic_state
from src.environment.action import change_phase
from src.environment.reward import calculate_reward


class SumoEnvironment:

    def __init__(self, use_gui=True):

        self.use_gui = use_gui

        # SUMO installation
        self.sumo_home = (
            r"C:\Program Files (x86)\Eclipse\Sumo"
        )

        # RESCO grid4x4
        self.sumo_config = (
            r"C:\Project\Adaptive-Traffic-Control"
            r"\sumo-rl\sumo_rl\nets\RESCO\grid4x4"
            r"\grid4x4.sumocfg"
        )

        # Select SUMO or SUMO-GUI
        if self.use_gui:
            self.sumo_binary = os.path.join(
                self.sumo_home,
                "bin",
                "sumo-gui.exe"
            )
        else:
            self.sumo_binary = os.path.join(
                self.sumo_home,
                "bin",
                "sumo.exe"
            )


    # --------------------------------------------------
    # Start environment
    # --------------------------------------------------

    def reset(self):

    # Close an existing SUMO connection
        try:
            traci.close()
        except:
            pass

        # Start SUMO
        traci.start([
            self.sumo_binary,
            "-c",
            self.sumo_config
        ])

        # Move simulation one step
        traci.simulationStep()

        # Get initial state
        state = get_traffic_state()

        return state


    # --------------------------------------------------
    # Take an action
    # --------------------------------------------------

    def step(self, tls_id):

        # Change traffic-light phase
        change_phase(tls_id)

        # Move SUMO forward
        traci.simulationStep()

        # Get new state
        state = get_traffic_state()

        # Calculate reward
        reward = calculate_reward(state)

        # Check whether simulation has finished
        done = (
            traci.simulation.getMinExpectedNumber()
            <= 0
        )

        return state, reward, done


    # --------------------------------------------------
    # Close environment
    # --------------------------------------------------

    def close(self):

            traci.close()