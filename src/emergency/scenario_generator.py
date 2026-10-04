import random


def generate_scenario(simulation_number, total_simulations=1000):
    """
    Generate emergency/VIP scenario for one simulation.

    Guarantees:
    - At least 50 emergency scenarios
    - At least 50 VIP scenarios
    - 1000 total simulations
    """

    remaining_simulations = total_simulations - simulation_number + 1

    emergency_needed = max(0, 50 - simulation_number + 1)
    vip_needed = max(0, 50 - simulation_number + 1)

    emergency_probability = 0.05
    vip_probability = 0.05

    emergency = False
    vip = False

    # Guarantee remaining emergency cases
    if emergency_needed >= remaining_simulations:
        emergency = True
    elif random.random() < emergency_probability:
        emergency = True

    # Guarantee remaining VIP cases
    if vip_needed >= remaining_simulations:
        vip = True
    elif random.random() < vip_probability:
        vip = True

    return {
        "simulation": simulation_number,
        "emergency": emergency,
        "vip": vip
    }