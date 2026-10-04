from src.emergency.scenario_routes import generate_special_vehicles


TOTAL_SIMULATIONS = 1000

emergency_count = 0
vip_count = 0

for simulation_number in range(1, TOTAL_SIMULATIONS + 1):

    vehicles = generate_special_vehicles(
        simulation_number,
        TOTAL_SIMULATIONS
    )

    for vehicle in vehicles:

        if vehicle["type"] == "emergency":
            emergency_count += 1

        elif vehicle["type"] == "vip":
            vip_count += 1


print("================================")
print("SCENARIO ROUTE TEST")
print("================================")

print("Total simulations:", TOTAL_SIMULATIONS)
print("Emergency scenarios:", emergency_count)
print("VIP scenarios:", vip_count)

print()

if emergency_count >= 50:
    print("Emergency requirement: PASSED")
else:
    print("Emergency requirement: FAILED")

if vip_count >= 50:
    print("VIP requirement: PASSED")
else:
    print("VIP requirement: FAILED")

if emergency_count >= 50 and vip_count >= 50:
    print()
    print("ALL REQUIREMENTS PASSED")