from src.emergency.scenario_generator import generate_scenario


total_simulations = 1000

emergency_count = 0
vip_count = 0

for simulation in range(1, total_simulations + 1):

    scenario = generate_scenario(
        simulation,
        total_simulations
    )

    if scenario["emergency"]:
        emergency_count += 1

    if scenario["vip"]:
        vip_count += 1


print("Total simulations:", total_simulations)
print("Emergency scenarios:", emergency_count)
print("VIP scenarios:", vip_count)

print()

if emergency_count >= 50:
    print("Emergency requirement PASSED")
else:
    print("Emergency requirement FAILED")

if vip_count >= 50:
    print("VIP requirement PASSED")
else:
    print("VIP requirement FAILED")