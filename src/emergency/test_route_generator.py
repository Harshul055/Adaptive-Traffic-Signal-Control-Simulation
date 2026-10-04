import os

from src.emergency.route_generator import (
    create_scenario_route_file
)


BASE_ROUTE = (
    r"C:\Project\Adaptive-Traffic-Control"
    r"\sumo-rl\sumo_rl\nets\RESCO\grid4x4"
    r"\grid4x4_1.rou.xml"
)

OUTPUT_ROUTE = (
    r"C:\Project\Adaptive-Traffic-Control"
    r"\sumo-rl\sumo_rl\nets\RESCO\grid4x4"
    r"\test_scenario.rou.xml"
)


result = create_scenario_route_file(
    simulation_number=1,
    base_route_file=BASE_ROUTE,
    output_route_file=OUTPUT_ROUTE
)

print("Route file created:")
print(result)

print()

if os.path.exists(result):
    print("TEST PASSED")
else:
    print("TEST FAILED")