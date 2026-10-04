import os
import sys
import xml.etree.ElementTree as ET

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.emergency.scenario_routes import generate_special_vehicles


OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "sumo-rl",
    "sumo_rl",
    "nets",
    "RESCO",
    "grid4x4",
    "test_scenario.rou.xml"
)

vehicles = generate_special_vehicles(
    simulation_number=1,
    total_emergency=5,
    total_vip=5
)

root = ET.Element("routes")

# Vehicle types
ET.SubElement(
    root,
    "vType",
    id="emergency",
    accel="2.6",
    decel="4.5",
    sigma="0.5",
    length="5",
    maxSpeed="13.9"
)

ET.SubElement(
    root,
    "vType",
    id="vip",
    accel="2.6",
    decel="4.5",
    sigma="0.5",
    length="5",
    maxSpeed="13.9"
)

# Route
ET.SubElement(
    root,
    "route",
    id="special_route",
    edges="left0A0 A0A1 A1A2 A2A3 A3top0"
)

# Vehicles
for vehicle in vehicles:

    ET.SubElement(
        root,
        "vehicle",
        id=vehicle["id"],
        type=vehicle["type"],
        depart=str(vehicle["depart"]),
        route="special_route"
    )

tree = ET.ElementTree(root)

ET.indent(tree, space="    ")

tree.write(
    OUTPUT_FILE,
    encoding="UTF-8",
    xml_declaration=True
)

emergency_count = sum(
    1 for v in vehicles
    if v["type"] == "emergency"
)

vip_count = sum(
    1 for v in vehicles
    if v["type"] == "vip"
)

print("=" * 60)
print("EMERGENCY / VIP SCENARIO GENERATED")
print("=" * 60)

print(f"Emergency vehicles : {emergency_count}")
print(f"VIP vehicles       : {vip_count}")
print(f"Total special      : {len(vehicles)}")

print("\nDeparture times:")

for vehicle in vehicles:
    print(
        f"{vehicle['id']:<15}"
        f" {vehicle['type']:<10}"
        f" step={vehicle['depart']}"
    )

print("\nSaved to:")
print(OUTPUT_FILE)

print("=" * 60)