"""Shared matched SUMO scenario generation for training and evaluation.

The same deterministic scenario must be used by MAPPO training and paper
evaluation so that reported comparisons are reproducible and matched.
"""

from pathlib import Path
import random
import xml.etree.ElementTree as ET


SPECIAL_ROUTE = "left0A0 A0A1 A1A2 A2A3 A3top0"
SPECIAL_COUNT = 5
DEPART_MIN = 100
DEPART_MAX = 900


def prepare_matched_scenario(grid_dir, seed=42):
    """Create the deterministic 5-emergency + 5-VIP test scenario."""
    grid_dir = Path(grid_dir)
    normal_file = grid_dir / "grid4x4_1.rou.xml"
    output_file = grid_dir / "test_scenario.rou.xml"

    if not normal_file.exists():
        raise FileNotFoundError(f"Normal route file not found: {normal_file}")

    rng = random.Random(seed)
    tree = ET.parse(normal_file)
    root = tree.getroot()

    for vehicle in list(root.findall("vehicle")):
        vehicle_id = vehicle.get("id", "")
        if vehicle_id.startswith("emergency_") or vehicle_id.startswith("vip_"):
            root.remove(vehicle)

    special = []

    for index in range(1, SPECIAL_COUNT + 1):
        special.append(
            {
                "id": f"emergency_{index}",
                "type": "emergency",
                "depart": rng.randint(DEPART_MIN, DEPART_MAX),
            }
        )

    for index in range(1, SPECIAL_COUNT + 1):
        special.append(
            {
                "id": f"vip_{index}",
                "type": "vip",
                "depart": rng.randint(DEPART_MIN, DEPART_MAX),
            }
        )

    for vehicle in special:
        element = ET.Element(
            "vehicle",
            {
                "id": vehicle["id"],
                "type": vehicle["type"],
                "depart": str(vehicle["depart"]),
            },
        )
        ET.SubElement(element, "route", {"edges": SPECIAL_ROUTE})
        root.append(element)

    vehicles = sorted(
        root.findall("vehicle"),
        key=lambda item: float(item.get("depart", "0")),
    )

    for vehicle in list(root.findall("vehicle")):
        root.remove(vehicle)

    for vehicle in vehicles:
        root.append(vehicle)

    tree.write(
        output_file,
        encoding="UTF-8",
        xml_declaration=True,
    )

    print("Matched scenario created:")
    print(output_file)
    print("Emergency vehicles: 5")
    print("VIP vehicles      : 5")
    print(f"Scenario seed     : {seed}")
    print("All special departures are before step 1000.")

    return output_file
