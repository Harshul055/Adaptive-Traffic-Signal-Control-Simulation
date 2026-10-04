import random
from pathlib import Path
import xml.etree.ElementTree as ET

# --------------------------------------------------
# FILES
# --------------------------------------------------

BASE_DIR = Path(r"C:\Project\Adaptive-Traffic-Control")

NORMAL_FILE = (
    BASE_DIR
    / "sumo-rl"
    / "sumo_rl"
    / "nets"
    / "RESCO"
    / "grid4x4"
    / "grid4x4_1.rou.xml"
)

OUTPUT_FILE = (
    BASE_DIR
    / "sumo-rl"
    / "sumo_rl"
    / "nets"
    / "RESCO"
    / "grid4x4"
    / "test_scenario.rou.xml"
)

# --------------------------------------------------
# SPECIAL VEHICLES
# --------------------------------------------------

random.seed(42)

special_vehicles = []

# 5 Emergency vehicles
for i in range(1, 6):
    special_vehicles.append({
        "id": f"emergency_{i}",
        "type": "emergency",
        "depart": random.randint(100, 900)
    })

# 5 VIP vehicles
for i in range(1, 6):
    special_vehicles.append({
        "id": f"vip_{i}",
        "type": "vip",
        "depart": random.randint(100, 900)
    })

# Sort special vehicles by departure time
special_vehicles.sort(key=lambda x: x["depart"])

# --------------------------------------------------
# CHECK NORMAL FILE
# --------------------------------------------------

if not NORMAL_FILE.exists():
    raise FileNotFoundError(
        f"Normal traffic file not found:\n{NORMAL_FILE}"
    )

# --------------------------------------------------
# LOAD ORIGINAL XML
# --------------------------------------------------

tree = ET.parse(NORMAL_FILE)
root = tree.getroot()

# --------------------------------------------------
# CREATE SPECIAL ROUTE
# --------------------------------------------------

SPECIAL_ROUTE = "left0A0 A0A1 A1A2 A2A3 A3top0"

# --------------------------------------------------
# ADD SPECIAL VEHICLES
# --------------------------------------------------

for vehicle in special_vehicles:

    vehicle_element = ET.Element(
        "vehicle",
        {
            "id": vehicle["id"],
            "type": vehicle["type"],
            "depart": str(vehicle["depart"])
        }
    )

    route_element = ET.SubElement(
        vehicle_element,
        "route",
        {
            "edges": SPECIAL_ROUTE
        }
    )

    root.append(vehicle_element)

# --------------------------------------------------
# SORT ALL VEHICLES BY DEPARTURE TIME
# --------------------------------------------------

vehicles = root.findall("vehicle")

vehicles.sort(
    key=lambda v: float(v.get("depart", "0"))
)

# Remove all vehicle elements
for vehicle in root.findall("vehicle"):
    root.remove(vehicle)

# Add them back in sorted order
for vehicle in vehicles:
    root.append(vehicle)

# --------------------------------------------------
# WRITE OUTPUT
# --------------------------------------------------

tree.write(
    OUTPUT_FILE,
    encoding="utf-8",
    xml_declaration=True
)

# --------------------------------------------------
# RESULT
# --------------------------------------------------

print()
print("==============================================")
print(" MERGED SUMO SCENARIO CREATED")
print("==============================================")
print()

print(f"Normal traffic source : {NORMAL_FILE.name}")
print(f"Output file           : {OUTPUT_FILE.name}")
print()

print("Normal traffic        : PRESERVED")
print("Emergency vehicles    : 5")
print("VIP vehicles          : 5")
print("Total special         : 10")

print()
print("SPECIAL VEHICLES")
print("----------------------------------------------")

for vehicle in special_vehicles:
    print(
        f"{vehicle['id']:15} "
        f"type={vehicle['type']:10} "
        f"depart={vehicle['depart']}"
    )

print()
print("All special vehicles depart before Step 1000.")
print("All vehicles sorted by departure time.")
print()
print("==============================================")
