import os
import xml.etree.ElementTree as ET

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

NET_FILE = os.path.join(
    PROJECT_ROOT,
    "sumo-rl",
    "sumo_rl",
    "nets",
    "RESCO",
    "grid4x4",
    "grid4x4.net.xml"
)

tree = ET.parse(NET_FILE)
root = tree.getroot()

for tl in root.findall("tlLogic"):
    tls_id = tl.get("id")

    print("\n" + "=" * 50)
    print("TRAFFIC LIGHT:", tls_id)
    print("=" * 50)

    connections = []

    for connection in root.findall("connection"):
        if connection.get("tl") == tls_id:
            connections.append(connection)

    print("Controlled connections:", len(connections))

    for connection in connections:
        print(
            "from:", connection.get("from"),
            "| to:", connection.get("to"),
            "| lane:", connection.get("fromLane"),
            "->",
            connection.get("toLane")
        )