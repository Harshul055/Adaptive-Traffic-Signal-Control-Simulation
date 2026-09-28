import os
import xml.etree.ElementTree as ET


# -----------------------------------
# Project paths
# -----------------------------------

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


# -----------------------------------
# Build graph
# -----------------------------------

def build_graph():

    tree = ET.parse(NET_FILE)
    root = tree.getroot()

    # --------------------------------
    # Get traffic-light IDs
    # --------------------------------

    traffic_lights = set()

    for tl_logic in root.findall("tlLogic"):

        tl_id = tl_logic.get("id")

        if tl_id:
            traffic_lights.add(tl_id)

    traffic_lights = sorted(traffic_lights)

    print("Traffic lights found:", len(traffic_lights))

    # --------------------------------
    # Create graph
    # --------------------------------

    graph = {}

    for tl_id in traffic_lights:
        graph[tl_id] = []

    # --------------------------------
    # Find connections between
    # traffic-light intersections
    # --------------------------------

    for connection in root.findall("connection"):

        from_edge = connection.get("from")
        to_edge = connection.get("to")

        if not from_edge or not to_edge:
            continue

        # Find edges
        from_edge_element = None
        to_edge_element = None

        for edge in root.findall("edge"):

            if edge.get("id") == from_edge:
                from_edge_element = edge

            if edge.get("id") == to_edge:
                to_edge_element = edge

        if (
            from_edge_element is None
            or to_edge_element is None
        ):
            continue

        from_node = from_edge_element.get("from")
        to_node = to_edge_element.get("to")

        if (
            from_node in graph
            and to_node in graph
            and from_node != to_node
        ):

            if to_node not in graph[from_node]:
                graph[from_node].append(to_node)

            if from_node not in graph[to_node]:
                graph[to_node].append(from_node)

    return graph


# -----------------------------------
# Adjacency matrix
# -----------------------------------

def create_adjacency_matrix(graph):

    nodes = sorted(graph.keys())

    matrix = []

    for node in nodes:

        row = []

        for other_node in nodes:

            if other_node in graph[node]:
                row.append(1)
            else:
                row.append(0)

        matrix.append(row)

    return nodes, matrix


# -----------------------------------
# Main
# -----------------------------------

if __name__ == "__main__":

    graph = build_graph()

    nodes, matrix = create_adjacency_matrix(graph)

    print("\n======================================")
    print("TRAFFIC INTERSECTION GRAPH")
    print("======================================")

    print("\nNumber of intersections:")
    print(len(nodes))

    print("\nIntersections and neighbors:")

    for node in nodes:

        print(
            node,
            "->",
            graph[node]
        )

    print("\nAdjacency Matrix:")

    print("     ", end="")

    for node in nodes:
        print(f"{node:>5}", end="")

    print()

    for i, node in enumerate(nodes):

        print(f"{node:>5}", end="")

        for value in matrix[i]:
            print(f"{value:>5}", end="")

        print()