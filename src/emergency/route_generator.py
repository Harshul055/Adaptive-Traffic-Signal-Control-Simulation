import os
import re

from src.emergency.scenario_routes import generate_special_vehicles


def create_scenario_route_file(
    simulation_number,
    base_route_file,
    output_route_file
):
    """
    Create a SUMO route file for one simulation.

    The original route file is never modified.
    Existing emergency/VIP vehicles are removed.
    Their vType definitions are preserved.
    """

    # 1. Read original route file
    with open(base_route_file, "r", encoding="utf-8") as file:
        content = file.read()

    # 2. Remove old emergency/VIP vehicles
    special_vehicle_pattern = re.compile(
        r'\s*<vehicle\s+'
        r'id="(?:emergency_\d+|vip_\d+)"'
        r'.*?</vehicle>',
        re.DOTALL
    )

    content = special_vehicle_pattern.sub("", content)

    # 3. Find all remaining normal vehicles
    vehicle_pattern = re.compile(
        r'<vehicle\b.*?</vehicle>',
        re.DOTALL
    )

    existing_blocks = vehicle_pattern.findall(content)

    parsed_vehicles = []

    for block in existing_blocks:

        match = re.search(
            r'depart="([0-9.]+)"',
            block
        )

        if match:

            depart_time = float(
                match.group(1)
            )

            parsed_vehicles.append(
                (
                    depart_time,
                    block
                )
            )

    # 4. Generate emergency/VIP vehicles
    vehicles = generate_special_vehicles(
        simulation_number
    )

    for vehicle in vehicles:

        block = f"""
    <vehicle id="{vehicle['id']}"
             type="{vehicle['type']}"
             depart="{vehicle['depart']:.2f}">
        <route edges="{vehicle['route']}"/>
    </vehicle>
"""

        parsed_vehicles.append(
            (
                vehicle["depart"],
                block
            )
        )

    # 5. Sort all vehicles by departure time
    parsed_vehicles.sort(
        key=lambda item: item[0]
    )

    # 6. Remove ALL vehicle blocks
    #    but keep vType definitions
    content_without_vehicles = vehicle_pattern.sub(
        "",
        content
    )

    # 7. Create sorted vehicle section
    sorted_vehicle_blocks = "\n".join(
        block
        for _, block in parsed_vehicles
    )

    # 8. Insert vehicles before </routes>
    routes_end = content_without_vehicles.rfind(
        "</routes>"
    )

    if routes_end == -1:
        raise RuntimeError(
            "Could not find </routes> in route file."
        )

    new_content = (
        content_without_vehicles[:routes_end]
        + "\n"
        + sorted_vehicle_blocks
        + "\n"
        + content_without_vehicles[routes_end:]
    )

    # 9. Create output directory
    output_directory = os.path.dirname(
        output_route_file
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    # 10. Save generated route file
    with open(
        output_route_file,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(new_content)

    return output_route_file