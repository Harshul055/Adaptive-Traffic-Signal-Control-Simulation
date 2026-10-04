import csv
import os
import traci


# ==================================================
# COLLECT ONE TRAFFIC STATE
# ==================================================

def collect_traffic_state(step):

    vehicle_ids = traci.vehicle.getIDList()

    vehicle_count = len(vehicle_ids)

    total_waiting_time = 0.0
    total_speed = 0.0
    queue_length = 0

    for vehicle_id in vehicle_ids:

        speed = traci.vehicle.getSpeed(
            vehicle_id
        )

        waiting_time = (
            traci.vehicle.getAccumulatedWaitingTime(
                vehicle_id
            )
        )

        total_speed += speed
        total_waiting_time += waiting_time

        if speed < 0.1:

            queue_length += 1

    if vehicle_count > 0:

        average_speed = (
            total_speed
            / vehicle_count
        )

    else:

        average_speed = 0.0

    return {
        "step": step,
        "vehicle_count": vehicle_count,
        "waiting_time": total_waiting_time,
        "average_speed": average_speed,
        "queue_length": queue_length
    }


# ==================================================
# COLLECT MULTIPLE STEPS
# ==================================================

def collect_traffic_data(
    max_steps=1000
):

    data = []

    for step in range(
        max_steps
    ):

        traci.simulationStep()

        state = collect_traffic_state(
            step
        )

        data.append(state)

    return data


# ==================================================
# SAVE DATA TO CSV
# ==================================================

def save_traffic_data(
    data,
    output_file
):

    output_directory = os.path.dirname(
        output_file
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    fieldnames = [
        "step",
        "vehicle_count",
        "waiting_time",
        "average_speed",
        "queue_length"
    ]

    with open(
        output_file,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(data)


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    print(
        "Traffic collector module loaded successfully."
    )