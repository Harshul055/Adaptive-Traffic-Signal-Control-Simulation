import pandas as pd


def calculate_metrics(data):
    """
    Calculate average traffic performance metrics.
    """

    metrics = {
        "average_vehicle_count": data["vehicle_count"].mean(),
        "average_waiting_time": data["waiting_time"].mean(),
        "average_speed": data["average_speed"].mean(),
        "average_queue_length": data["queue_length"].mean(),
    }

    if "reward" in data.columns:
        metrics["total_reward"] = data["reward"].iloc[-1]
    else:
        metrics["total_reward"] = None

    return metrics


def calculate_improvement(fixed_value, mappo_value, lower_is_better=True):
    """
    Calculate percentage improvement of MAPPO over Fixed-Time.
    """

    if fixed_value == 0:
        return 0.0

    if lower_is_better:
        improvement = (
            (fixed_value - mappo_value)
            / fixed_value
        ) * 100
    else:
        improvement = (
            (mappo_value - fixed_value)
            / fixed_value
        ) * 100

    return float(improvement)


def compare_metrics(fixed_metrics, mappo_metrics):
    """
    Compare Fixed-Time and MAPPO metrics.
    """

    comparison = {
        "Average Vehicles": {
            "Fixed-Time": fixed_metrics["average_vehicle_count"],
            "MAPPO": mappo_metrics["average_vehicle_count"],
            "Improvement (%)": calculate_improvement(
                fixed_metrics["average_vehicle_count"],
                mappo_metrics["average_vehicle_count"],
                lower_is_better=True
            )
        },

        "Average Waiting Time": {
            "Fixed-Time": fixed_metrics["average_waiting_time"],
            "MAPPO": mappo_metrics["average_waiting_time"],
            "Improvement (%)": calculate_improvement(
                fixed_metrics["average_waiting_time"],
                mappo_metrics["average_waiting_time"],
                lower_is_better=True
            )
        },

        "Average Speed": {
            "Fixed-Time": fixed_metrics["average_speed"],
            "MAPPO": mappo_metrics["average_speed"],
            "Improvement (%)": calculate_improvement(
                fixed_metrics["average_speed"],
                mappo_metrics["average_speed"],
                lower_is_better=False
            )
        },

        "Average Queue Length": {
            "Fixed-Time": fixed_metrics["average_queue_length"],
            "MAPPO": mappo_metrics["average_queue_length"],
            "Improvement (%)": calculate_improvement(
                fixed_metrics["average_queue_length"],
                mappo_metrics["average_queue_length"],
                lower_is_better=True
            )
        }
    }

    return comparison