import pandas as pd


# ==================================================
# TRAFFIC FEATURES
# ==================================================

FEATURES = [
    "vehicle_count",
    "waiting_time",
    "average_speed",
    "queue_length"
]


# ==================================================
# NORMALIZE DATA
# ==================================================

def normalize_data(data):

    data = data.copy()

    for feature in FEATURES:

        if feature not in data.columns:
            raise ValueError(
                f"Missing feature: {feature}"
            )

        minimum = data[feature].min()
        maximum = data[feature].max()

        # Avoid division by zero
        if maximum == minimum:

            data[feature] = 0.0

        else:

            data[feature] = (
                data[feature] - minimum
            ) / (
                maximum - minimum
            )

    return data


# ==================================================
# DENORMALIZE DATA
# ==================================================

def denormalize_value(
    value,
    minimum,
    maximum
):

    return (
        value * (maximum - minimum)
        + minimum
    )


# ==================================================
# PREPARE FEATURES
# ==================================================

def prepare_features(data):

    normalized_data = normalize_data(
        data
    )

    return normalized_data[FEATURES]


# ==================================================
# MAIN TEST
# ==================================================

if __name__ == "__main__":

    print()
    print("=" * 50)
    print("DATA PREPROCESSING")
    print("=" * 50)

    # Test dataset
    test_data = pd.DataFrame({
        "vehicle_count": [10, 20, 30],
        "waiting_time": [100, 200, 300],
        "average_speed": [5, 10, 15],
        "queue_length": [2, 4, 6]
    })

    normalized = normalize_data(
        test_data
    )

    print()
    print("Original data:")
    print(test_data)

    print()
    print("Normalized data:")
    print(normalized)

    print()
    print("Preprocessing completed.")

    print("=" * 50)