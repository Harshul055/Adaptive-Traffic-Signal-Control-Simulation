import os
import pandas as pd


# ==================================================
# PROJECT PATH
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results",
    "traffic_data.csv"
)


# ==================================================
# LOAD DATASET
# ==================================================

def load_dataset():

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Traffic dataset not found:\n{DATA_FILE}"
        )

    data = pd.read_csv(DATA_FILE)

    return data


# ==================================================
# SELECT FEATURES
# ==================================================

def get_features(data):

    features = [
        "vehicle_count",
        "waiting_time",
        "average_speed",
        "queue_length"
    ]

    for feature in features:

        if feature not in data.columns:
            raise ValueError(
                f"Missing feature: {feature}"
            )

    return data[features]


# ==================================================
# PREPARE DATASET
# ==================================================

def prepare_dataset():

    data = load_dataset()

    features = get_features(data)

    features = features.astype(float)

    return features


# ==================================================
# MAIN TEST
# ==================================================

if __name__ == "__main__":

    print()
    print("=" * 50)
    print("TRAFFIC DATASET")
    print("=" * 50)

    data = load_dataset()

    print(
        f"Total records: {len(data)}"
    )

    print(
        f"Columns: {list(data.columns)}"
    )

    features = prepare_dataset()

    print(
        f"Feature shape: {features.shape}"
    )

    print()
    print("Dataset loaded successfully.")

    print("=" * 50)