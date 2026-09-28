import os
import pandas as pd
import numpy as np


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results",
    "traffic_data.csv"
)


def load_data():

    data = pd.read_csv(INPUT_FILE)

    print("\nDataset loaded successfully!")
    print("Rows:", len(data))
    print("Columns:", list(data.columns))

    return data


def create_sequences(data, sequence_length=10):

    values = data[
        [
            "vehicle_count",
            "waiting_time",
            "average_speed",
            "queue_length"
        ]
    ].values

    X = []
    y = []

    for i in range(len(values) - sequence_length):

        X.append(
            values[i:i + sequence_length]
        )

        # Predict the next traffic state
        y.append(
            values[i + sequence_length]
        )

    return np.array(X), np.array(y)


if __name__ == "__main__":

    data = load_data()

    X, y = create_sequences(
        data,
        sequence_length=10
    )

    print("\n======================================")
    print("Preprocessing completed!")
    print("======================================")

    print("X shape:", X.shape)
    print("y shape:", y.shape)