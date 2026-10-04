import os
import sys

import numpy as np
import pandas as pd
import torch

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results",
    "traffic_data.csv"
)

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "lstm",
    "traffic_lstm.pth"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results",
    "lstm_predictions.csv"
)


# --------------------------------------------------
# IMPORT MODEL
# --------------------------------------------------

from src.prediction.lstm import TrafficLSTM


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

SEQUENCE_LENGTH = 10

FEATURES = [
    "vehicle_count",
    "waiting_time",
    "average_speed",
    "queue_length"
]


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 60)
print("LSTM TRAFFIC PREDICTION")
print("=" * 60)

if not os.path.exists(DATA_FILE):
    print("Traffic data not found:")
    print(DATA_FILE)
    sys.exit()

if not os.path.exists(MODEL_FILE):
    print("LSTM model not found:")
    print(MODEL_FILE)
    sys.exit()


data = pd.read_csv(DATA_FILE)

print(f"\nLoaded rows: {len(data)}")


# --------------------------------------------------
# CHECK FEATURES
# --------------------------------------------------

for feature in FEATURES:

    if feature not in data.columns:

        print(
            f"Missing feature: {feature}"
        )

        sys.exit()


# --------------------------------------------------
# CLEAN DATA
# --------------------------------------------------

data = data.dropna(
    subset=FEATURES
).reset_index(drop=True)


# --------------------------------------------------
# NORMALIZE DATA
# --------------------------------------------------

values = data[FEATURES].values.astype(
    np.float32
)

minimum = values.min(axis=0)
maximum = values.max(axis=0)

range_values = maximum - minimum

range_values[range_values == 0] = 1

normalized = (
    (values - minimum)
    / range_values
)


# --------------------------------------------------
# CREATE MODEL
# --------------------------------------------------
model = TrafficLSTM(
    input_size=4,
    hidden_size=64,
    output_size=4,
    num_layers=1
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model.to(device)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()


# --------------------------------------------------
# GENERATE PREDICTIONS
# --------------------------------------------------

predictions = []

steps = []

with torch.no_grad():

    for i in range(
        SEQUENCE_LENGTH,
        len(normalized)
    ):

        sequence = normalized[
            i - SEQUENCE_LENGTH:i
        ]

        tensor = torch.tensor(
            sequence,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        prediction = model(
            tensor
        )

        prediction = prediction.cpu().numpy()[0]

        predictions.append(
            prediction
        )

        steps.append(
            int(data.iloc[i]["step"])
        )


# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------

predictions = np.array(
    predictions
)

lstm_data = pd.DataFrame({

    "step": steps,

    "vehicle_count": predictions[:, 0],

    "waiting_time": predictions[:, 1],

    "average_speed": predictions[:, 2],

    "queue_length": predictions[:, 3]
})


lstm_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print("\nLSTM prediction completed.")

print(
    f"Predictions generated: {len(lstm_data)}"
)

print("\nSaved to:")

print(OUTPUT_FILE)

print("\nColumns:")

for column in lstm_data.columns:
    print(f"✓ {column}")

print("=" * 60)