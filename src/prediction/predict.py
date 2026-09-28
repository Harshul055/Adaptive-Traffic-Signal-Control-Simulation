import os
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.prediction.lstm import TrafficLSTM


# -----------------------------
# Settings
# -----------------------------

SEQUENCE_LENGTH = 10

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


COLUMNS = [
    "vehicle_count",
    "waiting_time",
    "average_speed",
    "queue_length"
]


# -----------------------------
# Load data
# -----------------------------

data = pd.read_csv(DATA_FILE)

data[COLUMNS] = data[COLUMNS].apply(
    pd.to_numeric,
    errors="coerce"
)

data = data.dropna(
    subset=COLUMNS
).reset_index(drop=True)


# -----------------------------
# Convert to numpy
# -----------------------------

features = data[COLUMNS].values.astype(
    np.float32
)


# -----------------------------
# Normalize
# -----------------------------

feature_min = features.min(axis=0)
feature_max = features.max(axis=0)

normalized = (
    features - feature_min
) / (
    feature_max - feature_min + 1e-8
)


# -----------------------------
# Take last 10 states
# -----------------------------

last_sequence = normalized[
    -SEQUENCE_LENGTH:
]


X = torch.tensor(
    last_sequence,
    dtype=torch.float32
).unsqueeze(0)


# -----------------------------
# Load LSTM
# -----------------------------

model = TrafficLSTM(
    input_size=4,
    hidden_size=64,
    num_layers=2,
    output_size=4
)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=torch.device("cpu")
    )
)

model.eval()


# -----------------------------
# Make prediction
# -----------------------------

with torch.no_grad():

    prediction = model(X)

prediction = prediction.numpy()[0]


# -----------------------------
# Convert prediction back
# -----------------------------

prediction = (
    prediction *
    (feature_max - feature_min)
) + feature_min


# -----------------------------
# Display result
# -----------------------------

print("\n======================================")
print("LSTM TRAFFIC PREDICTION")
print("======================================")

print(
    "Current vehicles:",
    int(features[-1][0])
)

print(
    "Predicted vehicles:",
    round(float(prediction[0]), 2)
)

print(
    "Current waiting time:",
    round(float(features[-1][1]), 2)
)

print(
    "Predicted waiting time:",
    round(float(prediction[1]), 2)
)

print(
    "Current average speed:",
    round(float(features[-1][2]), 2)
)

print(
    "Predicted average speed:",
    round(float(prediction[2]), 2)
)

print(
    "Current queue length:",
    int(features[-1][3])
)

print(
    "Predicted queue length:",
    round(float(prediction[3]), 2)
)

print("\nPrediction completed successfully!")