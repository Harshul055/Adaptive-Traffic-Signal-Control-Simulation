import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.prediction.lstm import TrafficLSTM

# Settings

SEQUENCE_LENGTH = 10
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001


DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "results",
    "traffic_data.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "lstm"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "traffic_lstm.pth"
)


# Load dataset

data = pd.read_csv(DATA_FILE)

columns = [
    "vehicle_count",
    "waiting_time",
    "average_speed",
    "queue_length"
]

# Convert values to numeric
for column in columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

# Replace infinite values with NaN
data[columns] = data[columns].replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove rows containing invalid values
data = data.dropna(
    subset=columns
).reset_index(drop=True)

print("Valid rows:", len(data))

features = data[columns].values.astype(
    np.float32
)
feature_min = features.min(axis=0)
feature_max = features.max(axis=0)

features = (
    features - feature_min
) / (
    feature_max - feature_min + 1e-8
)

print("\nNormalization check:")
print("Minimum values:", features.min(axis=0))
print("Maximum values:", features.max(axis=0))

# -----------------------------
# Create sequences
# -----------------------------

X = []
y = []

for i in range(
    len(features) - SEQUENCE_LENGTH
):

    X.append(
        features[
            i:i + SEQUENCE_LENGTH
        ]
    )

    y.append(
        features[
            i + SEQUENCE_LENGTH
        ]
    )


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)


print("Dataset shape:")
print("X:", X.shape)
print("y:", y.shape)


# -----------------------------
# Convert to PyTorch tensors
# -----------------------------

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)


dataset = TensorDataset(
    X_tensor,
    y_tensor
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# -----------------------------
# Create model
# -----------------------------

model = TrafficLSTM(
    input_size=4,
    hidden_size=64,
    num_layers=2,
    output_size=4
)


criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Training
# -----------------------------

print("\nStarting LSTM training...\n")

for epoch in range(EPOCHS):

    total_loss = 0

    for batch_X, batch_y in loader:

        optimizer.zero_grad()

        prediction = model(batch_X)

        loss = criterion(
            prediction,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(loader)
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.6f}"
    )


# -----------------------------
# Save model
# -----------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_FILE
)

print("\n======================================")
print("LSTM training completed!")
print("======================================")

print("Model saved to:")
print(MODEL_FILE)