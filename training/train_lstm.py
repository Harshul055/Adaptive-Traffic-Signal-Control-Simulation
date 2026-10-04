import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==================================================
# PATHS
# ==================================================

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


# ==================================================
# SETTINGS
# ==================================================

SEQUENCE_LENGTH = 10
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

FEATURES = [
    "vehicle_count",
    "waiting_time",
    "average_speed",
    "queue_length"
]


# ==================================================
# LSTM MODEL
# ==================================================

class TrafficLSTM(nn.Module):

    def __init__(
        self,
        input_size=4,
        hidden_size=64,
        output_size=4
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        return self.fc(last_output)


# ==================================================
# LOAD DATA
# ==================================================

print()
print("=" * 50)
print("LSTM TRAINING")
print("=" * 50)

print()
print("Loading data...")

data = pd.read_csv(DATA_FILE)

print(
    f"Records loaded: {len(data)}"
)


# ==================================================
# SELECT FEATURES
# ==================================================

data = data[FEATURES]

data = data.replace(
    [float("inf"), float("-inf")],
    float("nan")
)

data = data.dropna()

print(
    f"Valid records: {len(data)}"
)


# ==================================================
# NORMALIZE DATA
# ==================================================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)


# ==================================================
# CREATE SEQUENCES
# ==================================================

X = []
y = []

for i in range(
    len(scaled_data) - SEQUENCE_LENGTH
):

    X.append(
        scaled_data[
            i:i + SEQUENCE_LENGTH
        ]
    )

    y.append(
        scaled_data[
            i + SEQUENCE_LENGTH
        ]
    )


X = torch.tensor(
    X,
    dtype=torch.float32
)

y = torch.tensor(
    y,
    dtype=torch.float32
)

print(
    f"Input shape: {X.shape}"
)

print(
    f"Target shape: {y.shape}"
)


# ==================================================
# DATASET
# ==================================================

dataset = TensorDataset(
    X,
    y
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==================================================
# MODEL
# ==================================================

model = TrafficLSTM()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==================================================
# TRAINING
# ==================================================

print()
print("Starting training...")

for epoch in range(EPOCHS):

    total_loss = 0.0

    for batch_x, batch_y in loader:

        optimizer.zero_grad()

        predictions = model(
            batch_x
        )

        loss = criterion(
            predictions,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss
        / len(loader)
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"Loss: {average_loss:.6f}"
    )


# ==================================================
# SAVE MODEL
# ==================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_FILE
)

print()
print("=" * 50)
print("LSTM TRAINING COMPLETE")
print("=" * 50)

print()
print(
    f"Model saved to:\n{MODEL_FILE}"
)