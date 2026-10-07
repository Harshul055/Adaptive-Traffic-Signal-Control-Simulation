import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction.lstm import TrafficLSTM


SEQUENCE_LENGTH = 10
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

FEATURES = [
    "vehicle_count",
    "waiting_time",
    "average_speed",
    "queue_length",
]

DATA_FILE = os.path.join(
    PROJECT_ROOT, "evaluation", "results", "traffic_data.csv"
)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "lstm")
MODEL_FILE = os.path.join(MODEL_DIR, "traffic_lstm.pth")


def train():
    print("\n" + "=" * 50)
    print("LSTM TRAINING")
    print("=" * 50)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Traffic data not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)
    missing = [column for column in FEATURES if column not in data.columns]
    if missing:
        raise ValueError(f"Missing LSTM features: {missing}")

    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=FEATURES).reset_index(drop=True)

    if len(data) <= SEQUENCE_LENGTH:
        raise ValueError(
            f"Need more than {SEQUENCE_LENGTH} valid rows; found {len(data)}."
        )

    values = data[FEATURES].to_numpy(dtype=np.float32)
    feature_min = values.min(axis=0)
    feature_max = values.max(axis=0)
    feature_range = feature_max - feature_min
    feature_range[feature_range == 0] = 1.0

    scaled = (values - feature_min) / feature_range

    X = []
    y = []
    for index in range(len(scaled) - SEQUENCE_LENGTH):
        X.append(scaled[index:index + SEQUENCE_LENGTH])
        y.append(scaled[index + SEQUENCE_LENGTH])

    X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(y), dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = TrafficLSTM(
        input_size=len(FEATURES),
        hidden_size=64,
        num_layers=1,
        output_size=len(FEATURES),
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Valid records: {len(data)}")
    print(f"Input shape: {tuple(X_tensor.shape)}")
    print(f"Target shape: {tuple(y_tensor.shape)}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / max(len(loader), 1)
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"- Loss: {average_loss:.6f}"
        )

    os.makedirs(MODEL_DIR, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_min": feature_min.tolist(),
        "feature_max": feature_max.tolist(),
        "sequence_length": SEQUENCE_LENGTH,
        "features": FEATURES,
    }
    torch.save(checkpoint, MODEL_FILE)

    print(f"\nModel saved to:\n{MODEL_FILE}")


if __name__ == "__main__":
    train()
