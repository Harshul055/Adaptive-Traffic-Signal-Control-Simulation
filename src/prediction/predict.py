import os
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction.lstm import TrafficLSTM


DATA_FILE = os.path.join(
    PROJECT_ROOT, "evaluation", "results", "traffic_data.csv"
)
MODEL_FILE = os.path.join(
    PROJECT_ROOT, "models", "lstm", "traffic_lstm.pth"
)
OUTPUT_FILE = os.path.join(
    PROJECT_ROOT, "evaluation", "results", "lstm_predictions.csv"
)

SEQUENCE_LENGTH = 10
FEATURES = [
    "vehicle_count",
    "waiting_time",
    "average_speed",
    "queue_length",
]


def main():
    print("=" * 60)
    print("LSTM TRAFFIC PREDICTION")
    print("=" * 60)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Traffic data not found: {DATA_FILE}")
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"LSTM model not found: {MODEL_FILE}")

    data = pd.read_csv(DATA_FILE)
    missing = [feature for feature in FEATURES if feature not in data.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=FEATURES).reset_index(drop=True)

    if len(data) <= SEQUENCE_LENGTH:
        raise ValueError(
            f"Need more than {SEQUENCE_LENGTH} valid rows; found {len(data)}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_FILE, map_location=device)

    model = TrafficLSTM(
        input_size=len(FEATURES),
        hidden_size=64,
        output_size=len(FEATURES),
        num_layers=1,
    ).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        feature_min = checkpoint.get("feature_min")
        feature_max = checkpoint.get("feature_max")
        checkpoint_sequence_length = int(
            checkpoint.get("sequence_length", SEQUENCE_LENGTH)
        )
        if checkpoint_sequence_length != SEQUENCE_LENGTH:
            raise ValueError(
                f"Checkpoint sequence length {checkpoint_sequence_length} "
                f"does not match configured length {SEQUENCE_LENGTH}."
            )
    else:
        # Backward compatibility for older plain state_dict checkpoints.
        model.load_state_dict(checkpoint)
        feature_min = None
        feature_max = None

    values = data[FEATURES].to_numpy(dtype=np.float32)

    if feature_min is None or feature_max is None:
        feature_min = values.min(axis=0)
        feature_max = values.max(axis=0)
    else:
        feature_min = np.asarray(feature_min, dtype=np.float32)
        feature_max = np.asarray(feature_max, dtype=np.float32)

    feature_range = feature_max - feature_min
    feature_range[feature_range == 0] = 1.0
    normalized = (values - feature_min) / feature_range

    model.eval()
    predictions = []
    steps = []

    with torch.no_grad():
        for index in range(SEQUENCE_LENGTH, len(normalized)):
            sequence = normalized[index - SEQUENCE_LENGTH:index]
            tensor = torch.tensor(
                sequence, dtype=torch.float32, device=device
            ).unsqueeze(0)

            prediction = model(tensor).cpu().numpy()[0]
            predictions.append(prediction)
            steps.append(int(data.iloc[index]["step"]))

    predictions = np.asarray(predictions, dtype=np.float32)

    # Convert predictions back to the original traffic-metric units.
    predictions = predictions * feature_range + feature_min

    output = pd.DataFrame(
        {
            "step": steps,
            "vehicle_count": predictions[:, 0],
            "waiting_time": predictions[:, 1],
            "average_speed": predictions[:, 2],
            "queue_length": predictions[:, 3],
        }
    )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print(f"Predictions generated: {len(output)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
