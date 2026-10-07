"""Online LSTM traffic predictor used by the MAPPO policy."""

from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.prediction.lstm import TrafficLSTM


FEATURES = [
    "vehicle_count",
    "waiting_time",
    "average_speed",
    "queue_length",
]


class OnlineTrafficPredictor:
    """Maintain a rolling traffic history and predict the next global state."""

    def __init__(self, model_path, sequence_length=10, device="cpu"):
        self.model_path = Path(model_path)
        self.sequence_length = int(sequence_length)
        self.device = torch.device(device)
        self.history = deque(maxlen=self.sequence_length)

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )

        self.model = TrafficLSTM(
            input_size=4,
            hidden_size=64,
            num_layers=1,
            output_size=4
        ).to(self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            saved_length = int(
                checkpoint.get("sequence_length", self.sequence_length)
            )
            if saved_length != self.sequence_length:
                raise ValueError(
                    f"LSTM checkpoint sequence length {saved_length} "
                    f"does not match {self.sequence_length}."
                )
            minimum = checkpoint.get("feature_min")
            maximum = checkpoint.get("feature_max")
            if minimum is None or maximum is None:
                raise ValueError(
                    "LSTM checkpoint is missing normalization metadata."
                )
            self.feature_min = np.asarray(minimum, dtype=np.float32)
            self.feature_max = np.asarray(maximum, dtype=np.float32)
        else:
            raise ValueError(
                "Unsupported LSTM checkpoint. Retrain with training/train_lstm.py."
            )

        self.feature_range = self.feature_max - self.feature_min
        self.feature_range[self.feature_range == 0] = 1.0
        self.model.eval()

    def reset(self):
        self.history.clear()

    @staticmethod
    def state_to_vector(state):
        return np.asarray(
            [
                state.get("vehicle_count", 0.0),
                state.get("waiting_time", 0.0),
                state.get("average_speed", 0.0),
                state.get("queue_length", 0.0),
            ],
            dtype=np.float32,
        )

    def predict(self, current_state):
        vector = self.state_to_vector(current_state)

        if not self.history:
            for _ in range(self.sequence_length):
                self.history.append(vector.copy())
        else:
            self.history.append(vector.copy())

        history = np.asarray(self.history, dtype=np.float32)
        normalized = (history - self.feature_min) / self.feature_range

        tensor = torch.tensor(
            normalized,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(tensor).cpu().numpy()[0]

        normalized_prediction = prediction
        prediction = (
            normalized_prediction * self.feature_range
            + self.feature_min
        )

        return torch.tensor(
            prediction,
            dtype=torch.float32
        )

    def predict_normalized(self, current_state):
        vector = self.state_to_vector(current_state)

        if not self.history:
            for _ in range(self.sequence_length):
                self.history.append(vector.copy())
        else:
            self.history.append(vector.copy())

        history = np.asarray(self.history, dtype=np.float32)
        normalized = (history - self.feature_min) / self.feature_range

        tensor = torch.tensor(
            normalized,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(tensor).cpu().numpy()[0]

        return torch.tensor(
            prediction,
            dtype=torch.float32
        )
