import torch
import torch.nn as nn


class TrafficLSTM(nn.Module):

    def __init__(
        self,
        input_size=4,
        hidden_size=64,
        num_layers=2,
        output_size=4
    ):

        super(TrafficLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        last_output = output[:, -1, :]

        prediction = self.fc(last_output)

        return prediction