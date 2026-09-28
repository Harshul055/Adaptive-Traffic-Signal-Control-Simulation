import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(
        self,
        input_size=32,
        hidden_size=64
    ):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(
            input_size,
            hidden_size
        )

        self.fc2 = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.output = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, state):

        x = F.relu(
            self.fc1(state)
        )

        x = F.relu(
            self.fc2(x)
        )

        value = self.output(x)

        return value