import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(
        self,
        input_size=32,
        hidden_size=64,
        action_size=4
    ):
        super(Actor, self).__init__()

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
            action_size
        )

    def forward(self, state):

        x = F.relu(
            self.fc1(state)
        )

        x = F.relu(
            self.fc2(x)
        )

        action_logits = self.output(x)

        action_probabilities = F.softmax(
            action_logits,
            dim=-1
        )

        return action_probabilities