import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):

    def __init__(self, input_features, output_features):
        super(GraphConvolution, self).__init__()

        self.linear = nn.Linear(
            input_features,
            output_features
        )

    def forward(self, x, adjacency):

        # Add self-connections
        identity = torch.eye(
            adjacency.size(0),
            device=adjacency.device
        )

        adjacency = adjacency + identity

        # Normalize adjacency matrix
        degree = adjacency.sum(dim=1)

        degree_inv_sqrt = torch.pow(
            degree,
            -0.5
        )

        degree_inv_sqrt[
            torch.isinf(degree_inv_sqrt)
        ] = 0

        D = torch.diag(degree_inv_sqrt)

        normalized_adjacency = (
            D @ adjacency @ D
        )

        # Message passing
        x = normalized_adjacency @ x

        # Feature transformation
        x = self.linear(x)

        return x


class TrafficGNN(nn.Module):

    def __init__(
        self,
        input_features=6,
        hidden_features=64,
        output_features=32
    ):
        super(TrafficGNN, self).__init__()

        self.layer1 = GraphConvolution(
            input_features,
            hidden_features
        )

        self.layer2 = GraphConvolution(
            hidden_features,
            output_features
        )

    def forward(self, x, adjacency):

        x = self.layer1(
            x,
            adjacency
        )

        x = F.relu(x)

        x = self.layer2(
            x,
            adjacency
        )

        return x