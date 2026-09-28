import torch

from src.agents.actor import Actor
from src.agents.critic import Critic
from src.agents.mappo import MAPPO


print("\n======================================")
print("MAPPO TEST")
print("======================================")


# Create dummy GNN state
state = torch.randn(32)


# --------------------------------------
# Actor test
# --------------------------------------

actor = Actor(
    input_size=32,
    hidden_size=64,
    action_size=4
)

action_probabilities = actor(
    state.unsqueeze(0)
)

print("\nActor output:")
print(action_probabilities)

print(
    "Actor shape:",
    action_probabilities.shape
)


# --------------------------------------
# Critic test
# --------------------------------------

critic = Critic(
    input_size=32,
    hidden_size=64
)

value = critic(
    state.unsqueeze(0)
)

print("\nCritic output:")
print(value)

print(
    "Critic shape:",
    value.shape
)


# --------------------------------------
# MAPPO test
# --------------------------------------

agent = MAPPO(
    state_size=32,
    action_size=4
)

action, log_prob, value = agent.select_action(
    state
)

print("\nMAPPO action:", action)
print("Log probability:", log_prob)
print("State value:", value)


print("\n======================================")
print("MAPPO TEST COMPLETED")
print("======================================")