import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.actor import Actor
from src.agents.critic import Critic


class MAPPO:

    def __init__(
        self,
        state_size=32,
        action_size=4,
        hidden_size=64,
        learning_rate=0.0003,
        gamma=0.99,
        clip_epsilon=0.2,
        ppo_epochs=5
    ):

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs

        self.actor = Actor(
            input_size=state_size,
            hidden_size=hidden_size,
            action_size=action_size
        )

        self.critic = Critic(
            input_size=state_size,
            hidden_size=hidden_size
        )

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=learning_rate
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=learning_rate
        )

    def select_action(self, state):

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(
                state,
                dtype=torch.float32
            )

        if state.dim() == 1:
            state = state.unsqueeze(0)

        probabilities = self.actor(state)

        distribution = torch.distributions.Categorical(
            probabilities
        )

        action = distribution.sample()

        log_probability = distribution.log_prob(action)

        value = self.critic(state)

        return (
            action.item(),
            log_probability.item(),
            value.item()
        )

    def update(
        self,
        states,
        actions,
        old_log_probs,
        rewards,
        values
    ):

        states = torch.stack(states).float()

        actions = torch.tensor(
            actions,
            dtype=torch.long
        )

        old_log_probs = torch.tensor(
            old_log_probs,
            dtype=torch.float32
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        )

        values = torch.tensor(
            values,
            dtype=torch.float32
        )

        # --------------------------------
        # Calculate discounted returns
        # --------------------------------

        returns = []
        discounted_reward = 0.0

        for reward in reversed(rewards):

            discounted_reward = (
                reward
                + self.gamma * discounted_reward
            )

            returns.insert(
                0,
                discounted_reward
            )

        returns = torch.tensor(
            returns,
            dtype=torch.float32
        )

        # --------------------------------
        # Normalize returns
        # --------------------------------

        returns = (
            returns - returns.mean()
        ) / (
            returns.std() + 1e-8
        )

        # --------------------------------
        # Calculate advantages
        # --------------------------------

        advantages = returns - values

        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )

        # Prevent unnecessary gradient tracking
        advantages = advantages.detach()

        # --------------------------------
        # Multiple PPO epochs
        # --------------------------------

        actor_loss_value = 0
        critic_loss_value = 0

        for epoch in range(self.ppo_epochs):

            # ==============================
            # ACTOR UPDATE
            # ==============================

            probabilities = self.actor(states)

            distribution = torch.distributions.Categorical(
                probabilities
            )

            new_log_probs = distribution.log_prob(
                actions
            )

            # PPO probability ratio
            ratio = torch.exp(
                new_log_probs - old_log_probs
            )

            # Clipped ratio
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.clip_epsilon,
                1 + self.clip_epsilon
            )

            # PPO objective
            policy_objective = torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            )

            actor_loss = -policy_objective.mean()

            self.actor_optimizer.zero_grad()

            actor_loss.backward()

            self.actor_optimizer.step()

            # ==============================
            # CRITIC UPDATE
            # ==============================

            predicted_values = self.critic(
                states
            ).squeeze(-1)

            critic_loss = nn.MSELoss()(
                predicted_values,
                returns
            )

            self.critic_optimizer.zero_grad()

            critic_loss.backward()

            self.critic_optimizer.step()

            actor_loss_value = actor_loss.item()
            critic_loss_value = critic_loss.item()

            print(
                f"PPO Epoch {epoch + 1}/{self.ppo_epochs} | "
                f"Ratio={ratio.mean().item():.4f} | "
                f"Actor Loss={actor_loss_value:.6f} | "
                f"Critic Loss={critic_loss_value:.6f}"
            )

        return (
            actor_loss_value,
            critic_loss_value
        )