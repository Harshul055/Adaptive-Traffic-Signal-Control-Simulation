import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.actor import Actor
from src.agents.critic import Critic


class MAPPO:

    def __init__(
        self,
        state_size=36,
        action_size=4,
        hidden_size=64,
        global_state_size=576,
        learning_rate=0.0003,
        gamma=0.99,
        clip_epsilon=0.2,
        ppo_epochs=5,
        gae_lambda=0.95,
        entropy_coefficient=0.01
    ):

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = entropy_coefficient
        self.global_state_size = global_state_size

        # ==================================================
        # ACTOR
        # ==================================================

        # Actor receives local state:
        # 32 GNN + 4 LSTM features for one intersection

        self.actor = Actor(
            input_size=state_size,
            hidden_size=hidden_size,
            action_size=action_size
        )

        # ==================================================
        # CENTRALIZED CRITIC
        # ==================================================

        # 16 intersections × 36 local policy features = 576

        self.critic = Critic(
            input_size=global_state_size,
            hidden_size=hidden_size
        )

        # ==================================================
        # OPTIMIZERS
        # ==================================================

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=learning_rate
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=learning_rate
        )

    # ==================================================
    # SELECT ACTION
    # ==================================================

    def select_action(
        self,
        state,
        global_state
    ):

        # --------------------------------------------------
        # LOCAL STATE
        # --------------------------------------------------

        if not isinstance(
            state,
            torch.Tensor
        ):

            state = torch.tensor(
                state,
                dtype=torch.float32
            )

        # --------------------------------------------------
        # GLOBAL STATE
        # --------------------------------------------------

        if not isinstance(
            global_state,
            torch.Tensor
        ):

            global_state = torch.tensor(
                global_state,
                dtype=torch.float32
            )

        # --------------------------------------------------
        # ADD BATCH DIMENSION
        # --------------------------------------------------

        if state.dim() == 1:

            state = state.unsqueeze(0)

        if global_state.dim() == 1:

            global_state = global_state.unsqueeze(0)

        # ==================================================
        # ACTOR
        # ==================================================

        probabilities = self.actor(
            state
        )

        distribution = torch.distributions.Categorical(
            probabilities
        )

        action = distribution.sample()

        log_probability = distribution.log_prob(
            action
        )

        # ==================================================
        # CENTRALIZED CRITIC
        # ==================================================

        value = self.critic(
            global_state
        )

        return (
            action.item(),
            log_probability.item(),
            value.item()
        )

    # ==================================================
    # UPDATE MAPPO
    # ==================================================

    def update(
        self,
        states,
        global_states,
        actions,
        old_log_probs,
        rewards,
        values,
        next_values,
        dones
    ):

        # ==================================================
        # CONVERT DATA TO TENSORS
        # ==================================================

        states = torch.stack(
            states
        ).float()

        global_states = torch.stack(
            global_states
        ).float()

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

        next_values = torch.tensor(
            next_values,
            dtype=torch.float32
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32
        )

        # ==================================================
        # GAE ADVANTAGE CALCULATION
        # ==================================================

        advantages = torch.zeros_like(
            rewards
        )

        gae = 0.0

        for t in reversed(
            range(len(rewards))
        ):

            # --------------------------------------------------
            # TD ERROR
            # --------------------------------------------------

            delta = (
                rewards[t]
                + self.gamma
                * next_values[t]
                * (1 - dones[t])
                - values[t]
            )

            # --------------------------------------------------
            # GAE
            # --------------------------------------------------

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * (1 - dones[t])
                * gae
            )

            advantages[t] = gae

        # ==================================================
        # CALCULATE RETURNS
        # ==================================================

        # IMPORTANT:
        # Calculate returns BEFORE normalizing advantages.

        returns = (
            advantages
            + values
        )

        returns = returns.detach()

        # ==================================================
        # NORMALIZE ADVANTAGES
        # ==================================================

        advantages = (
            advantages
            - advantages.mean()
        ) / (
            advantages.std()
            + 1e-8
        )

        advantages = advantages.detach()

        # ==================================================
        # PPO EPOCHS
        # ==================================================

        actor_loss_value = 0.0

        critic_loss_value = 0.0

        for epoch in range(
            self.ppo_epochs
        ):

            # ==================================================
            # ACTOR
            # ==================================================

            probabilities = self.actor(
                states
            )

            distribution = (
                torch.distributions.Categorical(
                    probabilities
                )
            )

            new_log_probs = (
                distribution.log_prob(
                    actions
                )
            )

            # --------------------------------------------------
            # PPO RATIO
            # --------------------------------------------------

            ratio = torch.exp(
                new_log_probs
                - old_log_probs
            )

            # --------------------------------------------------
            # PPO CLIPPING
            # --------------------------------------------------

            clipped_ratio = torch.clamp(
                ratio,
                1 - self.clip_epsilon,
                1 + self.clip_epsilon
            )

            # --------------------------------------------------
            # PPO OBJECTIVE
            # --------------------------------------------------

            policy_objective = torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            )

            # --------------------------------------------------
            # ENTROPY
            # --------------------------------------------------

            entropy = (
                distribution
                .entropy()
                .mean()
            )

            # --------------------------------------------------
            # ACTOR LOSS
            # --------------------------------------------------

            actor_loss = (
                -policy_objective.mean()
                - self.entropy_coefficient
                * entropy
            )

            # --------------------------------------------------
            # ACTOR UPDATE
            # --------------------------------------------------

            self.actor_optimizer.zero_grad()

            actor_loss.backward()

            self.actor_optimizer.step()

            # ==================================================
            # CENTRALIZED CRITIC
            # ==================================================

            predicted_values = (
                self.critic(
                    global_states
                ).squeeze(-1)
            )

            # --------------------------------------------------
            # CRITIC LOSS
            # --------------------------------------------------

            critic_loss = nn.MSELoss()(
                predicted_values,
                returns
            )

            # --------------------------------------------------
            # CRITIC UPDATE
            # --------------------------------------------------

            self.critic_optimizer.zero_grad()

            critic_loss.backward()

            self.critic_optimizer.step()

            # ==================================================
            # STORE LOSSES
            # ==================================================

            actor_loss_value = (
                actor_loss.item()
            )

            critic_loss_value = (
                critic_loss.item()
            )

            # ==================================================
            # TRAINING INFORMATION
            # ==================================================

            print(
                f"PPO Epoch "
                f"{epoch + 1}/{self.ppo_epochs} | "
                f"Ratio="
                f"{ratio.mean().item():.4f} | "
                f"Entropy="
                f"{entropy.item():.4f} | "
                f"Actor Loss="
                f"{actor_loss_value:.6f} | "
                f"Critic Loss="
                f"{critic_loss_value:.6f}"
            )

        return (
            actor_loss_value,
            critic_loss_value
        )