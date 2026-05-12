"""
SharedActorCritic — Phase 2
=============================
Shared-backbone design with:
  - Common trunk:  Linear(144,256)+ReLU -> Linear(256,256)+ReLU
  - Actor heads:   16 x [Linear(256,4)+Softmax]
  - Critic head:   Linear(256,128)+ReLU -> Linear(128,1)

The multi-head actor produces one Categorical distribution per
intersection, enabling factored joint policy without exponential blowup.

For TorchScript export (Phase 5), the forward() method returns
(log_probs_list, value, entropy) when called with actions provided.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Optional, Tuple


N_INTERSECTIONS = 16
N_PHASES        = 4
OBS_DIM         = 144
TRUNK_HIDDEN    = 256
CRITIC_HIDDEN   = 128


class SharedActorCritic(nn.Module):
    """
    Shared trunk + multi-head actor + scalar critic.

    Forward pass returns (distributions, value):
        distributions : list of 16 Categorical distributions
        value         : tensor of shape (..., 1)

    Supports both single-obs [144] and batched obs [B, 144].
    """

    def __init__(
        self,
        obs_dim:          int = OBS_DIM,
        n_intersections:  int = N_INTERSECTIONS,
        n_phases:         int = N_PHASES,
        trunk_hidden:     int = TRUNK_HIDDEN,
        critic_hidden:    int = CRITIC_HIDDEN,
        init_std:         float = 0.01,
    ):
        super().__init__()
        self.n_intersections = n_intersections
        self.n_phases        = n_phases

        # --- Shared trunk ---
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, trunk_hidden),
            nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.ReLU(),
        )

        # --- Actor heads (one per intersection) ---
        self.actor_heads = nn.ModuleList([
            nn.Linear(trunk_hidden, n_phases)
            for _ in range(n_intersections)
        ])

        # --- Critic head ---
        self.critic = nn.Sequential(
            nn.Linear(trunk_hidden, critic_hidden),
            nn.ReLU(),
            nn.Linear(critic_hidden, 1),
        )

        # Weight initialization
        self._init_weights(init_std)

    # ------------------------------------------------------------------
    def _init_weights(self, std: float):
        for module in self.trunk.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)
        for head in self.actor_heads:
            nn.init.orthogonal_(head.weight, gain=std)
            nn.init.zeros_(head.bias)
        for module in self.critic.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[List[Categorical], torch.Tensor]:
        """
        Args:
            obs: tensor of shape (B, obs_dim) or (obs_dim,)
        Returns:
            distributions: list of N_INTERSECTIONS Categorical distributions
            value:         tensor of shape (B, 1)
        """
        latent = self.trunk(obs)                     # (B, 256)
        value  = self.critic(latent)                  # (B, 1)
        distributions = [
            Categorical(logits=head(latent))
            for head in self.actor_heads
        ]
        return distributions, value

    # ------------------------------------------------------------------
    def act(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the current policy.

        Returns:
            actions   : (B, N_INTERSECTIONS) int tensor
            log_probs : (B, N_INTERSECTIONS) float tensor
            value     : (B, 1) float tensor
        """
        distributions, value = self.forward(obs)
        actions   = torch.stack([d.sample()     for d in distributions], dim=-1)
        log_probs = torch.stack([
            distributions[i].log_prob(actions[..., i])
            for i in range(self.n_intersections)
        ], dim=-1)
        return actions, log_probs, value

    # ------------------------------------------------------------------
    def evaluate_actions(
        self,
        obs:     torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log-probs and entropy for given (obs, actions).

        Args:
            obs:     (B, obs_dim)
            actions: (B, N_INTERSECTIONS)
        Returns:
            log_probs : (B, N_INTERSECTIONS)
            value     : (B, 1)
            entropy   : scalar (mean entropy across intersections and batch)
        """
        distributions, value = self.forward(obs)
        log_probs = torch.stack([
            distributions[i].log_prob(actions[..., i])
            for i in range(self.n_intersections)
        ], dim=-1)                                   # (B, N_INTERSECTIONS)

        entropy = torch.stack([
            d.entropy() for d in distributions
        ], dim=-1).mean()                            # scalar

        return log_probs, value, entropy

    # ------------------------------------------------------------------
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute only the value estimate (no sampling)."""
        latent = self.trunk(obs)
        return self.critic(latent)                   # (B, 1)
