"""
RolloutBuffer — Phase 2 / 3
==============================
Stores on-policy experience collected from a vectorized environment.

Supports:
  - n-step returns         (Phase 2, A2C)
  - GAE lambda-returns     (Phase 3, PPO)

Buffer size: n_steps * n_envs transitions (stored in CPU tensors).
"""

import torch
import numpy as np
from typing import Generator, Tuple


class RolloutBuffer:
    """
    Circular (single-fill) rollout buffer for on-policy algorithms.

    Layout: each field has shape (n_steps, n_envs, ...).
    After compute_returns_and_advantages() the buffer is ready for
    mini-batch iteration via get_minibatches().
    """

    def __init__(
        self,
        n_steps:         int,
        n_envs:          int,
        obs_dim:         int,
        n_intersections: int,
        gamma:           float = 0.99,
        gae_lambda:      float = 0.95,
        device:          str   = "cpu",
    ):
        self.n_steps         = n_steps
        self.n_envs          = n_envs
        self.obs_dim         = obs_dim
        self.n_intersections = n_intersections
        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.device          = torch.device(device)

        self._ptr = 0    # current write pointer
        self._full = False

        self._allocate()

    # ------------------------------------------------------------------
    def _allocate(self):
        T, E = self.n_steps, self.n_envs
        D    = self.obs_dim
        A    = self.n_intersections

        self.obs        = torch.zeros(T, E, D,  dtype=torch.float32)
        self.actions    = torch.zeros(T, E, A,  dtype=torch.long)
        self.rewards    = torch.zeros(T, E,      dtype=torch.float32)
        self.values     = torch.zeros(T, E,      dtype=torch.float32)
        self.log_probs  = torch.zeros(T, E, A,  dtype=torch.float32)
        self.dones      = torch.zeros(T, E,      dtype=torch.float32)

        # Computed after collection
        self.advantages = torch.zeros(T, E,      dtype=torch.float32)
        self.returns    = torch.zeros(T, E,      dtype=torch.float32)

    # ------------------------------------------------------------------
    def reset(self):
        self._ptr  = 0
        self._full = False

    # ------------------------------------------------------------------
    def add(
        self,
        obs:       torch.Tensor,   # (E, D)
        actions:   torch.Tensor,   # (E, A)
        rewards:   torch.Tensor,   # (E,)
        values:    torch.Tensor,   # (E,)
        log_probs: torch.Tensor,   # (E, A)
        dones:     torch.Tensor,   # (E,)
    ):
        t = self._ptr
        self.obs[t]       = obs.cpu()
        self.actions[t]   = actions.cpu()
        self.rewards[t]   = rewards.cpu()
        self.values[t]    = values.cpu().squeeze(-1)
        self.log_probs[t] = log_probs.cpu()
        self.dones[t]     = dones.cpu()
        self._ptr += 1
        if self._ptr >= self.n_steps:
            self._full = True

    # ------------------------------------------------------------------
    def compute_gae(self, last_value: torch.Tensor, last_done: torch.Tensor):
        """
        Generalized Advantage Estimation (GAE).

        Args:
            last_value : (E,) or (E, 1) — V(s_T) bootstrap
            last_done  : (E,) — whether last state is terminal
        """
        last_value = last_value.cpu().squeeze(-1)  # (E,)
        last_done  = last_done.cpu().float()

        gae = torch.zeros(self.n_envs)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value        = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value        = self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    # ------------------------------------------------------------------
    def compute_nstep_returns(
        self, last_value: torch.Tensor, last_done: torch.Tensor
    ):
        """
        Standard n-step discounted returns (used in baseline A2C).
        Advantages = returns - values.
        """
        last_value = last_value.cpu().squeeze(-1)
        last_done  = last_done.cpu().float()

        R = last_value * (1.0 - last_done)
        for t in reversed(range(self.n_steps)):
            non_terminal = 1.0 - self.dones[t]
            R = self.rewards[t] + self.gamma * R * non_terminal
            self.returns[t] = R

        self.advantages = self.returns - self.values

    # ------------------------------------------------------------------
    def normalize_advantages(self):
        """Standardize advantages to zero mean, unit variance."""
        adv = self.advantages
        self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

    # ------------------------------------------------------------------
    def get_minibatches(
        self, batch_size: int
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Iterate over shuffled mini-batches of size batch_size.
        Flattens the (n_steps, n_envs) dimensions into (n_steps*n_envs,).
        """
        T, E = self.n_steps, self.n_envs
        N = T * E

        obs       = self.obs.reshape(N, -1).to(self.device)
        actions   = self.actions.reshape(N, -1).to(self.device)
        log_probs = self.log_probs.reshape(N, -1).to(self.device)
        advantages= self.advantages.reshape(N).to(self.device)
        returns   = self.returns.reshape(N).to(self.device)
        values    = self.values.reshape(N).to(self.device)

        indices = torch.randperm(N)
        for start in range(0, N, batch_size):
            idx = indices[start : start + batch_size]
            yield (
                obs[idx],
                actions[idx],
                log_probs[idx],
                advantages[idx],
                returns[idx],
                values[idx],
            )

    @property
    def is_full(self) -> bool:
        return self._full
