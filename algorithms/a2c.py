"""
A2CTrainer — Phase 2
======================
Baseline Advantage Actor-Critic trainer.

Key characteristics matching the original A2C notebook:
  - n-step returns (no GAE)
  - Single gradient update per rollout (1 epoch, no mini-batch shuffling)
  - No PPO clipping
  - Combined loss: L = -P_Loss - c_e * Entropy + c_v * V_Loss

Improvements over the reference notebook:
  - Gradient clipping (max_norm=0.5)  [already safer than the reference]
  - Vectorized environments (16 parallel instances)
  - TensorBoard logging

Phase 2 purpose: establish a reward baseline to compare against PPO in Phase 3.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional

from algorithms.rollout_buffer import RolloutBuffer
from utils.running_stats import RunningMeanStd
from utils.logger import MetricLogger


class A2CTrainer:
    """
    Synchronous A2C trainer with vectorized environments.

    Args:
        env         : vectorized gymnasium env (SyncVectorEnv)
        model       : SharedActorCritic
        n_steps     : rollout length per env per update (default 128)
        n_updates   : total number of update iterations (default 200)
        gamma       : discount factor
        lr          : Adam learning rate
        c_v         : value loss coefficient
        c_e         : entropy coefficient (static in A2C baseline)
        max_grad_norm: gradient clipping max norm
        device      : torch device string
        log_dir     : TensorBoard log directory
        save_dir    : checkpoint save directory
    """

    def __init__(
        self,
        env,
        model,
        n_steps:       int   = 128,
        n_updates:     int   = 200,
        gamma:         float = 0.99,
        lr:            float = 3e-4,
        c_v:           float = 0.5,
        c_e:           float = 0.01,
        max_grad_norm: float = 0.5,
        device:        str   = "cpu",
        log_dir:       str   = "results/a2c",
        save_dir:      str   = "checkpoints/a2c",
        normalize_rewards: bool = False,
    ):
        self.env           = env
        self.model         = model.to(device)
        self.n_steps       = n_steps
        self.n_updates     = n_updates
        self.gamma         = gamma
        self.c_v           = c_v
        self.c_e           = c_e
        self.max_grad_norm = max_grad_norm
        self.device        = torch.device(device)
        self.save_dir      = save_dir

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # Infer dimensions from env
        self.n_envs          = env.num_envs
        self.obs_dim         = env.single_observation_space.shape[0]
        self.n_intersections = env.single_action_space.nvec.shape[0]

        self.buffer = RolloutBuffer(
            n_steps         = n_steps,
            n_envs          = self.n_envs,
            obs_dim         = self.obs_dim,
            n_intersections = self.n_intersections,
            gamma           = gamma,
            gae_lambda      = 1.0,    # not used in A2C, but set for completeness
            device          = device,
        )

        self.reward_normalizer = (
            RunningMeanStd(shape=()) if normalize_rewards else None
        )
        self.logger = MetricLogger(log_dir=log_dir, algo_name="A2C")

        # Episode tracking
        self._ep_rewards  = np.zeros(self.n_envs, dtype=np.float32)
        self._ep_lengths  = np.zeros(self.n_envs, dtype=np.int32)
        self._completed_rewards: list = []
        self._completed_lengths: list = []

    # ------------------------------------------------------------------
    def train(self) -> dict:
        """
        Run the full A2C training loop.

        Returns:
            history dict with keys: p_loss, v_loss, entropy, avg_reward, avg_len
        """
        history = {
            "p_loss":     [],
            "v_loss":     [],
            "entropy":    [],
            "avg_reward": [],
            "avg_len":    [],
        }

        obs_np, _ = self.env.reset()
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)

        done_np = np.zeros(self.n_envs, dtype=np.float32)

        print(f"[A2C] Starting training: {self.n_updates} updates x "
              f"{self.n_steps} steps x {self.n_envs} envs")
        t0 = time.time()

        for update in range(1, self.n_updates + 1):
            self.buffer.reset()

            # ---- Rollout collection ----
            for _ in range(self.n_steps):
                with torch.no_grad():
                    actions, log_probs, values = self.model.act(obs)

                actions_np = actions.cpu().numpy()
                obs_next_np, rewards_np, term_np, trunc_np, infos = (
                    self.env.step(actions_np)
                )
                done_np = np.logical_or(term_np, trunc_np).astype(np.float32)

                # Normalize rewards if requested
                if self.reward_normalizer is not None:
                    self.reward_normalizer.update(rewards_np)
                    rewards_np = rewards_np / (self.reward_normalizer.std + 1e-8)

                # Track episode stats
                self._ep_rewards += rewards_np
                self._ep_lengths += 1
                for i, done in enumerate(done_np):
                    if done:
                        self._completed_rewards.append(self._ep_rewards[i])
                        self._completed_lengths.append(self._ep_lengths[i])
                        self._ep_rewards[i] = 0.0
                        self._ep_lengths[i] = 0

                self.buffer.add(
                    obs       = obs,
                    actions   = actions,
                    rewards   = torch.tensor(rewards_np, dtype=torch.float32),
                    values    = values,
                    log_probs = log_probs,
                    dones     = torch.tensor(done_np,    dtype=torch.float32),
                )

                obs = torch.tensor(obs_next_np, dtype=torch.float32,
                                   device=self.device)

            # Bootstrap last value
            with torch.no_grad():
                last_value = self.model.get_value(obs)

            # n-step returns (A2C, not GAE)
            self.buffer.compute_nstep_returns(
                last_value = last_value,
                last_done  = torch.tensor(done_np, dtype=torch.float32),
            )
            self.buffer.normalize_advantages()

            # ---- Single gradient update ----
            p_loss_val, v_loss_val, ent_val = self._update()

            # ---- Logging ----
            avg_r = (float(np.mean(self._completed_rewards[-self.n_envs*2:]))
                     if self._completed_rewards else 0.0)
            avg_l = (float(np.mean(self._completed_lengths[-self.n_envs*2:]))
                     if self._completed_lengths else 0.0)

            history["p_loss"].append(p_loss_val)
            history["v_loss"].append(v_loss_val)
            history["entropy"].append(ent_val)
            history["avg_reward"].append(avg_r)
            history["avg_len"].append(avg_l)

            self.logger.log_scalar("P_Loss",     p_loss_val, update)
            self.logger.log_scalar("V_Loss",     v_loss_val, update)
            self.logger.log_scalar("Entropy",    ent_val,    update)
            self.logger.log_scalar("Avg_Reward", avg_r,      update)
            self.logger.log_scalar("Avg_Len",    avg_l,      update)

            if update % 10 == 0 or update == 1:
                elapsed = time.time() - t0
                print(
                    f"[A2C] iter {update:4d}/{self.n_updates} | "
                    f"P_Loss: {p_loss_val:7.4f} | "
                    f"V_Loss: {v_loss_val:7.4f} | "
                    f"Entropy: {ent_val:.4f} | "
                    f"Avg Reward: {avg_r:7.3f} | "
                    f"Avg Len: {avg_l:6.1f} | "
                    f"t: {elapsed:.1f}s"
                )

        self.logger.flush()
        print(f"[A2C] Training complete in {time.time()-t0:.1f}s")
        return history

    # ------------------------------------------------------------------
    def _update(self):
        """Single full-batch A2C gradient update (1 epoch, no clipping)."""
        T, E = self.n_steps, self.n_envs
        N = T * E
        device = self.device

        obs       = self.buffer.obs.reshape(N, -1).to(device)
        actions   = self.buffer.actions.reshape(N, -1).to(device)
        advantages= self.buffer.advantages.reshape(N).to(device)
        returns   = self.buffer.returns.reshape(N).to(device)

        log_probs_new, values_new, entropy = self.model.evaluate_actions(
            obs, actions
        )

        # Policy loss (actor)
        policy_loss = -(log_probs_new.sum(dim=-1) * advantages.detach()).mean()

        # Value loss (critic)
        value_loss = nn.functional.mse_loss(
            values_new.squeeze(-1), returns.detach()
        )

        # Total loss
        loss = policy_loss + self.c_v * value_loss - self.c_e * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return (
            float(policy_loss.item()),
            float(value_loss.item()),
            float(entropy.item()),
        )

    # ------------------------------------------------------------------
    def save(self, path: str):
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        payload = {
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        # F1 (A2C): persist RunningMeanStd if reward normalization was used
        if self.reward_normalizer is not None:
            payload["rms_mean"]  = float(self.reward_normalizer.mean)
            payload["rms_var"]   = float(self.reward_normalizer.var)
            payload["rms_count"] = float(self.reward_normalizer.count)
        torch.save(payload, path)
        print(f"[A2C] Saved checkpoint to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.reward_normalizer is not None and "rms_mean" in ckpt:
            self.reward_normalizer.mean  = float(ckpt["rms_mean"])
            self.reward_normalizer.var   = float(ckpt["rms_var"])
            self.reward_normalizer.count = float(ckpt["rms_count"])
        print(f"[A2C] Loaded checkpoint from {path}")

