"""
A2CTrainer — Phase 2
======================
Baseline Advantage Actor-Critic trainer extending ActorCriticTrainer.

Key characteristics:
  - n-step returns (no GAE)
  - Single gradient update per rollout (1 epoch, no mini-batch shuffling)
  - No PPO clipping
  - Combined loss: L = -P_Loss - c_e * Entropy + c_v * V_Loss

Improvements over the reference notebook:
  - Gradient clipping (max_norm=0.5)
  - Vectorized environments (16 parallel instances)
  - TensorBoard logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional

from algorithms.base_trainer import ActorCriticTrainer
from algorithms.rollout_buffer import RolloutBuffer
from utils.running_stats import RunningMeanStd


class A2CTrainer(ActorCriticTrainer):
    """
    Synchronous A2C trainer with vectorized environments.
    Extends ActorCriticTrainer with n-step returns and single-epoch updates.
    """

    def __init__(
        self,
        env,
        model,
        n_steps: int = 128,
        n_updates: int = 200,
        gamma: float = 0.99,
        lr: float = 3e-4,
        c_v: float = 0.5,
        c_e: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        log_dir: str = "results/a2c",
        save_dir: str = "checkpoints/a2c",
        normalize_rewards: bool = False,
    ):
        super().__init__(
            env=env,
            model=model,
            n_steps=n_steps,
            n_updates=n_updates,
            gamma=gamma,
            c_v=c_v,
            c_e=c_e,
            max_grad_norm=max_grad_norm,
            device=device,
            log_dir=log_dir,
            save_dir=save_dir,
            normalize_rewards=normalize_rewards,
            track_raw_rewards=False,
            algo_name="A2C",
        )
        self.lr = lr
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Buffer
    # ------------------------------------------------------------------

    def _create_buffer(self) -> RolloutBuffer:
        return RolloutBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            obs_dim=self.obs_dim,
            n_intersections=self.n_intersections,
            gamma=self.gamma,
            gae_lambda=1.0,
            device=str(self.device),
        )

    # ------------------------------------------------------------------
    # Advantages
    # ------------------------------------------------------------------

    def _compute_advantages(self, last_value: torch.Tensor, last_done: torch.Tensor):
        self.buffer.compute_nstep_returns(last_value, last_done)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _update_model(self, update: int, c_e: float) -> Dict:
        T, E = self.n_steps, self.n_envs
        N = T * E
        device = self.device

        obs = self.buffer.obs.reshape(N, -1).to(device)
        actions = self.buffer.actions.reshape(N, -1).to(device)
        advantages = self.buffer.advantages.reshape(N).to(device)
        returns = self.buffer.returns.reshape(N).to(device)

        log_probs_new, values_new, entropy = self.model.evaluate_actions(obs, actions)

        policy_loss = -(log_probs_new.sum(dim=-1) * advantages.detach()).mean()
        value_loss = nn.functional.mse_loss(values_new.squeeze(-1), returns.detach())
        loss = policy_loss + self.c_v * value_loss - c_e * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "p_loss": float(policy_loss.item()),
            "v_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _init_history(self) -> Dict:
        return {"p_loss": [], "v_loss": [], "entropy": [],
                "avg_reward": [], "avg_len": []}

    def _update_history(self, history, metrics, update, c_e, avg_r, avg_l):
        history["p_loss"].append(metrics["p_loss"])
        history["v_loss"].append(metrics["v_loss"])
        history["entropy"].append(metrics["entropy"])
        history["avg_reward"].append(avg_r)
        history["avg_len"].append(avg_l)
        return history

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_metrics(self, metrics, update, c_e, avg_r, avg_l):
        self.logger.log_scalar("P_Loss", metrics["p_loss"], update)
        self.logger.log_scalar("V_Loss", metrics["v_loss"], update)
        self.logger.log_scalar("Entropy", metrics["entropy"], update)
        self.logger.log_scalar("Avg_Reward", avg_r, update)
        self.logger.log_scalar("Avg_Len", avg_l, update)

    def _log_status(self, update, metrics, avg_r, avg_l, elapsed):
        print(
            f"[A2C] iter {update:4d}/{self.n_updates} | "
            f"P_Loss: {metrics['p_loss']:7.4f} | "
            f"V_Loss: {metrics['v_loss']:7.4f} | "
            f"Entropy: {metrics['entropy']:.4f} | "
            f"Avg Reward: {avg_r:7.3f} | "
            f"Avg Len: {avg_l:6.1f} | "
            f"t: {elapsed:.1f}s"
        )

    # ------------------------------------------------------------------
    # Save / Load (extend base with optimizer state)
    # ------------------------------------------------------------------

    def save(self, path: str):
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.reward_normalizer is not None:
            payload["rms_mean"] = float(self.reward_normalizer.mean)
            payload["rms_var"] = float(self.reward_normalizer.var)
            payload["rms_count"] = float(self.reward_normalizer.count)
        torch.save(payload, path)
        print(f"[A2C] Saved checkpoint to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.reward_normalizer is not None and "rms_mean" in ckpt:
            self.reward_normalizer.mean = float(ckpt["rms_mean"])
            self.reward_normalizer.var = float(ckpt["rms_var"])
            self.reward_normalizer.count = float(ckpt["rms_count"])
        print(f"[A2C] Loaded checkpoint from {path}")
