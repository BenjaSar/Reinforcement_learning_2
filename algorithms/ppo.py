"""
PPOTrainer — v3 (Base class refactored)
=========================================
PPO-GAE trainer extending ActorCriticTrainer with all improvements:

ALGORITHMIC
  - PPO clipped surrogate objective (epsilon=0.2)
  - Generalized Advantage Estimation (lambda=0.95)
  - Reward normalization via RunningMeanStd (Welford)
  - Gradient clipping (max_norm=0.5)
  - Entropy coefficient decay (0.05 -> 0.01)
  - K=4 PPO epochs with mini-batches
  - Value function clipping (PPO-style VF clip)
  - Separate Adam optimizers: actor_lr=3e-4, critic_lr=1e-3
  - Reward normalizer warmup (512 random steps)

STRUCTURAL
  - LR decay with floor (5% of initial)
  - Raw reward tracked & logged separately
  - Periodic + best-model checkpointing
  - TorchScript export
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict

from algorithms.base_trainer import ActorCriticTrainer
from algorithms.rollout_buffer import RolloutBuffer


def _explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    var_returns = np.var(returns)
    if var_returns < 1e-8:
        return 0.0
    return float(1.0 - np.var(returns - values) / var_returns)


class PPOTrainer(ActorCriticTrainer):
    """
    Fixed and improved PPO trainer extending ActorCriticTrainer.
    Uses separate actor/critic optimizers, value function clipping,
    entropy decay, warmup, and best-model checkpointing.
    """

    def __init__(
        self,
        env,
        model,
        n_steps: int = 128,
        n_updates: int = 500,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        lr_decay: bool = True,
        lr_min_frac: float = 0.05,
        c_v: float = 0.5,
        c_e_start: float = 0.05,
        c_e_end: float = 0.01,
        n_epochs: int = 4,
        batch_size: int = 256,
        max_grad_norm: float = 0.5,
        kl_target: float = 0.02,
        normalize_rewards: bool = True,
        warmup_steps: int = 512,
        device: str = "cpu",
        log_dir: str = "results/ppo",
        save_dir: str = "checkpoints/ppo",
        save_freq: int = 50,
    ):
        # Set all attributes BEFORE super().__init__() so that
        # _create_buffer() (called during __init__) can access them.
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.lr_decay = lr_decay
        self.lr_min_frac = lr_min_frac
        self.c_e_start = c_e_start
        self.c_e_end = c_e_end
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.kl_target = kl_target
        self.warmup_steps = warmup_steps
        self.save_freq = save_freq

        super().__init__(
            env=env,
            model=model,
            n_steps=n_steps,
            n_updates=n_updates,
            gamma=gamma,
            c_v=c_v,
            c_e=c_e_start,
            max_grad_norm=max_grad_norm,
            device=device,
            log_dir=log_dir,
            save_dir=save_dir,
            normalize_rewards=normalize_rewards,
            track_raw_rewards=True,
            algo_name="PPO",
        )

        os.makedirs(save_dir, exist_ok=True)

        # Separate optimizers for actor vs. critic
        actor_params = (
            list(model.trunk.parameters())
            + [p for head in model.actor_heads for p in head.parameters()]
        )
        critic_params = list(model.critic.parameters())

        self.actor_optimizer = optim.Adam(actor_params, lr=actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(critic_params, lr=critic_lr, eps=1e-5)

        # LR schedulers with min-fraction floor
        if lr_decay:
            self.actor_scheduler = optim.lr_scheduler.LinearLR(
                self.actor_optimizer, start_factor=1.0,
                end_factor=lr_min_frac, total_iters=n_updates,
            )
            self.critic_scheduler = optim.lr_scheduler.LinearLR(
                self.critic_optimizer, start_factor=1.0,
                end_factor=lr_min_frac, total_iters=n_updates,
            )
        else:
            self.actor_scheduler = None
            self.critic_scheduler = None

    # ------------------------------------------------------------------
    # Buffer: GAE-compatible
    # ------------------------------------------------------------------

    def _create_buffer(self) -> RolloutBuffer:
        return RolloutBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            obs_dim=self.obs_dim,
            n_intersections=self.n_intersections,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=str(self.device),
        )

    # ------------------------------------------------------------------
    # Advantages: GAE
    # ------------------------------------------------------------------

    def _compute_advantages(self, last_value: torch.Tensor, last_done: torch.Tensor):
        self.buffer.compute_gae(last_value, last_done)

    # ------------------------------------------------------------------
    # Entropy decay
    # ------------------------------------------------------------------

    def _get_entropy_coeff(self, update: int) -> float:
        frac = (update - 1) / max(self.n_updates - 1, 1)
        return self.c_e_start + frac * (self.c_e_end - self.c_e_start)

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _on_training_start(self):
        if self.reward_normalizer is None or self.warmup_steps <= 0:
            return
        print(f"[PPO] Warming up reward normalizer ({self.warmup_steps} steps)...")
        obs_np, _ = self.env.reset()
        for _ in range(self.warmup_steps):
            actions_np = np.array([
                self.env.single_action_space.sample()
                for _ in range(self.n_envs)
            ])
            obs_next_np, rewards_np, _, _, _ = self.env.step(actions_np)
            self.reward_normalizer.update(rewards_np)
            obs_np = obs_next_np
        print(f"[PPO] Normalizer seeded: mean={self.reward_normalizer.mean:.4f}, "
              f"std={float(self.reward_normalizer.std):.4f}")

    # ------------------------------------------------------------------
    # Update: K-epoch PPO with VF clipping
    # ------------------------------------------------------------------

    def _update_model(self, update: int, c_e: float) -> Dict:
        p_losses, v_losses, entropies, kl_divs, grad_norms = [], [], [], [], []

        returns_flat = self.buffer.returns.reshape(-1)
        values_flat = self.buffer.values.reshape(-1)
        ev = _explained_variance(values_flat.numpy(), returns_flat.numpy())

        for _ in range(self.n_epochs):
            epoch_kl = 0.0
            n_batches = 0

            for batch in self.buffer.get_minibatches(self.batch_size):
                obs_b, act_b, old_lp_b, adv_b, ret_b, old_val_b = batch
                adv_b = adv_b.to(self.device)
                ret_b = ret_b.to(self.device)
                old_val_b = old_val_b.to(self.device)

                new_log_probs, values_new, entropy = self.model.evaluate_actions(
                    obs_b, act_b
                )
                values_new = values_new.squeeze(-1)

                # Policy loss (clipped surrogate)
                old_log_prob = old_lp_b.sum(dim=-1)
                new_log_prob = new_log_probs.sum(dim=-1)
                ratio = torch.exp(new_log_prob - old_log_prob)

                surr1 = ratio * adv_b
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Actor step
                actor_loss = policy_loss - c_e * entropy
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                actor_gn = nn.utils.clip_grad_norm_(
                    [p for pg in self.actor_optimizer.param_groups
                     for p in pg["params"]],
                    self.max_grad_norm,
                )
                self.actor_optimizer.step()

                # Critic step with VF clipping
                values_critic = self.model.get_value(obs_b).squeeze(-1)
                v_clipped = old_val_b + torch.clamp(
                    values_critic - old_val_b,
                    -self.clip_epsilon, +self.clip_epsilon,
                )
                crit_loss1 = nn.functional.mse_loss(values_critic, ret_b)
                crit_loss2 = nn.functional.mse_loss(v_clipped, ret_b)
                value_loss = torch.max(crit_loss1, crit_loss2)

                self.critic_optimizer.zero_grad()
                (self.c_v * value_loss).backward()
                critic_gn = nn.utils.clip_grad_norm_(
                    [p for pg in self.critic_optimizer.param_groups
                     for p in pg["params"]],
                    self.max_grad_norm,
                )
                self.critic_optimizer.step()

                grad_norm_combined = float((actor_gn**2 + critic_gn**2)**0.5)
                with torch.no_grad():
                    approx_kl = (old_log_prob - new_log_prob).mean().abs().item()

                p_losses.append(float(policy_loss.item()))
                v_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                kl_divs.append(approx_kl)
                grad_norms.append(grad_norm_combined)

                epoch_kl += approx_kl
                n_batches += 1

            if n_batches > 0 and (epoch_kl / n_batches) > self.kl_target:
                break

        return {
            "p_loss": float(np.mean(p_losses)),
            "v_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(entropies)),
            "kl_div": float(np.mean(kl_divs)),
            "grad_norm": float(np.mean(grad_norms)),
            "explained_var": float(ev),
        }

    # ------------------------------------------------------------------
    # Scheduler stepping
    # ------------------------------------------------------------------

    def _step_schedulers(self):
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _init_history(self) -> Dict:
        return {
            "p_loss": [], "v_loss": [], "entropy": [], "kl_div": [],
            "grad_norm": [], "explained_var": [],
            "avg_reward": [], "avg_reward_raw": [], "avg_len": [], "c_e": [],
        }

    def _update_history(self, history, metrics, update, c_e, avg_r, avg_l):
        window = self.n_envs * 4
        avg_r_raw = (
            float(np.mean(self._completed_rewards_raw[-window:]))
            if self._completed_rewards_raw else 0.0
        )
        for k in ("p_loss", "v_loss", "entropy", "kl_div", "grad_norm", "explained_var"):
            history[k].append(metrics[k])
        history["avg_reward"].append(avg_r)
        history["avg_reward_raw"].append(avg_r_raw)
        history["avg_len"].append(avg_l)
        history["c_e"].append(c_e)
        return history

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_metrics(self, metrics, update, c_e, avg_r, avg_l):
        window = self.n_envs * 4
        avg_r_raw = (
            float(np.mean(self._completed_rewards_raw[-window:]))
            if self._completed_rewards_raw else 0.0
        )
        for k, v in metrics.items():
            self.logger.log_scalar(k, v, update)
        self.logger.log_scalar("Avg_Reward", avg_r, update)
        self.logger.log_scalar("Raw_Reward", avg_r_raw, update)
        self.logger.log_scalar("Avg_Len", avg_l, update)
        self.logger.log_scalar("Entropy_Coeff", c_e, update)
        self.logger.log_scalar("Actor_LR",
                               self.actor_optimizer.param_groups[0]["lr"], update)
        self.logger.log_scalar("Critic_LR",
                               self.critic_optimizer.param_groups[0]["lr"], update)

    def _log_status(self, update, metrics, avg_r, avg_l, elapsed):
        window = self.n_envs * 4
        avg_r_raw = (
            float(np.mean(self._completed_rewards_raw[-window:]))
            if self._completed_rewards_raw else 0.0
        )
        print(
            f"[PPO] iter {update:4d}/{self.n_updates} | "
            f"P:{metrics['p_loss']:7.4f} | "
            f"V:{metrics['v_loss']:8.2f} | "
            f"Ent:{metrics['entropy']:.3f} | "
            f"KL:{metrics['kl_div']:.4f} | "
            f"EV:{metrics['explained_var']:.3f} | "
            f"R(norm):{avg_r:8.3f} | "
            f"R(raw):{avg_r_raw:8.1f} | "
            f"t:{elapsed:.1f}s"
        )

    # ------------------------------------------------------------------
    # Post-update: checkpointing + best-model tracking
    # ------------------------------------------------------------------

    def _on_update_end(self, update, metrics, avg_r, avg_l):
        # Periodic checkpoint
        if update % self.save_freq == 0:
            self.save(os.path.join(self.save_dir, f"ppo_update_{update}.pt"))

        # Best-model gate: only real episodes with negative reward
        window = self.n_envs * 4
        avg_r_raw = (
            float(np.mean(self._completed_rewards_raw[-window:]))
            if self._completed_rewards_raw else 0.0
        )
        if (
            self._completed_rewards_raw
            and avg_r_raw < 0
            and avg_r_raw > self._best_reward
        ):
            self._best_reward = avg_r_raw
            self.save(os.path.join(self.save_dir, "ppo_best.pt"))

    def _on_training_end(self, history: Dict):
        print(f"[PPO] Best raw reward: {self._best_reward:.2f}")

    # ------------------------------------------------------------------
    # Save / Load (extend base with separate optimizer states)
    # ------------------------------------------------------------------

    def save(self, path: str):
        payload = {
            "model_state_dict": self.model.state_dict(),
            "actor_optimizer_state": self.actor_optimizer.state_dict(),
            "critic_optimizer_state": self.critic_optimizer.state_dict(),
        }
        if self.reward_normalizer is not None:
            payload["rms_mean"] = float(self.reward_normalizer.mean)
            payload["rms_var"] = float(self.reward_normalizer.var)
            payload["rms_count"] = float(self.reward_normalizer.count)
        torch.save(payload, path)
        print(f"[PPO] Saved checkpoint -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "actor_optimizer_state" in ckpt:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state"])
        if "critic_optimizer_state" in ckpt:
            self.critic_optimizer.load_state_dict(ckpt["critic_optimizer_state"])
        if self.reward_normalizer is not None and "rms_mean" in ckpt:
            self.reward_normalizer.mean = np.float64(ckpt["rms_mean"])
            self.reward_normalizer.var = np.float64(ckpt["rms_var"])
            self.reward_normalizer.count = np.float64(ckpt["rms_count"])
        print(f"[PPO] Loaded checkpoint <- {path}")

    # ------------------------------------------------------------------
    # TorchScript export
    # ------------------------------------------------------------------

    def export_torchscript(self, path: str):
        self.model.eval()
        example_obs = torch.zeros(1, self.obs_dim, device=self.device)
        traced = torch.jit.trace_module(
            self.model, inputs={"get_value": example_obs},
        )
        traced.save(path)
        print(f"[PPO] TorchScript exported -> {path}")
        self.model.train()
