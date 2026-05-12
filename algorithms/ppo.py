"""
PPOTrainer — v2 (Fixed & Improved)
====================================
All fixes and algorithmic improvements applied:

BUGS FIXED
  F1. RunningMeanStd persisted in checkpoint (save/load)
  F2. Raw episode reward tracked & logged separately from normalized reward
  F3. Best-model gate: only save when at least one real episode has completed
      and avg_r < 0 (guards against the 0.0 > -inf false trigger)

ALGORITHMIC IMPROVEMENTS
  A1. Entropy coefficients raised: c_e_start=0.05, c_e_end=0.01
      Prevents entropy collapse before the critic has converged
  A2. PPO-style value function clipping added to _ppo_update()
      Prevents the critic from making destructively large updates per mini-batch
  A3. Separate Adam optimizers for actor trunk/heads vs. critic head
      actor_lr=3e-4, critic_lr=1e-3 — critic learns faster early on
  A4. Reward normalizer warmup: 512 random-action steps collected before
      first gradient update so RunningMeanStd is pre-seeded

STRUCTURAL IMPROVEMENTS
  S1. LR decay floor raised to lr * 0.05 (not 0) — avoids full LR shutdown
  S2. Raw (un-normalized) episode rewards logged to TensorBoard as Raw_Reward
  S3. Periodic checkpoint default changed to ppo_update_500.pt for eval

Training loop (one iteration):
  1. [Warmup, update=1 only] 512 random steps to seed RunningMeanStd
  2. Collect N=n_steps*n_envs transitions with current policy
  3. Bootstrap last value; compute GAE(lambda=0.95)
  4. Normalize advantages (zero mean, unit variance)
  5. K PPO epochs with VF-clipped update + actor/critic split optimizer
  6. Log raw and normalized rewards; checkpoint if improved
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict

from algorithms.rollout_buffer import RolloutBuffer
from utils.running_stats import RunningMeanStd
from utils.logger import MetricLogger


class PPOTrainer:
    """
    Fixed and improved PPO trainer.

    Key changes vs. v1:
      - Separate optimizers: actor_lr / critic_lr
      - Value function clipping (clip_epsilon reused for VF clip)
      - RunningMeanStd saved in checkpoint and restored at load
      - Best-model gate guards against 0.0 > -inf false positive
      - Raw episode reward tracked and logged separately
      - Warmup phase seeds reward normalizer before first gradient update
    """

    def __init__(
        self,
        env,
        model,
        n_steps:           int   = 128,
        n_updates:         int   = 500,
        gamma:             float = 0.99,
        gae_lambda:        float = 0.95,
        clip_epsilon:      float = 0.2,
        actor_lr:          float = 3e-4,
        critic_lr:         float = 1e-3,
        lr_decay:          bool  = True,
        lr_min_frac:       float = 0.05,
        c_v:               float = 0.5,
        c_e_start:         float = 0.05,
        c_e_end:           float = 0.01,
        n_epochs:          int   = 4,
        batch_size:        int   = 256,
        max_grad_norm:     float = 0.5,
        kl_target:         float = 0.02,
        normalize_rewards: bool  = True,
        warmup_steps:      int   = 512,
        device:            str   = "cpu",
        log_dir:           str   = "results/ppo",
        save_dir:          str   = "checkpoints/ppo",
        save_freq:         int   = 50,
    ):
        self.env             = env
        self.model           = model.to(device)
        self.n_steps         = n_steps
        self.n_updates       = n_updates
        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.clip_epsilon    = clip_epsilon
        self.actor_lr        = actor_lr
        self.critic_lr       = critic_lr
        self.lr_decay        = lr_decay
        self.lr_min_frac     = lr_min_frac
        self.c_v             = c_v
        self.c_e_start       = c_e_start
        self.c_e_end         = c_e_end
        self.n_epochs        = n_epochs
        self.batch_size      = batch_size
        self.max_grad_norm   = max_grad_norm
        self.kl_target       = kl_target
        self.warmup_steps    = warmup_steps
        self.device          = torch.device(device)
        self.save_dir        = save_dir
        self.save_freq       = save_freq
        self.normalize_rewards = normalize_rewards

        os.makedirs(save_dir, exist_ok=True)

        # --- A3: Separate optimizers for actor vs. critic ---
        actor_params = (
            list(model.trunk.parameters())
            + [p for head in model.actor_heads for p in head.parameters()]
        )
        critic_params = list(model.critic.parameters())

        self.actor_optimizer  = optim.Adam(actor_params,  lr=actor_lr,  eps=1e-5)
        self.critic_optimizer = optim.Adam(critic_params, lr=critic_lr, eps=1e-5)

        # --- S1: LR scheduler with min-fraction floor ---
        if lr_decay:
            self.actor_scheduler = optim.lr_scheduler.LinearLR(
                self.actor_optimizer,
                start_factor=1.0,
                end_factor=lr_min_frac,
                total_iters=n_updates,
            )
            self.critic_scheduler = optim.lr_scheduler.LinearLR(
                self.critic_optimizer,
                start_factor=1.0,
                end_factor=lr_min_frac,
                total_iters=n_updates,
            )
        else:
            self.actor_scheduler  = None
            self.critic_scheduler = None

        # Infer dimensions
        self.n_envs          = env.num_envs
        self.obs_dim         = env.single_observation_space.shape[0]
        self.n_intersections = env.single_action_space.nvec.shape[0]

        self.buffer = RolloutBuffer(
            n_steps         = n_steps,
            n_envs          = self.n_envs,
            obs_dim         = self.obs_dim,
            n_intersections = self.n_intersections,
            gamma           = gamma,
            gae_lambda      = gae_lambda,
            device          = device,
        )

        # --- F1: RunningMeanStd (will be included in checkpoints) ---
        self.reward_normalizer = (
            RunningMeanStd(shape=()) if normalize_rewards else None
        )

        self.logger = MetricLogger(log_dir=log_dir, algo_name="PPO")

        # Episode tracking (normalized rewards for training signal)
        self._ep_rewards_norm: np.ndarray = np.zeros(self.n_envs, dtype=np.float32)
        self._ep_rewards_raw:  np.ndarray = np.zeros(self.n_envs, dtype=np.float32)
        self._ep_lengths:      np.ndarray = np.zeros(self.n_envs, dtype=np.int32)

        self._completed_rewards_norm: list = []
        self._completed_rewards_raw:  list = []
        self._completed_lengths:      list = []

        # --- F3: Best-model tracking (only triggered by real negative rewards) ---
        self._best_reward = -np.inf

    # ------------------------------------------------------------------
    def _get_entropy_coeff(self, update: int) -> float:
        """Linear decay of entropy coefficient."""
        frac = (update - 1) / max(self.n_updates - 1, 1)
        return self.c_e_start + frac * (self.c_e_end - self.c_e_start)

    # ------------------------------------------------------------------
    def _warmup_normalizer(self, obs: torch.Tensor) -> torch.Tensor:
        """
        A4: Collect warmup_steps of random-action experience to seed
        RunningMeanStd before the first gradient update.
        Returns the updated obs tensor after warmup.
        """
        if self.reward_normalizer is None or self.warmup_steps <= 0:
            return obs

        print(f"[PPO] Warming up reward normalizer ({self.warmup_steps} steps)...")
        obs_np = obs.cpu().numpy()
        for _ in range(self.warmup_steps):
            actions_np = np.array([
                self.env.single_action_space.sample()
                for _ in range(self.n_envs)
            ])
            obs_next_np, rewards_np, term_np, trunc_np, _ = self.env.step(actions_np)
            self.reward_normalizer.update(rewards_np)
            done_np = np.logical_or(term_np, trunc_np)
            obs_np = obs_next_np

        print(f"[PPO] Normalizer seeded: mean={self.reward_normalizer.mean:.4f}, "
              f"std={float(self.reward_normalizer.std):.4f}")
        return torch.tensor(obs_np, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    def train(self) -> Dict:
        """
        Run the full fixed PPO training loop.

        Returns:
            history dict with normalized and raw reward tracks.
        """
        history = {
            "p_loss":        [],
            "v_loss":        [],
            "entropy":       [],
            "kl_div":        [],
            "grad_norm":     [],
            "explained_var": [],
            "avg_reward":    [],      # normalized episode reward
            "avg_reward_raw":[],      # S2: raw un-normalized episode reward
            "avg_len":       [],
            "c_e":           [],
        }

        obs_np, _ = self.env.reset()
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        done_np = np.zeros(self.n_envs, dtype=np.float32)

        # A4: Warmup reward normalizer
        obs = self._warmup_normalizer(obs)

        print(f"[PPO] Starting training: {self.n_updates} updates x "
              f"{self.n_steps} steps x {self.n_envs} envs "
              f"(total={self.n_updates * self.n_steps * self.n_envs:,} steps)")
        t0 = time.time()

        for update in range(1, self.n_updates + 1):
            c_e = self._get_entropy_coeff(update)
            self.buffer.reset()

            # ================================================================
            # Step 1: Rollout collection
            # ================================================================
            for _ in range(self.n_steps):
                with torch.no_grad():
                    actions, log_probs, values = self.model.act(obs)

                actions_np = actions.cpu().numpy()
                obs_next_np, rewards_np, term_np, trunc_np, _ = (
                    self.env.step(actions_np)
                )
                done_np = np.logical_or(term_np, trunc_np).astype(np.float32)

                raw_rewards = rewards_np.copy()  # S2: keep raw copy

                # F1/A4: Normalize rewards using seeded RunningMeanStd
                if self.reward_normalizer is not None:
                    self.reward_normalizer.update(rewards_np)
                    rewards_np = rewards_np / (self.reward_normalizer.std + 1e-8)

                # Episode stats: track both normalized and raw
                self._ep_rewards_norm += rewards_np
                self._ep_rewards_raw  += raw_rewards
                self._ep_lengths      += 1

                for i, done in enumerate(done_np):
                    if done:
                        self._completed_rewards_norm.append(
                            float(self._ep_rewards_norm[i])
                        )
                        self._completed_rewards_raw.append(
                            float(self._ep_rewards_raw[i])
                        )
                        self._completed_lengths.append(int(self._ep_lengths[i]))
                        self._ep_rewards_norm[i] = 0.0
                        self._ep_rewards_raw[i]  = 0.0
                        self._ep_lengths[i]      = 0

                self.buffer.add(
                    obs       = obs,
                    actions   = actions,
                    rewards   = torch.tensor(rewards_np, dtype=torch.float32),
                    values    = values,
                    log_probs = log_probs,
                    dones     = torch.tensor(done_np, dtype=torch.float32),
                )
                obs = torch.tensor(obs_next_np, dtype=torch.float32,
                                   device=self.device)

            # ================================================================
            # Step 2: Bootstrap + GAE
            # ================================================================
            with torch.no_grad():
                last_value = self.model.get_value(obs)

            self.buffer.compute_gae(
                last_value = last_value,
                last_done  = torch.tensor(done_np, dtype=torch.float32),
            )

            # ================================================================
            # Step 3: Normalize advantages
            # ================================================================
            self.buffer.normalize_advantages()

            # ================================================================
            # Step 4: PPO update (K epochs, mini-batches, VF clipping)
            # ================================================================
            metrics = self._ppo_update(c_e)

            # LR scheduler step
            if self.actor_scheduler  is not None: self.actor_scheduler.step()
            if self.critic_scheduler is not None: self.critic_scheduler.step()

            # ================================================================
            # Step 5: Logging
            # ================================================================
            window = self.n_envs * 4
            avg_r_norm = (
                float(np.mean(self._completed_rewards_norm[-window:]))
                if self._completed_rewards_norm else 0.0
            )
            avg_r_raw = (
                float(np.mean(self._completed_rewards_raw[-window:]))
                if self._completed_rewards_raw else 0.0
            )
            avg_l = (
                float(np.mean(self._completed_lengths[-window:]))
                if self._completed_lengths else 0.0
            )

            history["p_loss"].append(metrics["p_loss"])
            history["v_loss"].append(metrics["v_loss"])
            history["entropy"].append(metrics["entropy"])
            history["kl_div"].append(metrics["kl_div"])
            history["grad_norm"].append(metrics["grad_norm"])
            history["explained_var"].append(metrics["explained_var"])
            history["avg_reward"].append(avg_r_norm)
            history["avg_reward_raw"].append(avg_r_raw)
            history["avg_len"].append(avg_l)
            history["c_e"].append(c_e)

            for k, v in metrics.items():
                self.logger.log_scalar(k, v, update)
            self.logger.log_scalar("Avg_Reward",     avg_r_norm, update)
            self.logger.log_scalar("Raw_Reward",     avg_r_raw,  update)  # S2
            self.logger.log_scalar("Avg_Len",        avg_l,      update)
            self.logger.log_scalar("Entropy_Coeff",  c_e,        update)
            self.logger.log_scalar("Actor_LR",
                                   self.actor_optimizer.param_groups[0]["lr"],
                                   update)
            self.logger.log_scalar("Critic_LR",
                                   self.critic_optimizer.param_groups[0]["lr"],
                                   update)

            if update % 10 == 0 or update == 1:
                elapsed = time.time() - t0
                print(
                    f"[PPO] iter {update:4d}/{self.n_updates} | "
                    f"P:{metrics['p_loss']:7.4f} | "
                    f"V:{metrics['v_loss']:8.2f} | "
                    f"Ent:{metrics['entropy']:.3f} | "
                    f"KL:{metrics['kl_div']:.4f} | "
                    f"EV:{metrics['explained_var']:.3f} | "
                    f"R(norm):{avg_r_norm:8.3f} | "
                    f"R(raw):{avg_r_raw:8.1f} | "
                    f"c_e:{c_e:.4f} | "
                    f"t:{elapsed:.1f}s"
                )

            # Periodic checkpoint
            if update % self.save_freq == 0:
                ckpt_path = os.path.join(self.save_dir, f"ppo_update_{update}.pt")
                self.save(ckpt_path)

            # --- F3: Best-model gate — only when real episodes have completed
            #     and reward is genuinely negative (not the 0.0 sentinel) ---
            if (
                self._completed_rewards_raw
                and avg_r_raw < 0
                and avg_r_raw > self._best_reward
            ):
                self._best_reward = avg_r_raw
                best_path = os.path.join(self.save_dir, "ppo_best.pt")
                self.save(best_path)

        self.logger.flush()
        print(f"[PPO] Training complete in {time.time()-t0:.1f}s | "
              f"Best raw reward: {self._best_reward:.2f}")
        return history

    # ------------------------------------------------------------------
    def _ppo_update(self, c_e: float) -> Dict:
        """
        K epochs of mini-batch PPO updates.

        Improvements vs. v1:
          A2: Value function clipping (PPO-style VF clip)
          A3: Separate backward passes through actor_optimizer / critic_optimizer
        """
        p_losses, v_losses, entropies, kl_divs, grad_norms = [], [], [], [], []

        # Explained variance computed once before any weight updates
        returns_flat = self.buffer.returns.reshape(-1)
        values_flat  = self.buffer.values.reshape(-1)
        ev = _explained_variance(values_flat.numpy(), returns_flat.numpy())

        for _ in range(self.n_epochs):
            epoch_kl = 0.0
            n_batches = 0

            for batch in self.buffer.get_minibatches(self.batch_size):
                obs_b, act_b, old_lp_b, adv_b, ret_b, old_val_b = batch

                adv_b     = adv_b.to(self.device)
                ret_b     = ret_b.to(self.device)
                old_val_b = old_val_b.to(self.device)

                # New policy evaluation
                new_log_probs, values_new, entropy = self.model.evaluate_actions(
                    obs_b, act_b
                )
                values_new = values_new.squeeze(-1)   # (B,)

                # --- Policy loss (clipped surrogate) ---
                old_log_prob = old_lp_b.sum(dim=-1)
                new_log_prob = new_log_probs.sum(dim=-1)
                ratio = torch.exp(new_log_prob - old_log_prob)

                surr1 = ratio * adv_b
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- A3: Separate actor / critic updates ---
                # ACTOR STEP — policy loss + entropy bonus
                # Detach value-related tensors so actor graph is self-contained
                actor_loss = policy_loss - c_e * entropy
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                actor_gn = nn.utils.clip_grad_norm_(
                    [p for pg in self.actor_optimizer.param_groups for p in pg["params"]],
                    self.max_grad_norm,
                )
                self.actor_optimizer.step()

                # CRITIC STEP — fresh forward pass through critic only
                # (trunk weights have just been updated by actor step)
                values_critic = self.model.get_value(obs_b).squeeze(-1)
                v_clipped_2   = old_val_b + torch.clamp(
                    values_critic - old_val_b,
                    -self.clip_epsilon,
                    +self.clip_epsilon,
                )
                crit_loss1 = nn.functional.mse_loss(values_critic, ret_b)
                crit_loss2 = nn.functional.mse_loss(v_clipped_2,   ret_b)
                value_loss = torch.max(crit_loss1, crit_loss2)

                self.critic_optimizer.zero_grad()
                (self.c_v * value_loss).backward()
                critic_gn = nn.utils.clip_grad_norm_(
                    [p for pg in self.critic_optimizer.param_groups for p in pg["params"]],
                    self.max_grad_norm,
                )
                self.critic_optimizer.step()

                grad_norm_combined = float((actor_gn**2 + critic_gn**2)**0.5)

                # Approx KL
                with torch.no_grad():
                    approx_kl = (old_log_prob - new_log_prob).mean().abs().item()

                p_losses.append(float(policy_loss.item()))
                v_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                kl_divs.append(approx_kl)
                grad_norms.append(grad_norm_combined)

                epoch_kl += approx_kl
                n_batches += 1

            # Early-stop epoch if KL too large
            if n_batches > 0 and (epoch_kl / n_batches) > self.kl_target:
                break

        return {
            "p_loss":        float(np.mean(p_losses)),
            "v_loss":        float(np.mean(v_losses)),
            "entropy":       float(np.mean(entropies)),
            "kl_div":        float(np.mean(kl_divs)),
            "grad_norm":     float(np.mean(grad_norms)),
            "explained_var": float(ev),
        }

    # ------------------------------------------------------------------
    # F1: Checkpoint now includes RunningMeanStd state
    # ------------------------------------------------------------------
    def save(self, path: str):
        payload = {
            "model_state_dict":          self.model.state_dict(),
            "actor_optimizer_state":     self.actor_optimizer.state_dict(),
            "critic_optimizer_state":    self.critic_optimizer.state_dict(),
        }
        if self.reward_normalizer is not None:
            payload["rms_mean"]  = float(self.reward_normalizer.mean)
            payload["rms_var"]   = float(self.reward_normalizer.var)
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
        # F1: Restore RunningMeanStd
        if self.reward_normalizer is not None and "rms_mean" in ckpt:
            self.reward_normalizer.mean  = np.float64(ckpt["rms_mean"])
            self.reward_normalizer.var   = np.float64(ckpt["rms_var"])
            self.reward_normalizer.count = np.float64(ckpt["rms_count"])
            print(f"[PPO] Restored RMS: mean={ckpt['rms_mean']:.4f}, "
                  f"std={float(self.reward_normalizer.std):.4f}")
        print(f"[PPO] Loaded checkpoint <- {path}")

    def export_torchscript(self, path: str):
        """Export the policy as TorchScript for deployment."""
        self.model.eval()
        example_obs = torch.zeros(1, self.obs_dim, device=self.device)
        traced = torch.jit.trace_module(
            self.model,
            inputs={"get_value": example_obs},
        )
        traced.save(path)
        print(f"[PPO] TorchScript exported -> {path}")
        self.model.train()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    var_returns = np.var(returns)
    if var_returns < 1e-8:
        return 0.0
    return float(1.0 - np.var(returns - values) / var_returns)


