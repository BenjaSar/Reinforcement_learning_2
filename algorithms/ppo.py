"""
PPOTrainer — Phase 3
======================
Full PPO-GAE implementation with all improvements over the A2C baseline:

  1. PPO Clipped Surrogate Objective (epsilon=0.2)
  2. Generalized Advantage Estimation (GAE, lambda=0.95)
  3. Reward Normalization via RunningMeanStd
  4. Gradient Clipping (max_norm=0.5)
  5. Entropy Coefficient Linear Decay (c_e: 0.01 -> 0.001)
  6. K=4 PPO update epochs with mini-batches
  7. Early stopping per epoch if approx KL > kl_target
  8. Explained variance tracking
  9. Model checkpointing every save_freq updates
  10. TensorBoard metric logging (all Section 7 metrics)

Training loop (one iteration):
  1. Collect N=n_steps*n_envs transitions
  2. Bootstrap last value; compute GAE
  3. Normalize advantages
  4. K PPO epochs over mini-batches of size batch_size
  5. Update theta_old <- theta for next rollout
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
    PPO trainer with GAE, reward normalization, entropy decay, and checkpointing.

    Args:
        env              : vectorized gymnasium env (SyncVectorEnv, n_envs instances)
        model            : SharedActorCritic
        n_steps          : steps per env per rollout (total = n_steps * n_envs)
        n_updates        : total PPO update cycles (default 500)
        gamma            : discount factor
        gae_lambda       : GAE lambda (0.95 = balanced bias/variance)
        clip_epsilon     : PPO clipping threshold
        lr               : Adam initial learning rate
        lr_decay         : apply linear lr decay to 0 over n_updates
        c_v              : value loss coefficient
        c_e_start        : initial entropy coefficient
        c_e_end          : final entropy coefficient (linear decay)
        n_epochs         : PPO update epochs per rollout (K)
        batch_size       : mini-batch size within PPO update
        max_grad_norm    : gradient clipping max norm
        kl_target        : approx KL threshold for early epoch stopping
        normalize_rewards: apply RunningMeanStd reward normalization
        device           : torch device string
        log_dir          : TensorBoard log directory
        save_dir         : checkpoint directory
        save_freq        : save checkpoint every N updates
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
        lr:                float = 3e-4,
        lr_decay:          bool  = True,
        c_v:               float = 0.5,
        c_e_start:         float = 0.01,
        c_e_end:           float = 0.001,
        n_epochs:          int   = 4,
        batch_size:        int   = 256,
        max_grad_norm:     float = 0.5,
        kl_target:         float = 0.02,
        normalize_rewards: bool  = True,
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
        self.lr              = lr
        self.lr_decay        = lr_decay
        self.c_v             = c_v
        self.c_e_start       = c_e_start
        self.c_e_end         = c_e_end
        self.n_epochs        = n_epochs
        self.batch_size      = batch_size
        self.max_grad_norm   = max_grad_norm
        self.kl_target       = kl_target
        self.device          = torch.device(device)
        self.save_dir        = save_dir
        self.save_freq       = save_freq

        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        if lr_decay:
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=n_updates,
            )
        else:
            self.scheduler = None

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

        self.reward_normalizer = (
            RunningMeanStd(shape=()) if normalize_rewards else None
        )

        self.logger = MetricLogger(log_dir=log_dir, algo_name="PPO")

        # Episode tracking
        self._ep_rewards: np.ndarray = np.zeros(self.n_envs, dtype=np.float32)
        self._ep_lengths: np.ndarray = np.zeros(self.n_envs, dtype=np.int32)
        self._completed_rewards: list = []
        self._completed_lengths: list = []

        # Best model tracking
        self._best_reward = -np.inf

    # ------------------------------------------------------------------
    def _get_entropy_coeff(self, update: int) -> float:
        """Linear decay of entropy coefficient."""
        frac = (update - 1) / max(self.n_updates - 1, 1)
        return self.c_e_start + frac * (self.c_e_end - self.c_e_start)

    # ------------------------------------------------------------------
    def train(self) -> Dict:
        """
        Run the full PPO training loop.

        Returns:
            history dict with all tracked metrics.
        """
        history = {
            "p_loss":     [],
            "v_loss":     [],
            "entropy":    [],
            "kl_div":     [],
            "grad_norm":  [],
            "explained_var": [],
            "avg_reward": [],
            "avg_len":    [],
            "c_e":        [],
        }

        obs_np, _ = self.env.reset()
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        done_np = np.zeros(self.n_envs, dtype=np.float32)

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

                # Reward normalization
                if self.reward_normalizer is not None:
                    self.reward_normalizer.update(rewards_np)
                    rewards_np = rewards_np / (self.reward_normalizer.std + 1e-8)

                # Episode stats
                self._ep_rewards += rewards_np
                self._ep_lengths += 1
                for i, done in enumerate(done_np):
                    if done:
                        self._completed_rewards.append(float(self._ep_rewards[i]))
                        self._completed_lengths.append(int(self._ep_lengths[i]))
                        self._ep_rewards[i] = 0.0
                        self._ep_lengths[i] = 0

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
            # Step 4: PPO update (K epochs, mini-batches)
            # ================================================================
            metrics = self._ppo_update(c_e)

            # Learning rate scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # ================================================================
            # Step 5: Logging
            # ================================================================
            window = self.n_envs * 4
            avg_r = (float(np.mean(self._completed_rewards[-window:]))
                     if self._completed_rewards else 0.0)
            avg_l = (float(np.mean(self._completed_lengths[-window:]))
                     if self._completed_lengths else 0.0)

            history["p_loss"].append(metrics["p_loss"])
            history["v_loss"].append(metrics["v_loss"])
            history["entropy"].append(metrics["entropy"])
            history["kl_div"].append(metrics["kl_div"])
            history["grad_norm"].append(metrics["grad_norm"])
            history["explained_var"].append(metrics["explained_var"])
            history["avg_reward"].append(avg_r)
            history["avg_len"].append(avg_l)
            history["c_e"].append(c_e)

            for k, v in metrics.items():
                self.logger.log_scalar(k, v, update)
            self.logger.log_scalar("Avg_Reward",    avg_r, update)
            self.logger.log_scalar("Avg_Len",       avg_l, update)
            self.logger.log_scalar("Entropy_Coeff", c_e,   update)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.log_scalar("LR", current_lr, update)

            if update % 10 == 0 or update == 1:
                elapsed = time.time() - t0
                print(
                    f"[PPO] iter {update:4d}/{self.n_updates} | "
                    f"P_Loss: {metrics['p_loss']:7.4f} | "
                    f"V_Loss: {metrics['v_loss']:7.4f} | "
                    f"Entropy: {metrics['entropy']:.4f} | "
                    f"KL: {metrics['kl_div']:.4f} | "
                    f"ExplVar: {metrics['explained_var']:.3f} | "
                    f"Avg R: {avg_r:7.3f} | "
                    f"c_e: {c_e:.5f} | "
                    f"t: {elapsed:.1f}s"
                )

            # Checkpointing
            if update % self.save_freq == 0:
                ckpt_path = os.path.join(self.save_dir, f"ppo_update_{update}.pt")
                self.save(ckpt_path)

            if avg_r > self._best_reward and self._completed_rewards:
                self._best_reward = avg_r
                best_path = os.path.join(self.save_dir, "ppo_best.pt")
                self.save(best_path)

        self.logger.flush()
        print(f"[PPO] Training complete in {time.time()-t0:.1f}s | "
              f"Best avg reward: {self._best_reward:.3f}")
        return history

    # ------------------------------------------------------------------
    def _ppo_update(self, c_e: float) -> Dict:
        """
        K epochs of mini-batch PPO updates.
        Returns aggregated metrics dict.
        """
        p_losses, v_losses, entropies, kl_divs, grad_norms = [], [], [], [], []

        # For explained variance (compute once before update)
        returns_flat = self.buffer.returns.reshape(-1)
        values_flat  = self.buffer.values.reshape(-1)
        ev = _explained_variance(values_flat.numpy(), returns_flat.numpy())

        for _ in range(self.n_epochs):
            epoch_kl = 0.0
            n_batches = 0

            for batch in self.buffer.get_minibatches(self.batch_size):
                obs_b, act_b, old_lp_b, adv_b, ret_b, _ = batch

                # New policy evaluation
                new_log_probs, values_new, entropy = self.model.evaluate_actions(
                    obs_b, act_b
                )

                # Probability ratio  r_t(theta)
                # old_lp_b: (B, N_INTERSECTIONS), sum over intersections for joint prob
                old_log_prob = old_lp_b.sum(dim=-1)           # (B,)
                new_log_prob = new_log_probs.sum(dim=-1)       # (B,)
                ratio = torch.exp(new_log_prob - old_log_prob) # (B,)

                # Clipped surrogate
                adv_b = adv_b.to(self.device)
                ret_b = ret_b.to(self.device)

                surr1 = ratio * adv_b
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(
                    values_new.squeeze(-1), ret_b
                )

                # Total loss
                loss = policy_loss + self.c_v * value_loss - c_e * entropy

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Approx KL divergence
                with torch.no_grad():
                    approx_kl = ((old_log_prob - new_log_prob)
                                 .mean()
                                 .abs()
                                 .item())

                p_losses.append(float(policy_loss.item()))
                v_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                kl_divs.append(approx_kl)
                grad_norms.append(float(grad_norm))

                epoch_kl += approx_kl
                n_batches += 1

            # Early stopping if KL is too high
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
    def save(self, path: str):
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"[PPO] Saved checkpoint -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[PPO] Loaded checkpoint <- {path}")

    def export_torchscript(self, path: str):
        """Export the policy as TorchScript for deployment (Phase 5)."""
        self.model.eval()
        example_obs = torch.zeros(1, self.obs_dim, device=self.device)
        # Use trace_module to correctly trace a module method
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
    """
    Fraction of variance in returns explained by the value function.
    EV = 1 - Var(returns - values) / Var(returns)
    """
    var_returns = np.var(returns)
    if var_returns < 1e-8:
        return 0.0
    return float(1.0 - np.var(returns - values) / var_returns)

