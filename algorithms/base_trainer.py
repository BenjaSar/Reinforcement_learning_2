"""
ActorCriticTrainer — Abstract base for on-policy actor-critic trainers.
========================================================================
Provides shared rollout collection, episode tracking, checkpointing,
and a callback hook system. Subclasses override algorithm-specific parts
(advantage computation, model update, metric logging).

Usage:
    class MyTrainer(ActorCriticTrainer):
        def _create_buffer(self) -> RolloutBuffer: ...
        def _update_model(self, update, c_e) -> dict: ...
        def _compute_advantages(self, last_value, last_done): ...
        # ... remaining abstract methods
"""

import os
import time
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from algorithms.rollout_buffer import RolloutBuffer
from utils.running_stats import RunningMeanStd
from utils.logger import MetricLogger


class TrainingCallback:
    """
    Hook interface for training lifecycle events.
    Subclass and override the relevant methods.
    """

    def on_training_start(self, trainer):
        pass

    def on_update_start(self, trainer, update: int):
        pass

    def on_update_end(self, trainer, update: int, metrics: dict):
        pass

    def on_checkpoint(self, trainer, path: str):
        pass

    def on_training_end(self, trainer, history: dict):
        pass


class CurriculumCallback(TrainingCallback):
    """
    Built-in callback that implements 3-stage curriculum learning.
    Replaces the duplicated training loop in train_ppo.py.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_update_start(self, trainer, update: int):
        demand = self.scheduler.get_demand_factor()
        trainer.env.call("set_demand_factor", demand)

    def on_update_end(self, trainer, update: int, metrics: dict):
        avg_r = metrics.get("avg_reward", 0.0)
        self.scheduler.step(avg_r)
        trainer.logger.log_scalar("Demand_Factor", self.scheduler.current_demand, update)


class ActorCriticTrainer(ABC):
    """
    Abstract base for synchronous on-policy actor-critic trainers.

    Shared responsibilities:
      - Environment interaction and rollout collection
      - Episode reward/length tracking (normalized and optional raw)
      - TensorBoard logging via MetricLogger
      - Checkpoint save/load with RunningMeanStd persistence
      - Callback management for curriculum, checkpoint scheduling, etc.

    Subclasses must implement:
      _create_buffer()     -> RolloutBuffer configured for the algorithm
      _compute_advantages() -> advantage estimation (GAE, n-step, etc.)
      _update_model()      -> loss computation + gradient step
      _init_history()      -> history dict
      _update_history()    -> populate history dict each iteration
      _log_metrics()       -> TensorBoard logging per update
      _log_status()        -> console print per update
    """

    def __init__(
        self,
        env,
        model,
        n_steps: int = 128,
        n_updates: int = 200,
        gamma: float = 0.99,
        c_v: float = 0.5,
        c_e: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        log_dir: str = "results",
        save_dir: str = "checkpoints",
        normalize_rewards: bool = False,
        track_raw_rewards: bool = False,
        algo_name: str = "ActorCritic",
    ):
        self.env = env
        self.model = model.to(device)
        self.n_steps = n_steps
        self.n_updates = n_updates
        self.gamma = gamma
        self.c_v = c_v
        self.c_e = c_e
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        self.save_dir = save_dir
        self.algo_name = algo_name

        # Infer dimensions
        self.n_envs = env.num_envs
        self.obs_dim = env.single_observation_space.shape[0]
        self.n_intersections = env.single_action_space.nvec.shape[0]

        # Algorithm-specific components
        self.buffer = self._create_buffer()
        self.reward_normalizer = (
            RunningMeanStd(shape=()) if normalize_rewards else None
        )
        self.logger = MetricLogger(log_dir=log_dir, algo_name=algo_name)

        # Episode tracking: normalized rewards (always)
        self._ep_rewards = np.zeros(self.n_envs, dtype=np.float32)
        self._ep_lengths = np.zeros(self.n_envs, dtype=np.int32)
        self._completed_rewards: List[float] = []
        self._completed_lengths: List[int] = []

        # Optional raw reward tracking (for evaluation comparison)
        self.track_raw_rewards = track_raw_rewards
        if track_raw_rewards:
            self._ep_rewards_raw = np.zeros(self.n_envs, dtype=np.float32)
            self._completed_rewards_raw: List[float] = []

        # Best-model tracking (raw reward when available, else normalized)
        self._best_reward = -np.inf
        self._callbacks: List[TrainingCallback] = []

    # ------------------------------------------------------------------
    # Template method: shared training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict:
        history = self._init_history()

        obs_np, _ = self.env.reset()
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        done_np = np.zeros(self.n_envs, dtype=np.float32)

        self._on_training_start()
        self._invoke_callbacks("on_training_start", self)

        print(f"[{self.algo_name}] Starting: {self.n_updates} updates x "
              f"{self.n_steps} steps x {self.n_envs} envs")
        t0 = time.time()

        for update in range(1, self.n_updates + 1):
            c_e = self._get_entropy_coeff(update)
            self.buffer.reset()
            self._invoke_callbacks("on_update_start", self, update)

            # --- Rollout collection ---
            for _ in range(self.n_steps):
                with torch.no_grad():
                    actions, log_probs, values = self.model.act(obs)

                obs_next_np, rewards_np, term_np, trunc_np, _ = self.env.step(
                    actions.cpu().numpy()
                )
                done_np = np.logical_or(term_np, trunc_np).astype(np.float32)

                raw_rewards = rewards_np.copy()

                if self.reward_normalizer is not None:
                    self.reward_normalizer.update(rewards_np)
                    rewards_np = rewards_np / (self.reward_normalizer.std + 1e-8)

                # Episode tracking
                self._ep_rewards += rewards_np
                self._ep_lengths += 1
                if self.track_raw_rewards:
                    self._ep_rewards_raw += raw_rewards

                for i, d in enumerate(done_np):
                    if d:
                        self._completed_rewards.append(float(self._ep_rewards[i]))
                        self._completed_lengths.append(int(self._ep_lengths[i]))
                        self._ep_rewards[i] = 0.0
                        self._ep_lengths[i] = 0
                        if self.track_raw_rewards:
                            self._completed_rewards_raw.append(
                                float(self._ep_rewards_raw[i])
                            )
                            self._ep_rewards_raw[i] = 0.0

                self.buffer.add(
                    obs=obs,
                    actions=actions,
                    rewards=torch.tensor(rewards_np, dtype=torch.float32),
                    values=values,
                    log_probs=log_probs,
                    dones=torch.tensor(done_np, dtype=torch.float32),
                )
                obs = torch.tensor(obs_next_np, dtype=torch.float32, device=self.device)

            # --- Bootstrap + advantages ---
            with torch.no_grad():
                last_value = self.model.get_value(obs)
            self._compute_advantages(
                last_value, torch.tensor(done_np, dtype=torch.float32)
            )
            self.buffer.normalize_advantages()

            # --- Model update ---
            metrics = self._update_model(update, c_e)

            # --- Scheduler step ---
            self._step_schedulers()

            # --- Aggregate window metrics ---
            window = self.n_envs * 4
            avg_r = (
                float(np.mean(self._completed_rewards[-window:]))
                if self._completed_rewards else 0.0
            )
            avg_l = (
                float(np.mean(self._completed_lengths[-window:]))
                if self._completed_lengths else 0.0
            )

            # --- History + logging ---
            history = self._update_history(history, metrics, update, c_e, avg_r, avg_l)
            self._log_metrics(metrics, update, c_e, avg_r, avg_l)

            if update % 10 == 0 or update == 1:
                self._log_status(update, metrics, avg_r, avg_l, time.time() - t0)

            # --- Post-update hooks ---
            self._on_update_end(update, metrics, avg_r, avg_l)
            self._invoke_callbacks("on_update_end", self, update, metrics)

        self.logger.flush()
        self._on_training_end(history)
        self._invoke_callbacks("on_training_end", self, history)
        print(f"[{self.algo_name}] Training complete in {time.time()-t0:.1f}s")
        return history

    # ------------------------------------------------------------------
    # Abstract methods — subclass must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_buffer(self) -> RolloutBuffer:
        """Create and return an empty RolloutBuffer."""

    @abstractmethod
    def _compute_advantages(
        self, last_value: torch.Tensor, last_done: torch.Tensor
    ):
        """Compute advantages and set buffer.returns / buffer.advantages."""

    @abstractmethod
    def _update_model(self, update: int, c_e: float) -> Dict:
        """
        Perform the gradient update step(s).

        Returns:
            dict of scalar metrics for this update (p_loss, v_loss, entropy, ...)
        """

    @abstractmethod
    def _init_history(self) -> Dict:
        """Return an empty history dict with the expected keys."""

    @abstractmethod
    def _update_history(
        self, history: Dict, metrics: Dict, update: int,
        c_e: float, avg_r: float, avg_l: float
    ) -> Dict:
        """Append current-iteration values to the history dict."""

    @abstractmethod
    def _log_metrics(
        self, metrics: Dict, update: int, c_e: float,
        avg_r: float, avg_l: float
    ):
        """Log metrics to TensorBoard."""

    @abstractmethod
    def _log_status(
        self, update: int, metrics: Dict, avg_r: float,
        avg_l: float, elapsed: float
    ):
        """Print a status line to console."""

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def _get_entropy_coeff(self, update: int) -> float:
        """Override for entropy decay schedule (PPO)."""
        return self.c_e

    def _on_training_start(self):
        """Override for warmup phase (PPO)."""

    def _on_update_end(self, update: int, metrics: Dict, avg_r: float, avg_l: float):
        """Override for best-model tracking, checkpointing (PPO)."""

    def _on_training_end(self, history: Dict):
        """Override for finalization."""

    def _step_schedulers(self):
        """Override for LR scheduler step (PPO)."""

    # ------------------------------------------------------------------
    # Callback management
    # ------------------------------------------------------------------

    def add_callback(self, callback: TrainingCallback):
        self._callbacks.append(callback)

    def _invoke_callbacks(self, method: str, *args):
        for cb in self._callbacks:
            getattr(cb, method)(*args)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(self.save_dir, exist_ok=True)
        payload = {"model_state_dict": self.model.state_dict()}
        if self.reward_normalizer is not None:
            payload["rms_mean"] = float(self.reward_normalizer.mean)
            payload["rms_var"] = float(self.reward_normalizer.var)
            payload["rms_count"] = float(self.reward_normalizer.count)
        torch.save(payload, path)
        print(f"[{self.algo_name}] Saved checkpoint -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if self.reward_normalizer is not None and "rms_mean" in ckpt:
            self.reward_normalizer.mean = np.float64(ckpt["rms_mean"])
            self.reward_normalizer.var = np.float64(ckpt["rms_var"])
            self.reward_normalizer.count = np.float64(ckpt["rms_count"])
        print(f"[{self.algo_name}] Loaded checkpoint <- {path}")
