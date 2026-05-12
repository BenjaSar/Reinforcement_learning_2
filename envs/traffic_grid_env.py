"""
MultiIntersectionEnv — Phase 1
================================
Pure-Python 4x4 grid traffic environment.

State space (per intersection, 9 features):
  - lane_queue_lengths  : 4 floats  (normalized by capacity)
  - phase_one_hot       : 4 floats  (current active phase)
  - time_since_change   : 1 float   (normalized by max_phase_time)
Total: 16 intersections x 9 = 144-dimensional flat vector

Action space:
  - MultiDiscrete([4]*16): select next phase for each intersection
  - Phase changes shorter than min_green are masked (no-op penalty)

Reward:
  r_t = -(sum_i queue_i(t)) / N_lanes  -  0.1 * sum_i waiting_i(t) / N_lanes
  No reward for phase changes < min_green_time (oscillation penalty enforced).

Vehicle arrivals: independent Poisson process per lane, parametrized by
  demand_factor in [0, 1] (supports curriculum learning).
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_SIZE        = 4            # 4x4 grid → 16 intersections
N_INTERSECTIONS  = GRID_SIZE * GRID_SIZE   # 16
N_PHASES         = 4            # discrete signal phases per intersection
N_LANES          = 4            # incoming lanes per intersection
OBS_PER_INTER    = N_LANES + N_PHASES + 1  # 4 queue + 4 phase_onehot + 1 time = 9
OBS_DIM          = N_INTERSECTIONS * OBS_PER_INTER   # 144
LANE_CAPACITY    = 20.0         # max vehicles per lane (for normalization)
MAX_PHASE_TIME   = 120.0        # seconds (for time normalization)
MIN_GREEN        = 5            # minimum green time steps before phase switch
DEFAULT_EPISODE_STEPS = 720     # ~1 hour at 5s per step


class MultiIntersectionEnv(gym.Env):
    """
    4x4 grid of traffic intersections.
    Single centralized agent with a factored action space (16 x Discrete(4)).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        demand_factor: float = 1.0,
        episode_steps: int = DEFAULT_EPISODE_STEPS,
        min_green: int = MIN_GREEN,
        seed: Optional[int] = None,
        reward_shaping: bool = True,
    ):
        super().__init__()
        self.demand_factor   = demand_factor
        self.episode_steps   = episode_steps
        self.min_green       = min_green
        self.reward_shaping  = reward_shaping

        # Observation: 144-dim continuous [0, 1]^144
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # Action: 16 independent Discrete(4)
        self.action_space = spaces.MultiDiscrete([N_PHASES] * N_INTERSECTIONS)

        # Internal state arrays  (shape: [N_INTERSECTIONS, N_LANES])
        self._queues:         np.ndarray = np.zeros((N_INTERSECTIONS, N_LANES), dtype=np.float32)
        self._waiting:        np.ndarray = np.zeros((N_INTERSECTIONS, N_LANES), dtype=np.float32)
        self._current_phase:  np.ndarray = np.zeros(N_INTERSECTIONS, dtype=np.int32)
        self._time_in_phase:  np.ndarray = np.zeros(N_INTERSECTIONS, dtype=np.int32)
        self._step_count:     int = 0

        # Base arrival rates per lane per step (vehicles/step at demand_factor=1)
        # Modeled as lambda for Poisson; typical urban rate ~0.3 veh/step/lane
        self._base_arrival_rates = np.full((N_INTERSECTIONS, N_LANES), 0.3, dtype=np.float32)

        # RNG
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._queues        = np.zeros((N_INTERSECTIONS, N_LANES), dtype=np.float32)
        self._waiting       = np.zeros((N_INTERSECTIONS, N_LANES), dtype=np.float32)
        self._current_phase = np.zeros(N_INTERSECTIONS, dtype=np.int32)
        self._time_in_phase = np.zeros(N_INTERSECTIONS, dtype=np.int32)
        self._step_count    = 0

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        action: int array of shape (N_INTERSECTIONS,), values in [0, N_PHASES).
        """
        action = np.asarray(action, dtype=np.int32)

        # 1. Phase transition (enforce min_green)
        for i in range(N_INTERSECTIONS):
            desired_phase = int(action[i])
            if desired_phase != self._current_phase[i]:
                if self._time_in_phase[i] >= self.min_green:
                    self._current_phase[i] = desired_phase
                    self._time_in_phase[i] = 0
                # else: phase change is blocked (no-op, time keeps accumulating)
            self._time_in_phase[i] += 1

        # 2. Vehicle arrivals (Poisson)
        lam = self._base_arrival_rates * self.demand_factor
        arrivals = self._rng.poisson(lam).astype(np.float32)
        self._queues = np.clip(self._queues + arrivals, 0.0, LANE_CAPACITY)

        # 3. Vehicle departures: active phase clears vehicles from 2 of 4 lanes
        for i in range(N_INTERSECTIONS):
            phase = self._current_phase[i]
            # Each phase services 2 lanes (phases 0-3 service lane pairs 0-1, 1-2, 2-3, 3-0)
            served_lanes = [phase % N_LANES, (phase + 1) % N_LANES]
            for lane in served_lanes:
                departed = min(self._queues[i, lane], 2.0)   # ~2 veh/step service rate
                self._queues[i, lane] -= departed

        self._queues = np.clip(self._queues, 0.0, LANE_CAPACITY)

        # 4. Update waiting times: all queued vehicles accumulate 1 step waiting
        self._waiting = np.where(self._queues > 0, self._waiting + 1.0, 0.0)

        # 5. Reward
        total_lanes = N_INTERSECTIONS * N_LANES
        queue_penalty   = self._queues.sum() / total_lanes
        waiting_penalty = self._waiting.sum() / total_lanes
        reward = -queue_penalty - 0.1 * waiting_penalty

        # Optional dense shaping: small bonus for any queue reduction
        if self.reward_shaping:
            pass  # shaping handled externally via delta reward in curriculum

        self._step_count += 1
        terminated = False
        truncated  = self._step_count >= self.episode_steps

        obs  = self._get_obs()
        info = self._get_info()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass  # no rendering needed

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for i in range(N_INTERSECTIONS):
            base = i * OBS_PER_INTER
            # Queue lengths normalized [0, 1]
            obs[base : base + N_LANES] = self._queues[i] / LANE_CAPACITY
            # Phase one-hot
            phase_oh = np.zeros(N_PHASES, dtype=np.float32)
            phase_oh[self._current_phase[i]] = 1.0
            obs[base + N_LANES : base + N_LANES + N_PHASES] = phase_oh
            # Time in phase normalized [0, 1]
            obs[base + N_LANES + N_PHASES] = min(
                self._time_in_phase[i] / MAX_PHASE_TIME, 1.0
            )
        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "mean_queue":   float(self._queues.mean()),
            "max_queue":    float(self._queues.max()),
            "mean_waiting": float(self._waiting.mean()),
            "step":         self._step_count,
            "demand_factor": self.demand_factor,
        }

    # ------------------------------------------------------------------
    # Convenience setters (used by curriculum scheduler)
    # ------------------------------------------------------------------

    def set_demand_factor(self, factor: float):
        self.demand_factor = float(np.clip(factor, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_env(
    demand_factor: float = 1.0,
    episode_steps: int = DEFAULT_EPISODE_STEPS,
    seed: Optional[int] = None,
    reward_shaping: bool = True,
):
    """Return a callable that creates a fresh MultiIntersectionEnv."""
    def _init():
        env = MultiIntersectionEnv(
            demand_factor=demand_factor,
            episode_steps=episode_steps,
            seed=seed,
            reward_shaping=reward_shaping,
        )
        return env
    return _init


def make_vec_env(
    n_envs: int = 16,
    demand_factor: float = 1.0,
    episode_steps: int = DEFAULT_EPISODE_STEPS,
    base_seed: int = 0,
    reward_shaping: bool = True,
) -> gym.vector.SyncVectorEnv:
    """
    Create a SyncVectorEnv with n_envs independent MultiIntersectionEnv instances.
    Each instance gets a unique seed derived from base_seed.
    """
    env_fns = [
        make_env(
            demand_factor=demand_factor,
            episode_steps=episode_steps,
            seed=base_seed + i,
            reward_shaping=reward_shaping,
        )
        for i in range(n_envs)
    ]
    return gym.vector.SyncVectorEnv(env_fns)
