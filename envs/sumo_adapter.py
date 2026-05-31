"""
Optional SUMO adapter — Phase 1 (bonus)
========================================
Drop-in replacement for MultiIntersectionEnv when SUMO is installed.

Usage:
    from envs.sumo_adapter import SumoGridEnv
    env = SumoGridEnv(net_file="...", route_file="...")

Falls back gracefully if sumo-rl is not installed.
"""

import numpy as np
import gymnasium as gym

try:
    import sumo_rl  # noqa: F401
    from sumo_rl.environment.observations import ObservationFunction
    from gymnasium import spaces
    _SUMO_AVAILABLE = True
except ImportError:
    ObservationFunction = object
    spaces = None
    _SUMO_AVAILABLE = False


class MatchingObservationFunction(ObservationFunction):
    """Produces 9-dim observation per agent matching MultiIntersectionEnv format.

    Pure-Python env format per intersection (9 values):
        queue_approach[0..3]  — normalized queue per approach direction
        phase_onehot[0..3]    — one-hot of current phase (0–3)
        phase_duration        — fraction of max duration elapsed
    """

    def __init__(self, ts):
        super().__init__(ts)
        self._max_queue = 20.0
        self._max_duration = 60.0

    def __call__(self):
        ts = self.ts
        # --- Queue: average 3 lanes per approach, normalize ---
        n_per = 3
        raw_q = ts.get_lanes_queue()
        queues = []
        for i in range(0, len(raw_q), n_per):
            queues.append(min(sum(raw_q[i:i + n_per]) / n_per, 1.0))

        # --- Phase: map 8 green phases -> 4 phases ---
        gp = ts.green_phase  # 0..7
        phase_idx = gp // 2  # 0,1,2,3
        phase_onehot = [0.0, 0.0, 0.0, 0.0]
        phase_onehot[phase_idx] = 1.0

        # --- Phase duration ---
        dur = min(ts.time_since_last_phase_change / self._max_duration, 1.0)

        obs = np.array(queues + phase_onehot + [dur], dtype=np.float32)
        return obs

    def observation_space(self):
        return spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)


class SumoGridEnv(gym.Env):
    """
    Thin adapter wrapping sumo_rl.parallel_env into a single-agent
    gymnasium.Env with the same 144-dim observation / MultiDiscrete(4^16)
    action interface as MultiIntersectionEnv.
    """

    N_INTERSECTIONS = 16
    N_PHASES        = 4

    def __init__(
        self,
        net_file: str,
        route_file: str,
        num_seconds: int = 3600,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        use_gui: bool = False,
    ):
        super().__init__()
        if not _SUMO_AVAILABLE:
            raise ImportError(
                "sumo-rl is not installed. "
                "Install with: pip install sumo-rl\n"
                "Also set SUMO_HOME to your SUMO installation directory."
            )

        self._pz_env = sumo_rl.parallel_env(
            net_file=net_file,
            route_file=route_file,
            use_gui=use_gui,
            num_seconds=num_seconds,
            delta_time=delta_time,
            yellow_time=yellow_time,
            min_green=min_green,
            observation_class=MatchingObservationFunction,
        )
        self._agents = None

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(144,), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete(
            [self.N_PHASES] * self.N_INTERSECTIONS
        )

    def reset(self, seed=None, options=None):
        result = self._pz_env.reset(seed=seed, options=options)
        if isinstance(result, tuple) and len(result) >= 1:
            obs_dict = result[0]
        else:
            obs_dict = result
        self._agents = sorted(obs_dict.keys()) if obs_dict else []
        obs = self._flatten_obs(obs_dict)
        return obs, {}

    def step(self, action):
        action_dict = {
            agent: int(action[i])
            for i, agent in enumerate(self._agents)
        }
        result = self._pz_env.step(action_dict)
        if isinstance(result, tuple) and len(result) >= 5:
            obs_dict, rew_dict, term_dict, trunc_dict = result[:4]
        else:
            obs_dict, rew_dict, term_dict, trunc_dict = result, {}, {}, {}
        obs        = self._flatten_obs(obs_dict or {})
        reward     = float(sum(rew_dict.values())) if rew_dict else 0.0
        terminated = all(term_dict.values()) if term_dict else False
        truncated  = all(trunc_dict.values()) if trunc_dict else False
        return obs, reward, terminated, truncated, {}

    def close(self):
        self._pz_env.close()

    def _flatten_obs(self, obs_dict: dict) -> np.ndarray:
        parts = []
        for agent in self._agents:
            agent_obs = obs_dict.get(agent, np.zeros(9, dtype=np.float32))
            agent_obs = np.array(agent_obs, dtype=np.float32).ravel()
            if len(agent_obs) != 9:
                agent_obs = np.zeros(9, dtype=np.float32) if len(agent_obs) != 9 else agent_obs
            parts.append(agent_obs)
        while len(parts) < self.N_INTERSECTIONS:
            parts.append(np.zeros(9, dtype=np.float32))
        return np.concatenate(parts[:self.N_INTERSECTIONS]).astype(np.float32)
