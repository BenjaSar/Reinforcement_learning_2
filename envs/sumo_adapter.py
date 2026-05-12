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
    _SUMO_AVAILABLE = True
except ImportError:
    _SUMO_AVAILABLE = False


class SumoGridEnv(gym.Env):
    """
    Thin adapter wrapping sumo_rl.parallel_env into a single-agent
    gymnasium.Env with the same 144-dim observation / MultiDiscrete(4^16)
    action interface as MultiIntersectionEnv.

    Requires:
        pip install sumo-rl
        SUMO_HOME environment variable set to SUMO installation directory.
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
        )
        self._agents = None

        # Mirror the same spaces as MultiIntersectionEnv
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(144,), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete(
            [self.N_PHASES] * self.N_INTERSECTIONS
        )

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        obs_dict, info_dict = self._pz_env.reset(seed=seed, options=options)
        self._agents = list(obs_dict.keys())
        obs = self._flatten_obs(obs_dict)
        return obs, {}

    def step(self, action):
        action_dict = {
            agent: int(action[i])
            for i, agent in enumerate(self._agents)
        }
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = (
            self._pz_env.step(action_dict)
        )
        obs        = self._flatten_obs(obs_dict)
        reward     = float(sum(rew_dict.values()))
        terminated = all(term_dict.values())
        truncated  = all(trunc_dict.values())
        return obs, reward, terminated, truncated, {}

    def close(self):
        self._pz_env.close()

    # ------------------------------------------------------------------
    def _flatten_obs(self, obs_dict: dict) -> np.ndarray:
        """Concatenate per-agent observations into a single flat vector."""
        parts = []
        for agent in self._agents:
            agent_obs = obs_dict.get(agent, np.zeros(9, dtype=np.float32))
            # sumo-rl obs may have different dim; pad/truncate to 9
            agent_obs = np.array(agent_obs, dtype=np.float32)
            if len(agent_obs) < 9:
                agent_obs = np.pad(agent_obs, (0, 9 - len(agent_obs)))
            else:
                agent_obs = agent_obs[:9]
            parts.append(agent_obs)
        # Pad to exactly 16 agents if fewer are active
        while len(parts) < self.N_INTERSECTIONS:
            parts.append(np.zeros(9, dtype=np.float32))
        return np.concatenate(parts[:self.N_INTERSECTIONS]).astype(np.float32)
