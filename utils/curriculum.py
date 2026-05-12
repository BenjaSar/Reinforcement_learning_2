"""
CurriculumScheduler — Phase 5
================================
Three-stage curriculum learning for traffic demand.

Stage 1: 30% demand  → train until avg reward passes threshold1
Stage 2: 60% demand  → train until avg reward passes threshold2
Stage 3: 100% demand → final stage until n_updates exhausted

The scheduler is queried each PPO update cycle and returns the current
demand_factor to pass to the vectorized environment instances.

Usage:
    scheduler = CurriculumScheduler(
        stage_thresholds=[threshold1, threshold2],
        update_budgets=[budget1, budget2, budget3],
    )

    for update in range(n_updates):
        demand = scheduler.get_demand_factor(avg_reward, update)
        env.call("set_demand_factor", demand)   # broadcast to all envs
        ...
        scheduler.step(avg_reward)
"""

import numpy as np
from typing import List


class CurriculumScheduler:
    """
    Three-stage curriculum demand scheduler.

    Args:
        stage_demands     : demand factor for each stage  [0.3, 0.6, 1.0]
        stage_thresholds  : avg reward thresholds to advance from stage 1->2, 2->3
                            Set to None to use only update_budgets
        update_budgets    : max updates to spend in each stage before forced advance
                            [budget_s1, budget_s2, budget_s3]
        window            : rolling window size for avg reward evaluation
    """

    STAGES = 3

    def __init__(
        self,
        stage_demands:    List[float] = None,
        stage_thresholds: List[float] = None,
        update_budgets:   List[int]   = None,
        window:           int         = 50,
    ):
        self.stage_demands    = stage_demands    or [0.3, 0.6, 1.0]
        self.stage_thresholds = stage_thresholds or [-5.0, -3.0]   # per-stage advance
        self.update_budgets   = update_budgets   or [150, 150, 200]
        self.window           = window

        assert len(self.stage_demands)    == self.STAGES
        assert len(self.stage_thresholds) == self.STAGES - 1
        assert len(self.update_budgets)   == self.STAGES

        self._stage         = 0       # 0-indexed
        self._stage_updates = 0       # updates spent in current stage
        self._reward_history: list = []

    # ------------------------------------------------------------------
    @property
    def current_stage(self) -> int:
        return self._stage

    @property
    def current_demand(self) -> float:
        return self.stage_demands[self._stage]

    # ------------------------------------------------------------------
    def step(self, avg_reward: float) -> bool:
        """
        Record avg_reward and check if the curriculum should advance.

        Returns True if the stage just advanced.
        """
        self._reward_history.append(avg_reward)
        self._stage_updates += 1

        if self._stage >= self.STAGES - 1:
            return False   # Already at final stage

        # Rolling average
        window_rewards = self._reward_history[-self.window:]
        rolling_avg = float(np.mean(window_rewards)) if window_rewards else avg_reward

        # Advance condition: threshold met OR budget exhausted
        threshold = self.stage_thresholds[self._stage]
        budget    = self.update_budgets[self._stage]

        advanced = False
        if rolling_avg >= threshold or self._stage_updates >= budget:
            self._stage += 1
            self._stage_updates = 0
            print(
                f"[Curriculum] Advanced to Stage {self._stage + 1} "
                f"(demand={self.current_demand:.0%}) "
                f"| rolling_avg={rolling_avg:.3f}, threshold={threshold}"
            )
            advanced = True

        return advanced

    # ------------------------------------------------------------------
    def get_demand_factor(self) -> float:
        return self.current_demand

    def __repr__(self) -> str:
        return (
            f"CurriculumScheduler(stage={self._stage + 1}/{self.STAGES}, "
            f"demand={self.current_demand:.0%}, "
            f"stage_updates={self._stage_updates})"
        )
