"""Unit tests for CurriculumScheduler."""

from utils.curriculum import CurriculumScheduler


def test_initial_state():
    sched = CurriculumScheduler()
    assert sched.current_stage == 0
    assert sched.current_demand == 0.3


def test_not_advanced_without_real_data():
    """No advancement when avg_reward is 0.0 (sentinel for no episodes)."""
    sched = CurriculumScheduler(
        stage_thresholds=[-5.0, -3.0],
        update_budgets=[150, 150, 200],
    )
    advanced = sched.step(0.0)
    assert not advanced
    assert sched.current_stage == 0


def test_advance_by_threshold():
    sched = CurriculumScheduler(
        stage_thresholds=[-5.0, -3.0],
        update_budgets=[150, 150, 200],
        window=10,
    )
    advanced = sched.step(-3.0)   # -3.0 >= -5.0, and non-zero
    assert advanced
    assert sched.current_stage == 1
    assert sched.current_demand == 0.6


def test_advance_by_budget():
    sched = CurriculumScheduler(
        stage_thresholds=[-100.0, -3.0],
        update_budgets=[3, 150, 200],
        window=10,
    )
    # Fill with real (non-zero) rewards but below threshold
    for _ in range(3):
        sched.step(-10.0)
    # After 3 updates, budget for stage 0 is exhausted
    assert sched.current_stage == 1
    assert sched.current_demand == 0.6


def test_stage_3_is_final():
    sched = CurriculumScheduler()
    # Force to stage 3
    for _ in range(2):
        sched.step(-1.0)
    # Now at stage 3 (0-indexed: 2)
    if sched.current_stage < 2:
        sched.step(-1.0)
    # Should not advance past stage 3
    advanced = sched.step(-1.0)
    assert not advanced


def test_get_demand_factor():
    sched = CurriculumScheduler()
    assert sched.get_demand_factor() == 0.3  # stage 0 = 30%


def test_custom_params():
    sched = CurriculumScheduler(
        stage_demands=[0.5, 0.75, 1.0],
        stage_thresholds=[-2.0, -1.0],
        update_budgets=[50, 50, 100],
    )
    assert sched.current_demand == 0.5
    sched.step(-1.0)
    assert sched.current_demand == 0.75
