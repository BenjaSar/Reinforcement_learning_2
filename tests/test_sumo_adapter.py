"""Tests for the SUMO adapter (requires sumo-rl and SUMO binary)."""
import os, sys
import numpy as np

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.sumo_adapter import SumoGridEnv, _SUMO_AVAILABLE

skip_no_sumo = pytest.mark.skipif(
    not _SUMO_AVAILABLE,
    reason="sumo-rl not installed"
)

_SUMO_HOME = r"C:\Users\PC\AppData\Local\Temp\opencode\sumo_bin\sumo-1.21.0"
_HAS_SUMO = os.path.isdir(_SUMO_HOME)


def _make_env():
    import sumo_rl
    d = os.path.dirname(sumo_rl.__file__)
    net = os.path.join(d, "nets", "RESCO", "grid4x4", "grid4x4.net.xml")
    route = os.path.join(
        os.path.dirname(__file__), "..", "sumo", "routes", "grid4x4_03.rou.xml"
    )
    return SumoGridEnv(
        net_file=net, route_file=route,
        num_seconds=180, delta_time=5, use_gui=False,
    )


@skip_no_sumo
def test_reset():
    os.environ["SUMO_HOME"] = _SUMO_HOME
    env = _make_env()
    obs, info = env.reset()
    assert obs.shape == (144,)
    assert obs.dtype == np.float32
    assert (obs >= 0).all() and (obs <= 1).all()
    assert info == {}
    env.close()


@skip_no_sumo
def test_step():
    os.environ["SUMO_HOME"] = _SUMO_HOME
    env = _make_env()
    env.reset()
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)
    assert obs.shape == (144,)
    assert isinstance(r, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    env.close()


@skip_no_sumo
def test_multiple_steps():
    os.environ["SUMO_HOME"] = _SUMO_HOME
    env = _make_env()
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(action)
        if term or trunc:
            break
        assert obs.shape == (144,)
    env.close()


@skip_no_sumo
def test_obs_nine_per_agent():
    os.environ["SUMO_HOME"] = _SUMO_HOME
    env = _make_env()
    obs, _ = env.reset()
    for i in range(16):
        o9 = obs[i * 9:(i + 1) * 9]
        assert len(o9) == 9
        phase_sum = o9[4:8].sum()
        assert abs(phase_sum - 1.0) < 1e-5 or abs(phase_sum) < 1e-5
    env.close()


@skip_no_sumo
def test_action_space():
    os.environ["SUMO_HOME"] = _SUMO_HOME
    env = _make_env()
    assert env.action_space.shape == (16,)
    assert (env.action_space.nvec == 4).all()
    env.close()


@skip_no_sumo
def test_observation_space():
    os.environ["SUMO_HOME"] = _SUMO_HOME
    env = _make_env()
    assert env.observation_space.shape == (144,)
    env.close()


@skip_no_sumo
def test_random_policy():
    os.environ["SUMO_HOME"] = _SUMO_HOME
    env = _make_env()
    obs, _ = env.reset()
    total_r = 0.0
    for _ in range(36):
        action = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(action)
        total_r += r
        if term or trunc:
            break
    assert isinstance(total_r, float)
    env.close()
