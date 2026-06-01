"""Unit tests for RunningMeanStd."""

import numpy as np
from utils.running_stats import RunningMeanStd


RTOL = 1e-3  # Welford alg with epsilon init has small bias on small samples


def test_initial_state():
    rms = RunningMeanStd(shape=())
    assert rms.mean == 0.0
    assert rms.var == 1.0
    assert rms.count > 0
    assert rms.std > 0.0


def test_update_single_value():
    rms = RunningMeanStd(shape=())
    rms.update(np.array([5.0]))
    assert abs(rms.mean - 5.0) < RTOL, f"mean={rms.mean}"
    assert rms.count > 1


def test_update_batch():
    rms = RunningMeanStd(shape=())
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rms.update(data)
    assert abs(rms.mean - 3.0) < RTOL, f"mean={rms.mean}"
    assert abs(rms.var - 2.0) < RTOL, f"var={rms.var}"


def test_multiple_batches():
    rms = RunningMeanStd(shape=())
    rms.update(np.array([0.0, 10.0]))
    rms.update(np.array([20.0, 30.0]))
    assert abs(rms.mean - 15.0) < RTOL, f"mean={rms.mean}"
    assert rms.count == 4 + 1e-4  # count = 4 + epsilon


def test_normalize():
    rms = RunningMeanStd(shape=())
    rms.update(np.array([0.0, 10.0]))
    normed = rms.normalize(np.array([5.0]))
    assert abs(normed[0]) < RTOL, f"normed={normed}"


def test_std_property():
    rms = RunningMeanStd(shape=())
    rms.update(np.array([0.0, 4.0]))
    expected_std = 2.0  # population std of {0, 4}
    assert abs(float(rms.std) - expected_std) < RTOL, f"std={rms.std}"


def test_convergence():
    rms = RunningMeanStd(shape=())
    rng = np.random.default_rng(42)
    data = rng.normal(loc=10.0, scale=2.0, size=10000)
    rms.update(data)
    assert abs(rms.mean - 10.0) < 0.2
    assert abs(float(np.sqrt(rms.var)) - 2.0) < 0.2
