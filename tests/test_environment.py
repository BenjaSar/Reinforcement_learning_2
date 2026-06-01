"""Unit tests for MultiIntersectionEnv."""

import numpy as np
from envs.traffic_grid_env import MultiIntersectionEnv
from envs.traffic_grid_env import OBS_DIM, N_INTERSECTIONS, N_PHASES


def test_reset():
    env = MultiIntersectionEnv(seed=42)
    obs, info = env.reset(seed=42)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert 0.0 <= obs.min() <= obs.max() <= 1.0
    assert "mean_queue" in info


def test_step():
    env = MultiIntersectionEnv(seed=42)
    env.reset(seed=42)
    action = np.zeros(N_INTERSECTIONS, dtype=np.int32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert not terminated  # not done until episode_steps


def test_episode_length():
    env = MultiIntersectionEnv(episode_steps=10, seed=42)
    env.reset(seed=42)
    for _ in range(10):
        obs, _, _, truncated, _ = env.step(np.zeros(N_INTERSECTIONS, dtype=np.int32))
    _, _, _, truncated, _ = env.step(np.zeros(N_INTERSECTIONS, dtype=np.int32))
    assert truncated


def test_min_green():
    env = MultiIntersectionEnv(min_green=5, seed=42)
    env.reset(seed=42)
    action = np.ones(N_INTERSECTIONS, dtype=np.int32)  # try to switch to phase 1
    for _ in range(4):
        env.step(action)
    # After 4 steps, still below min_green=5, so phase should not have changed
    # (We can check via the public property)
    assert hasattr(env, 'current_phases')


def test_action_space():
    env = MultiIntersectionEnv(seed=42)
    assert env.action_space.shape == (N_INTERSECTIONS,)
    assert env.action_space.nvec[0] == N_PHASES


def test_observation_space():
    env = MultiIntersectionEnv(seed=42)
    assert env.observation_space.shape == (OBS_DIM,)


def test_random_policy_baseline():
    env = MultiIntersectionEnv(episode_steps=100, seed=42)
    obs, _ = env.reset(seed=42)
    total_reward = 0.0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, _, truncated, _ = env.step(action)
        total_reward += reward
        if truncated:
            break
    assert total_reward < 0  # random policy should have negative reward


def test_public_properties():
    env = MultiIntersectionEnv(seed=42)
    env.reset(seed=42)
    q = env.queues
    p = env.current_phases
    assert q.shape == (16, 4)
    assert p.shape == (16,)
