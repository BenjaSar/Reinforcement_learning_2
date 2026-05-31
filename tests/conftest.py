"""Shared fixtures for tests."""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from envs.traffic_grid_env import MultiIntersectionEnv
from models.actor_critic import SharedActorCritic


@pytest.fixture
def env():
    return MultiIntersectionEnv(seed=42)


@pytest.fixture
def vec_env():
    from envs.traffic_grid_env import make_vec_env
    return make_vec_env(n_envs=2, base_seed=42)


@pytest.fixture
def model():
    return SharedActorCritic()


@pytest.fixture
def obs(env):
    obs, _ = env.reset(seed=42)
    return obs
