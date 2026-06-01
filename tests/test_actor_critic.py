"""Unit tests for SharedActorCritic."""

import torch
from models.actor_critic import SharedActorCritic


def test_forward_shapes():
    model = SharedActorCritic()
    obs = torch.randn(4, 144)
    distributions, value = model(obs)
    assert len(distributions) == 16
    assert value.shape == (4, 1)


def test_forward_batched():
    model = SharedActorCritic()
    obs = torch.randn(8, 144)
    _, value = model(obs)
    assert value.shape == (8, 1)


def test_forward_single():
    model = SharedActorCritic()
    obs = torch.randn(144)
    distributions, value = model(obs)
    assert len(distributions) == 16
    assert value.shape == (1,)


def test_act():
    model = SharedActorCritic()
    obs = torch.randn(4, 144)
    actions, log_probs, value = model.act(obs)
    assert actions.shape == (4, 16)
    assert log_probs.shape == (4, 16)
    assert value.shape == (4, 1)


def test_evaluate_actions():
    model = SharedActorCritic()
    obs = torch.randn(4, 144)
    actions = torch.randint(0, 4, (4, 16))
    log_probs, value, entropy = model.evaluate_actions(obs, actions)
    assert log_probs.shape == (4, 16)
    assert value.shape == (4, 1)
    assert entropy.ndim == 0  # scalar


def test_get_value():
    model = SharedActorCritic()
    obs = torch.randn(4, 144)
    value = model.get_value(obs)
    assert value.shape == (4, 1)
