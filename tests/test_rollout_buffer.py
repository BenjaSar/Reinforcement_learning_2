"""Unit tests for RolloutBuffer."""

import torch
from algorithms.rollout_buffer import RolloutBuffer


def make_buffer(n_steps=4, n_envs=2, obs_dim=144, n_intersections=16):
    return RolloutBuffer(
        n_steps=n_steps,
        n_envs=n_envs,
        obs_dim=obs_dim,
        n_intersections=n_intersections,
        gamma=0.99,
        gae_lambda=0.95,
    )


def test_initial_state():
    buf = make_buffer()
    assert buf._ptr == 0
    assert not buf._full
    assert buf.obs.shape == (4, 2, 144)


def test_add():
    buf = make_buffer()
    obs = torch.randn(2, 144)
    actions = torch.randint(0, 4, (2, 16))
    rewards = torch.randn(2)
    values = torch.randn(2, 1)
    log_probs = torch.randn(2, 16)
    dones = torch.zeros(2)

    buf.add(obs, actions, rewards, values, log_probs, dones)
    assert buf._ptr == 1
    assert not buf._full


def test_reset():
    buf = make_buffer()
    buf.add(
        torch.randn(2, 144), torch.randint(0, 4, (2, 16)),
        torch.randn(2), torch.randn(2, 1),
        torch.randn(2, 16), torch.zeros(2),
    )
    buf.reset()
    assert buf._ptr == 0
    assert not buf._full


def test_compute_gae():
    buf = make_buffer(n_steps=4, n_envs=2)
    # Fill buffer
    for t in range(4):
        buf.add(
            torch.randn(2, 144), torch.randint(0, 4, (2, 16)),
            torch.randn(2), torch.randn(2, 1),
            torch.randn(2, 16), torch.zeros(2),
        )
    last_value = torch.randn(2, 1)
    last_done = torch.zeros(2)
    buf.compute_gae(last_value, last_done)
    assert buf.advantages.shape == (4, 2)
    assert buf.returns.shape == (4, 2)


def test_compute_nstep_returns():
    buf = make_buffer(n_steps=4, n_envs=2)
    for t in range(4):
        buf.add(
            torch.randn(2, 144), torch.randint(0, 4, (2, 16)),
            torch.ones(2) * (-1.0), torch.randn(2, 1),
            torch.randn(2, 16), torch.zeros(2),
        )
    last_value = torch.randn(2, 1)
    last_done = torch.zeros(2)
    buf.compute_nstep_returns(last_value, last_done)
    assert buf.advantages.shape == (4, 2)
    assert buf.returns.shape == (4, 2)


def test_normalize_advantages():
    buf = make_buffer(n_steps=4, n_envs=2)
    for t in range(4):
        buf.add(
            torch.randn(2, 144), torch.randint(0, 4, (2, 16)),
            torch.randn(2), torch.randn(2, 1),
            torch.randn(2, 16), torch.zeros(2),
        )
    buf.advantages = torch.randn(4, 2) * 3.0 + 5.0
    orig_mean = buf.advantages.mean()
    buf.normalize_advantages()
    assert abs(float(buf.advantages.mean())) < 1e-6
    assert abs(float(buf.advantages.std()) - 1.0) < 1e-5


def test_get_minibatches():
    buf = make_buffer(n_steps=4, n_envs=4)
    for t in range(4):
        buf.add(
            torch.randn(4, 144), torch.randint(0, 4, (4, 16)),
            torch.randn(4), torch.randn(4, 1),
            torch.randn(4, 16), torch.zeros(4),
        )
    buf.returns = torch.randn(4, 4)
    buf.advantages = torch.randn(4, 4)
    buf.values = torch.randn(4, 4)
    buf.log_probs = torch.randn(4, 4, 16)

    batches = list(buf.get_minibatches(batch_size=4))
    assert len(batches) == 4  # 16 flattened steps / 4 batch_size
    obs_b, act_b, lp_b, adv_b, ret_b, val_b = batches[0]
    assert obs_b.shape == (4, 144)
    assert act_b.shape == (4, 16)
    assert adv_b.shape == (4,)
    assert ret_b.shape == (4,)
    assert val_b.shape == (4,)
