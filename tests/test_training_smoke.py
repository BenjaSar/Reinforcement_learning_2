"""Integration smoke tests: full training loops with minimal settings."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.traffic_grid_env import make_vec_env
from models.actor_critic import SharedActorCritic
from algorithms.a2c import A2CTrainer
from algorithms.ppo import PPOTrainer
from algorithms.base_trainer import CurriculumCallback
from utils.curriculum import CurriculumScheduler
from utils import set_seed


def test_a2c_smoke():
    """A2C training: 5 updates, 2 envs, 8 steps — verify completion."""
    set_seed(42)
    env = make_vec_env(n_envs=2, base_seed=42)
    model = SharedActorCritic()
    trainer = A2CTrainer(
        env=env, model=model,
        n_steps=8, n_updates=5, device="cpu",
    )
    history = trainer.train()
    env.close()
    assert "p_loss" in history
    assert "v_loss" in history
    assert "entropy" in history
    assert "avg_reward" in history
    assert len(history["p_loss"]) == 5


def test_ppo_smoke():
    """PPO training: 5 updates, 2 envs, 8 steps — verify completion."""
    set_seed(42)
    env = make_vec_env(n_envs=2, base_seed=42)
    model = SharedActorCritic()
    trainer = PPOTrainer(
        env=env, model=model,
        n_steps=8, n_updates=5,
        warmup_steps=64,  # small warmup for speed
        device="cpu",
    )
    history = trainer.train()
    env.close()
    assert "p_loss" in history
    assert "v_loss" in history
    assert "entropy" in history
    assert "kl_div" in history
    assert "avg_reward" in history
    assert "avg_reward_raw" in history
    assert len(history["p_loss"]) == 5


def test_ppo_curriculum_smoke():
    """PPO with curriculum callback: verify no crash."""
    set_seed(42)
    env = make_vec_env(n_envs=2, base_seed=42, demand_factor=0.3)
    model = SharedActorCritic()
    trainer = PPOTrainer(
        env=env, model=model,
        n_steps=8, n_updates=5,
        warmup_steps=64,
        device="cpu",
    )
    curriculum = CurriculumScheduler(
        stage_demands=[0.3, 0.6, 1.0],
        stage_thresholds=[-5.5, -3.5],
        update_budgets=[100, 100, 200],
    )
    trainer.add_callback(CurriculumCallback(curriculum))
    history = trainer.train()
    env.close()
    assert "p_loss" in history


def test_checkpoint_save_load(tmp_path):
    """Verify save and load produce identical model weights."""
    set_seed(42)
    from envs.traffic_grid_env import make_vec_env
    from models.actor_critic import SharedActorCritic

    env = make_vec_env(n_envs=2, base_seed=42)
    model = SharedActorCritic()
    trainer = PPOTrainer(
        env=env, model=model,
        n_steps=8, n_updates=2,
        warmup_steps=0,  # skip warmup for speed
        device="cpu",
        save_dir=str(tmp_path),
    )
    _ = trainer.train()
    ckpt_path = os.path.join(str(tmp_path), "test.pt")
    trainer.save(ckpt_path)
    assert os.path.exists(ckpt_path)

    # Load into fresh trainer
    fresh_env = make_vec_env(n_envs=2, base_seed=42)
    fresh_model = SharedActorCritic()
    fresh_trainer = PPOTrainer(
        env=fresh_env, model=fresh_model,
        n_steps=8, n_updates=1,
        warmup_steps=0,
        device="cpu",
        save_dir=str(tmp_path),
    )
    fresh_trainer.load(ckpt_path)

    # Weights should match
    for p1, p2 in zip(trainer.model.parameters(), fresh_trainer.model.parameters()):
        assert p1.cpu().data.equal(p2.cpu().data)

    fresh_env.close()
    env.close()
