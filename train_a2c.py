"""
train_a2c.py — Phase 2 entry point
=====================================
Trains the baseline A2C agent on the MultiIntersectionEnv.

Usage:
    python train_a2c.py [--n_updates 200] [--n_steps 128] [--n_envs 16]
                        [--gamma 0.99] [--lr 3e-4] [--c_v 0.5] [--c_e 0.01]
                        [--device cpu] [--seed 0]

Outputs:
    - TensorBoard logs in results/a2c/
    - Checkpoints in checkpoints/a2c/
    - Training curve plot in results/a2c/training_curves.png
    - history.npy in results/a2c/
"""

import argparse
import sys
import os
import numpy as np
import torch

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.traffic_grid_env import make_vec_env
from models.actor_critic import SharedActorCritic
from algorithms.a2c import A2CTrainer
from utils.logger import MetricLogger


def parse_args():
    p = argparse.ArgumentParser(description="Train A2C baseline")
    p.add_argument("--n_updates",   type=int,   default=200)
    p.add_argument("--n_steps",     type=int,   default=128)
    p.add_argument("--n_envs",      type=int,   default=16)
    p.add_argument("--gamma",       type=float, default=0.99)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--c_v",         type=float, default=0.5)
    p.add_argument("--c_e",         type=float, default=0.01)
    p.add_argument("--grad_clip",   type=float, default=0.5)
    p.add_argument("--device",      type=str,   default="cpu")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--demand",      type=float, default=1.0,
                   help="Traffic demand factor [0-1]")
    p.add_argument("--normalize_rewards", action="store_true", default=False)
    return p.parse_args()


def random_policy_baseline(n_episodes: int = 10, seed: int = 0) -> dict:
    """
    Run a random policy for n_episodes and report baseline statistics.
    Returns dict with mean_reward, mean_queue, mean_waiting.
    """
    from envs.traffic_grid_env import MultiIntersectionEnv
    env = MultiIntersectionEnv(seed=seed)
    rewards, queues, waitings = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_queues, ep_waitings = [], []
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            ep_queues.append(info["mean_queue"])
            ep_waitings.append(info["mean_waiting"])
            done = term or trunc
        rewards.append(ep_reward)
        queues.append(np.mean(ep_queues))
        waitings.append(np.mean(ep_waitings))

    result = {
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "mean_queue":   float(np.mean(queues)),
        "mean_waiting": float(np.mean(waitings)),
    }
    print("\n[Baseline] Random Policy (10 episodes):")
    for k, v in result.items():
        print(f"  {k:20s}: {v:.4f}")
    return result


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  Phase 2 — Baseline A2C Training")
    print("=" * 60)
    print(f"  n_updates : {args.n_updates}")
    print(f"  n_steps   : {args.n_steps}")
    print(f"  n_envs    : {args.n_envs}")
    print(f"  gamma     : {args.gamma}")
    print(f"  lr        : {args.lr}")
    print(f"  device    : {args.device}")
    print(f"  seed      : {args.seed}")
    print("=" * 60)

    # --- Random policy baseline ---
    baseline = random_policy_baseline(n_episodes=10, seed=args.seed)

    # --- Environment ---
    vec_env = make_vec_env(
        n_envs        = args.n_envs,
        demand_factor = args.demand,
        base_seed     = args.seed,
    )

    # --- Model ---
    model = SharedActorCritic()

    # --- Trainer ---
    trainer = A2CTrainer(
        env               = vec_env,
        model             = model,
        n_steps           = args.n_steps,
        n_updates         = args.n_updates,
        gamma             = args.gamma,
        lr                = args.lr,
        c_v               = args.c_v,
        c_e               = args.c_e,
        max_grad_norm     = args.grad_clip,
        device            = args.device,
        log_dir           = "results/a2c",
        save_dir          = "checkpoints/a2c",
        normalize_rewards = args.normalize_rewards,
    )

    # --- Train ---
    history = trainer.train()

    # --- Save history ---
    os.makedirs("results/a2c", exist_ok=True)
    np.save("results/a2c/history.npy", history)

    # --- Plots ---
    logger = trainer.logger
    logger.plot_training_curves(
        save_path = "results/a2c/training_curves.png",
        metrics   = ["P_Loss", "V_Loss", "Entropy", "Avg_Reward", "Avg_Len"],
        smooth    = 10,
    )

    # --- Final checkpoint ---
    trainer.save("checkpoints/a2c/a2c_final.pt")

    # --- Summary ---
    final_reward = history["avg_reward"][-1] if history["avg_reward"] else 0.0
    print("\n" + "=" * 60)
    print("  A2C Training Summary")
    print("=" * 60)
    print(f"  Random baseline reward : {baseline['mean_reward']:.4f}")
    print(f"  A2C final reward       : {final_reward:.4f}")
    improvement = final_reward - baseline["mean_reward"]
    print(f"  Improvement            : {improvement:+.4f}")
    print(f"  TensorBoard logs       : results/a2c/")
    print(f"  Checkpoint             : checkpoints/a2c/a2c_final.pt")
    print("=" * 60)

    vec_env.close()
    return history, baseline


if __name__ == "__main__":
    main()
