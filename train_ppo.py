"""
train_ppo.py — Phase 3 entry point
=====================================
Trains the PPO-GAE agent with all improvements over A2C.

Usage:
    python train_ppo.py [--n_updates 500] [--n_steps 128] [--n_envs 16]
                        [--gamma 0.99] [--gae_lambda 0.95]
                        [--clip_epsilon 0.2] [--actor_lr 3e-4] [--critic_lr 1e-3]
                        [--c_v 0.5] [--c_e_start 0.05] [--c_e_end 0.01]
                        [--n_epochs 4] [--batch_size 256] [--grad_clip 0.5]
                        [--kl_target 0.02] [--device cpu] [--seed 0]
                        [--curriculum]

Outputs:
    - TensorBoard logs in results/ppo/
    - Checkpoints in checkpoints/ppo/ (every 50 updates + best model)
    - Training curve plot in results/ppo/training_curves.png
    - history.npy in results/ppo/
    - TorchScript export at checkpoints/ppo/policy_scripted.pt
"""

import argparse
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.traffic_grid_env import make_vec_env
from models.actor_critic import SharedActorCritic
from algorithms.ppo import PPOTrainer
from algorithms.base_trainer import CurriculumCallback
from utils.curriculum import CurriculumScheduler
from utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO-GAE agent (v3 refactored)")
    p.add_argument("--n_updates",       type=int,   default=500)
    p.add_argument("--n_steps",         type=int,   default=128)
    p.add_argument("--n_envs",          type=int,   default=16)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--gae_lambda",      type=float, default=0.95)
    p.add_argument("--clip_epsilon",    type=float, default=0.2)
    p.add_argument("--actor_lr",        type=float, default=3e-4)
    p.add_argument("--critic_lr",       type=float, default=1e-3)
    p.add_argument("--lr_decay",        action="store_true", default=True)
    p.add_argument("--lr_min_frac",     type=float, default=0.05)
    p.add_argument("--c_v",             type=float, default=0.5)
    p.add_argument("--c_e_start",       type=float, default=0.05)
    p.add_argument("--c_e_end",         type=float, default=0.01)
    p.add_argument("--n_epochs",        type=int,   default=4)
    p.add_argument("--batch_size",      type=int,   default=256)
    p.add_argument("--grad_clip",       type=float, default=0.5)
    p.add_argument("--kl_target",       type=float, default=0.02)
    p.add_argument("--warmup_steps",    type=int,   default=512)
    p.add_argument("--device",          type=str,   default="cpu")
    p.add_argument("--seed",            type=int,   default=0)
    p.add_argument("--curriculum",      action="store_true", default=False,
                   help="Enable curriculum learning (3-stage demand)")
    p.add_argument("--save_freq",       type=int,   default=50)
    p.add_argument("--no_norm_rewards", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("  Phase 3 — PPO-GAE Training")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k:20s}: {v}")
    print("=" * 60)

    initial_demand = 0.3 if args.curriculum else 1.0

    vec_env = make_vec_env(
        n_envs=args.n_envs,
        demand_factor=initial_demand,
        base_seed=args.seed,
    )

    model = SharedActorCritic()

    trainer = PPOTrainer(
        env=vec_env,
        model=model,
        n_steps=args.n_steps,
        n_updates=args.n_updates,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        lr_decay=args.lr_decay,
        lr_min_frac=args.lr_min_frac,
        c_v=args.c_v,
        c_e_start=args.c_e_start,
        c_e_end=args.c_e_end,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        max_grad_norm=args.grad_clip,
        kl_target=args.kl_target,
        warmup_steps=args.warmup_steps,
        normalize_rewards=not args.no_norm_rewards,
        device=args.device,
        log_dir="results/ppo",
        save_dir="checkpoints/ppo",
        save_freq=args.save_freq,
    )

    # Attach curriculum as a callback (no duplicated training loop)
    if args.curriculum:
        curriculum = CurriculumScheduler(
            stage_demands=[0.3, 0.6, 1.0],
            stage_thresholds=[-5.5, -3.5],
            update_budgets=[150, 150, args.n_updates],
        )
        trainer.add_callback(CurriculumCallback(curriculum))
        print("[PPO] Curriculum learning ENABLED via callback")

    history = trainer.train()

    # Save history
    os.makedirs("results/ppo", exist_ok=True)
    np.save("results/ppo/history.npy", history)

    # Save experiment manifest
    final_reward_raw = float(np.mean(history["avg_reward_raw"][-10:])) if history.get("avg_reward_raw") else 0.0
    from utils.manifest import save_manifest
    save_manifest(
        config=vars(args),
        results={
            "final_avg_reward_raw": final_reward_raw,
            "best_raw_reward": trainer._best_reward if hasattr(trainer, "_best_reward") else 0.0,
            "n_updates": args.n_updates,
        },
        manifest_dir="results/ppo/manifests",
        seed=args.seed,
        variant="curriculum" if args.curriculum else "fixed",
    )

    # Plots
    trainer.logger.plot_training_curves(
        save_path="results/ppo/training_curves.png",
        metrics=["p_loss", "v_loss", "entropy", "kl_div",
                 "grad_norm", "explained_var", "Avg_Reward", "Raw_Reward"],
        smooth=20,
    )

    # Final checkpoint + TorchScript
    trainer.save("checkpoints/ppo/ppo_final.pt")
    trainer.export_torchscript("checkpoints/ppo/policy_scripted.pt")

    print("\n[PPO] Done. Logs at results/ppo/")
    vec_env.close()
    return history


if __name__ == "__main__":
    main()
