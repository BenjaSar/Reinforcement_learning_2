"""
sweep.py — Phase 4: Hyperparameter Sweep
==========================================
Systematic grid search over:
    epsilon in {0.1, 0.2, 0.3}
    lambda  in {0.9, 0.95, 0.99}
    lr      in {3e-4, 1e-3}

Each configuration is trained for n_updates_sweep iterations on a
single environment and the final avg_reward is recorded.

Usage:
    python sweep.py [--n_updates_sweep 100] [--n_envs 4] [--seed 0]

Outputs:
    - results/sweep/sweep_results.csv
    - results/sweep/sweep_heatmap.png
"""

import argparse
import sys
import os
import csv
import itertools
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.traffic_grid_env import make_vec_env
from models.actor_critic import SharedActorCritic
from algorithms.ppo import PPOTrainer


SWEEP_GRID = {
    "clip_epsilon": [0.1, 0.2, 0.3],
    "gae_lambda":   [0.9, 0.95, 0.99],
    "lr":           [3e-4, 1e-3],
}


def parse_args():
    p = argparse.ArgumentParser(description="Hyperparameter sweep")
    p.add_argument("--n_updates_sweep", type=int,   default=100,
                   help="Updates per configuration")
    p.add_argument("--n_steps",         type=int,   default=64,
                   help="Rollout steps (reduced for sweep speed)")
    p.add_argument("--n_envs",          type=int,   default=4)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--device",          type=str,   default="cpu")
    return p.parse_args()


def run_config(config: dict, args) -> float:
    """Train for n_updates_sweep and return final avg_reward."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vec_env = make_vec_env(
        n_envs    = args.n_envs,
        base_seed = args.seed,
    )
    model = SharedActorCritic()

    log_tag = (f"eps{config['clip_epsilon']}_"
               f"lam{config['gae_lambda']}_"
               f"lr{config['lr']:.0e}")

    trainer = PPOTrainer(
        env            = vec_env,
        model          = model,
        n_steps        = args.n_steps,
        n_updates      = args.n_updates_sweep,
        clip_epsilon   = config["clip_epsilon"],
        gae_lambda     = config["gae_lambda"],
        lr             = config["lr"],
        lr_decay       = False,
        normalize_rewards = True,
        device         = args.device,
        log_dir        = f"results/sweep/{log_tag}",
        save_dir       = f"checkpoints/sweep/{log_tag}",
        save_freq      = args.n_updates_sweep + 1,  # no intermediate saves
    )

    history = trainer.train()
    vec_env.close()

    # Return mean of last 20% of updates
    rewards = history["avg_reward"]
    final = float(np.mean(rewards[-max(1, len(rewards)//5):])) if rewards else -999.0
    return final


def main():
    args = parse_args()
    os.makedirs("results/sweep", exist_ok=True)

    keys   = list(SWEEP_GRID.keys())
    values = list(SWEEP_GRID.values())
    combos = list(itertools.product(*values))

    print("=" * 60)
    print(f"  Phase 4 — Hyperparameter Sweep ({len(combos)} configs)")
    print("=" * 60)

    results = []
    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        print(f"\n[Sweep {i+1}/{len(combos)}] {config}")
        final_reward = run_config(config, args)
        config["final_reward"] = final_reward
        results.append(config)
        print(f"  -> Final avg reward: {final_reward:.4f}")

    # Sort by reward
    results.sort(key=lambda x: x["final_reward"], reverse=True)

    # Save CSV
    csv_path = "results/sweep/sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys + ["final_reward"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[Sweep] Results saved -> {csv_path}")

    # Print top 5
    print("\n  Top 5 configurations:")
    print(f"  {'epsilon':>8} {'lambda':>8} {'lr':>8} {'reward':>12}")
    print("  " + "-" * 42)
    for r in results[:5]:
        print(f"  {r['clip_epsilon']:>8.2f} "
              f"{r['gae_lambda']:>8.3f} "
              f"{r['lr']:>8.1e} "
              f"{r['final_reward']:>12.4f}")

    # Heatmap (epsilon vs lambda, best lr per cell)
    _plot_heatmap(results)


def _plot_heatmap(results):
    """Plot reward heatmap: epsilon (rows) x lambda (cols), best lr."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epsilons = sorted(set(r["clip_epsilon"] for r in results))
    lambdas  = sorted(set(r["gae_lambda"]   for r in results))

    matrix = np.full((len(epsilons), len(lambdas)), np.nan)
    for r in results:
        i = epsilons.index(r["clip_epsilon"])
        j = lambdas.index(r["gae_lambda"])
        if np.isnan(matrix[i, j]) or r["final_reward"] > matrix[i, j]:
            matrix[i, j] = r["final_reward"]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                   vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([str(l) for l in lambdas])
    ax.set_yticks(range(len(epsilons)))
    ax.set_yticklabels([str(e) for e in epsilons])
    ax.set_xlabel("GAE Lambda (λ)", fontsize=12)
    ax.set_ylabel("PPO Clip Epsilon (ε)", fontsize=12)
    ax.set_title("Hyperparameter Sweep — Final Avg Reward\n(best lr per cell)",
                 fontsize=12)
    for i in range(len(epsilons)):
        for j in range(len(lambdas)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=10, color="black")
    plt.colorbar(im, ax=ax, label="Final Avg Reward")
    plt.tight_layout()
    path = "results/sweep/sweep_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Sweep] Heatmap saved -> {path}")


if __name__ == "__main__":
    main()
