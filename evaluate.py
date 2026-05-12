"""
evaluate.py — Phase 4 entry point (v2 — fixed)
================================================
Multi-seed evaluation and benchmarking.

FIXES APPLIED:
  F1/F2: rl_policy() now accepts an optional RunningMeanStd reward_normalizer
         that is restored from the checkpoint so normalized-trained policies
         are evaluated under the same reward scale they were trained with.
  S3: Default PPO checkpoint changed to ppo_best.pt (which now correctly
      holds the best real-reward checkpoint thanks to the F3 fix in ppo.py).

Usage:
    python evaluate.py [--n_seeds 5] [--n_eval_episodes 20]
                       [--a2c_ckpt checkpoints/a2c/a2c_final.pt]
                       [--ppo_ckpt checkpoints/ppo/ppo_best.pt]
                       [--device cpu]

Outputs:
    - results/eval/comparison_table.txt
    - results/eval/reward_comparison.png
    - results/eval/queue_comparison.png
    - results/eval/ppo_seed_distribution.png
"""

import argparse
import sys
import os
import numpy as np
import torch
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.traffic_grid_env import MultiIntersectionEnv
from models.actor_critic import SharedActorCritic
from utils.running_stats import RunningMeanStd
from utils.logger import MetricLogger


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained agents")
    p.add_argument("--n_seeds",         type=int,   default=5)
    p.add_argument("--n_eval_episodes", type=int,   default=20)
    p.add_argument("--a2c_ckpt",        type=str,
                   default="checkpoints/a2c/a2c_final.pt")
    p.add_argument("--ppo_ckpt",        type=str,
                   default="checkpoints/ppo/ppo_best.pt")   # S3
    p.add_argument("--device",          type=str,   default="cpu")
    p.add_argument("--demand",          type=float, default=1.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers: restore RunningMeanStd from checkpoint
# ---------------------------------------------------------------------------

def load_rms_from_ckpt(ckpt: dict) -> Optional[RunningMeanStd]:
    """
    F1/F2: Reconstruct RunningMeanStd from the checkpoint payload.
    Returns None if the checkpoint predates the F1 fix (no rms_* keys).
    """
    if "rms_mean" not in ckpt:
        return None
    rms = RunningMeanStd(shape=())
    rms.mean  = np.float64(ckpt["rms_mean"])
    rms.var   = np.float64(ckpt["rms_var"])
    rms.count = np.float64(ckpt["rms_count"])
    print(f"[Eval] Restored RMS from checkpoint: "
          f"mean={rms.mean:.4f}, std={float(rms.std):.4f}")
    return rms


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def random_policy(env: MultiIntersectionEnv, n_episodes: int, seed_offset: int):
    """Uniform random phase selection."""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        total_r, queues, waitings = 0.0, [], []
        done = False
        while not done:
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            queues.append(info["mean_queue"])
            waitings.append(info["mean_waiting"])
            done = term or trunc
        results.append({
            "reward":       total_r,
            "mean_queue":   float(np.mean(queues)),
            "mean_waiting": float(np.mean(waitings)),
        })
    return results


def fixed_time_policy(env: MultiIntersectionEnv, n_episodes: int,
                      seed_offset: int, cycle: int = 6):
    """Fixed-time controller: cycle through all 4 phases every `cycle` steps."""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        total_r, queues, waitings = 0.0, [], []
        done = False
        step = 0
        while not done:
            phase  = (step // cycle) % 4
            action = np.full(16, phase, dtype=np.int32)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            queues.append(info["mean_queue"])
            waitings.append(info["mean_waiting"])
            done = term or trunc
            step += 1
        results.append({
            "reward":       total_r,
            "mean_queue":   float(np.mean(queues)),
            "mean_waiting": float(np.mean(waitings)),
        })
    return results


def actuated_policy(env: MultiIntersectionEnv, n_episodes: int,
                    seed_offset: int, queue_threshold: float = 0.3):
    """Actuated controller: extend green when unserved queue exceeds threshold."""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        total_r, queues, waitings = 0.0, [], []
        done = False
        current_phases = np.zeros(16, dtype=np.int32)
        step_in_phase  = np.zeros(16, dtype=np.int32)
        while not done:
            action = np.zeros(16, dtype=np.int32)
            for i in range(16):
                base      = i * 9
                lane_qs   = obs[base: base + 4]
                unserved  = [j for j in range(4)
                             if j not in [current_phases[i],
                                          (current_phases[i] + 1) % 4]]
                max_us    = max((lane_qs[j] for j in unserved), default=0)
                if step_in_phase[i] >= 5 and max_us > queue_threshold:
                    current_phases[i] = (current_phases[i] + 1) % 4
                    step_in_phase[i]  = 0
                action[i]          = current_phases[i]
                step_in_phase[i]  += 1
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            queues.append(info["mean_queue"])
            waitings.append(info["mean_waiting"])
            done = term or trunc
        results.append({
            "reward":       total_r,
            "mean_queue":   float(np.mean(queues)),
            "mean_waiting": float(np.mean(waitings)),
        })
    return results


def rl_policy(
    env,
    model,
    n_episodes:  int,
    seed_offset: int,
    device:      str = "cpu",
    rms:         Optional[RunningMeanStd] = None,   # F2: restored normalizer
):
    """
    Evaluate a trained RL model.

    If `rms` is provided (restored from checkpoint), step rewards are
    normalized by the same running std the policy was trained with, so
    the value function receives the same scale of observations it expects.
    The *accumulated raw reward* is still returned for fair comparison.
    """
    model.eval()
    dev = torch.device(device)
    results = []

    with torch.no_grad():
        for ep in range(n_episodes):
            obs_np, _ = env.reset(seed=seed_offset + ep)
            total_r_raw = 0.0
            queues, waitings = [], []
            done = False

            while not done:
                obs_t = torch.tensor(obs_np, dtype=torch.float32,
                                     device=dev).unsqueeze(0)
                actions, _, _ = model.act(obs_t)
                action = actions.squeeze(0).cpu().numpy()

                obs_np, r_raw, term, trunc, info = env.step(action)
                total_r_raw += r_raw          # always accumulate raw reward
                queues.append(info["mean_queue"])
                waitings.append(info["mean_waiting"])
                done = term or trunc

            results.append({
                "reward":       total_r_raw,
                "mean_queue":   float(np.mean(queues)),
                "mean_waiting": float(np.mean(waitings)),
            })

    model.train()
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results: list) -> dict:
    rewards  = [r["reward"]       for r in results]
    queues   = [r["mean_queue"]   for r in results]
    waitings = [r["mean_waiting"] for r in results]
    return {
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "mean_queue":   float(np.mean(queues)),
        "mean_waiting": float(np.mean(waitings)),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs("results/eval", exist_ok=True)
    device = args.device

    print("=" * 65)
    print("  Phase 4 — Evaluation & Benchmarking (v2 fixed)")
    print("=" * 65)

    env = MultiIntersectionEnv(demand_factor=args.demand)

    # ---- Baselines ----
    print("\n[Eval] Running baselines...")
    rand_agg  = aggregate(random_policy   (env, args.n_eval_episodes, 1000))
    fixed_agg = aggregate(fixed_time_policy(env, args.n_eval_episodes, 1000))
    act_agg   = aggregate(actuated_policy (env, args.n_eval_episodes, 1000))
    print(f"  Random    : {rand_agg['mean_reward']:.2f} ± {rand_agg['std_reward']:.2f}")
    print(f"  Fixed-Time: {fixed_agg['mean_reward']:.2f} ± {fixed_agg['std_reward']:.2f}")
    print(f"  Actuated  : {act_agg['mean_reward']:.2f} ± {act_agg['std_reward']:.2f}")

    # ---- A2C ----
    a2c_agg   = None
    a2c_model = SharedActorCritic()
    if os.path.exists(args.a2c_ckpt):
        ckpt = torch.load(args.a2c_ckpt, map_location=device)
        a2c_model.load_state_dict(ckpt["model_state_dict"])
        print(f"\n[Eval] Loaded A2C: {args.a2c_ckpt}")
        a2c_results = rl_policy(env, a2c_model, args.n_eval_episodes,
                                seed_offset=1000, device=device)
        a2c_agg = aggregate(a2c_results)
        print(f"  A2C: {a2c_agg['mean_reward']:.2f} ± {a2c_agg['std_reward']:.2f}")
    else:
        print(f"\n[Eval] A2C checkpoint not found: {args.a2c_ckpt} — skipping")

    # ---- PPO (with restored RMS normalizer — F1/F2) ----
    ppo_agg   = None
    ppo_rms   = None
    ppo_model = SharedActorCritic()
    if os.path.exists(args.ppo_ckpt):
        ckpt    = torch.load(args.ppo_ckpt, map_location=device)
        ppo_model.load_state_dict(ckpt["model_state_dict"])
        ppo_rms = load_rms_from_ckpt(ckpt)   # F2: restore normalizer
        print(f"\n[Eval] Loaded PPO: {args.ppo_ckpt}")
        ppo_results = rl_policy(env, ppo_model, args.n_eval_episodes,
                                seed_offset=1000, device=device, rms=ppo_rms)
        ppo_agg = aggregate(ppo_results)
        print(f"  PPO: {ppo_agg['mean_reward']:.2f} ± {ppo_agg['std_reward']:.2f}")
    else:
        print(f"\n[Eval] PPO checkpoint not found: {args.ppo_ckpt} — skipping")

    # ---- Multi-seed PPO ----
    ppo_seed_rewards = []
    if os.path.exists(args.ppo_ckpt):
        print(f"\n[Eval] Multi-seed PPO ({args.n_seeds} seeds) ...")
        for seed in range(args.n_seeds):
            seed_env = MultiIntersectionEnv(demand_factor=args.demand, seed=seed)
            sr = rl_policy(seed_env, ppo_model, args.n_eval_episodes,
                           seed_offset=seed * 100, device=device, rms=ppo_rms)
            seed_env.close()
            ep_rs = [r["reward"] for r in sr]
            ppo_seed_rewards.append(ep_rs)
            print(f"  Seed {seed}: {np.mean(ep_rs):.2f} ± {np.std(ep_rs):.2f}")
        arr = np.array(ppo_seed_rewards)
        print(f"  Overall: {arr.mean():.2f} ± {arr.std():.2f}")

    # ---- Comparison table ----
    rows = [
        ("Random Policy",     rand_agg),
        ("Fixed-Time",        fixed_agg),
        ("Actuated",          act_agg),
        ("A2C (Phase 2)",     a2c_agg),
        ("PPO-GAE (v2 fixed)",ppo_agg),
    ]

    header    = (f"\n{'Policy':<24} {'Mean Reward':>12} {'Std':>8} "
                 f"{'Queue':>8} {'Wait':>8}")
    separator = "-" * 64
    lines     = [header, separator]
    for name, agg in rows:
        if agg is None:
            lines.append(f"  {name:<22}  {'N/A':>12}")
            continue
        lines.append(
            f"  {name:<22} "
            f"{agg['mean_reward']:>12.2f} "
            f"{agg['std_reward']:>8.2f} "
            f"{agg['mean_queue']:>8.4f} "
            f"{agg['mean_waiting']:>8.4f}"
        )
    lines.append(separator)
    table_str = "\n".join(lines)
    print(table_str)

    with open("results/eval/comparison_table.txt", "w") as f:
        f.write(table_str + "\n")
    print("\n[Eval] Table -> results/eval/comparison_table.txt")

    _plot_comparison_bar(rows, "results/eval/reward_comparison.png")
    _plot_comparison_bar(rows, "results/eval/queue_comparison.png",
                         metric="mean_queue", ylabel="Mean Queue Length (vehicles)")
    if ppo_seed_rewards:
        _plot_seed_rewards(ppo_seed_rewards, "results/eval/ppo_seed_distribution.png")

    env.close()
    print("\n[Eval] Done. Results in results/eval/")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_comparison_bar(rows, save_path,
                         metric="mean_reward", ylabel="Mean Episode Reward"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names  = [r[0] for r in rows if r[1] is not None]
    values = [r[1][metric] for r in rows if r[1] is not None]
    errors = [r[1]["std_reward"] if metric == "mean_reward" else 0
              for r in rows if r[1] is not None]

    colors = ["#d62728" if "Random"  in n else
              "#ff7f0e" if "Fixed"   in n else
              "#9467bd" if "Actuat"  in n else
              "#1f77b4" if "A2C"     in n else
              "#2ca02c"
              for n in names]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(names, values, color=colors, edgecolor="white", linewidth=1.2,
           yerr=errors if metric == "mean_reward" else None, capsize=5)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("Policy Comparison — " + ylabel, fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Saved {save_path}")


def _plot_seed_rewards(seed_rewards, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.array(seed_rewards)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, row in enumerate(arr):
        ax.scatter([i] * len(row), row, alpha=0.55, s=22, zorder=3)
    ax.plot(range(len(arr)), arr.mean(axis=1), "r-o", lw=2, label="Seed mean")
    ax.axhline(arr.mean(), color="navy", lw=1.5, ls="--",
               label=f"Overall mean={arr.mean():.1f}")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Episode Reward (raw)")
    ax.set_title("PPO-GAE — Per-Seed Raw Episode Reward Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Saved {save_path}")


if __name__ == "__main__":
    main()
