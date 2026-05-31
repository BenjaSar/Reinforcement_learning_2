"""
multi_seed.py — Multi-seed PPO training runner.
Runs 3 seeds × 2 variants (fixed-demand, curriculum) and evaluates all.
"""
import sys, os, subprocess, argparse, csv, datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEEDS = [0, 1, 2]
VARIANTS = [
    {"name": "fixed",      "extra": []},
    {"name": "curriculum", "extra": ["--curriculum"]},
]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PPO_SCRIPT = os.path.join(BASE_DIR, "train_ppo.py")
EVAL_SCRIPT = os.path.join(BASE_DIR, "evaluate.py")


def run_training(seed, variant, n_updates=500):
    log_dir = f"results/multi_seed/{variant['name']}_{seed}"
    ckpt_dir = f"checkpoints/multi_seed/{variant['name']}_{seed}"
    cmd = [
        sys.executable, PPO_SCRIPT,
        "--n_updates", str(n_updates),
        "--n_steps", "128",
        "--n_envs", "16",
        "--seed", str(seed),
    ] + variant["extra"]
    env = os.environ.copy()
    print(f"\n{'='*60}")
    print(f"  Training: {variant['name']} seed={seed} ({n_updates} updates)")
    print(f"{'='*60}")
    # Skip if checkpoint already exists (saves time on re-runs)
    ckpt_dst = os.path.join(BASE_DIR, ckpt_dir, "ppo_final.pt")
    if os.path.exists(ckpt_dst):
        print(f"  [Skip] checkpoint already exists at {ckpt_dst}")
        return

    t0 = datetime.datetime.now()
    subprocess.run(cmd, env=env, cwd=BASE_DIR, check=True)
    elapsed = (datetime.datetime.now() - t0).total_seconds()
    print(f"  Done in {elapsed:.1f}s")

    # Copy final checkpoint to seed-specific location
    final_ckpt = os.path.join(BASE_DIR, "checkpoints/ppo/ppo_final.pt")
    if os.path.exists(final_ckpt):
        os.makedirs(os.path.join(BASE_DIR, ckpt_dir), exist_ok=True)
        import shutil
        shutil.copy2(final_ckpt, os.path.join(BASE_DIR, ckpt_dir, "ppo_final.pt"))
        shutil.copy2(
            os.path.join(BASE_DIR, "checkpoints/ppo/ppo_best.pt"),
            os.path.join(BASE_DIR, ckpt_dir, "ppo_best.pt"),
        )


def run_evaluation(seed, variant):
    ckpt_path = f"checkpoints/multi_seed/{variant['name']}_{seed}/ppo_final.pt"
    if not os.path.exists(os.path.join(BASE_DIR, ckpt_path)):
        print(f"  [Skip eval] no checkpoint at {ckpt_path}")
        return None

    from envs.traffic_grid_env import MultiIntersectionEnv
    from models.actor_critic import SharedActorCritic
    from evaluate import rl_policy, aggregate

    model = SharedActorCritic()
    ckpt = torch.load(os.path.join(BASE_DIR, ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    env = MultiIntersectionEnv(demand_factor=1.0)
    results = rl_policy(env, model, 20, seed * 1000 + 5000, device="cpu")
    agg = aggregate(results)
    env.close()
    return agg


if __name__ == "__main__":
    import torch
    parser = argparse.ArgumentParser(description="Multi-seed PPO runner")
    parser.add_argument("--n_updates", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--eval_only", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs("results/multi_seed", exist_ok=True)

    results = []
    for variant in VARIANTS:
        for seed in args.seeds:
            if not args.eval_only:
                run_training(seed, variant, args.n_updates)
            agg = run_evaluation(seed, variant)
            row = {
                "variant": variant["name"],
                "seed": seed,
                "mean_reward": agg["mean_reward"] if agg else None,
                "std_reward": agg["std_reward"] if agg else None,
                "mean_queue": agg["mean_queue"] if agg else None,
            }
            results.append(row)
            if agg:
                print(f"  {variant['name']} seed={seed}: {agg['mean_reward']:.2f} +/- {agg['std_reward']:.2f}")

    # CSV
    csv_path = "results/multi_seed/summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "seed", "mean_reward", "std_reward", "mean_queue"])
        w.writeheader()
        w.writerows(results)
    print(f"\n[MultiSeed] Summary -> {csv_path}")

    # Aggregated stats
    print("\n  Aggregated Results")
    print(f"  {'Variant':<15} {'Mean':>10} {'Std':>10} {'Queue':>8}")
    print("  " + "-" * 45)
    for variant in VARIANTS:
        rs = [r for r in results if r["variant"] == variant["name"] and r["mean_reward"] is not None]
        if rs:
            means = [r["mean_reward"] for r in rs]
            print(f"  {variant['name']:<15} {np.mean(means):>10.2f} {np.std(means):>10.2f} "
                  f"{np.mean([r['mean_queue'] for r in rs]):>8.4f}")
