"""Generate results/summary.json from all experiment data."""
import sys, os, json, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.manifest import get_git_commit


def load_csv(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


def build_summary():
    commit = get_git_commit()
    timestamp = __import__("time").strftime("%Y-%m-%dT%H:%M:%S")

    # Multi-seed results
    multi_seed = {}
    ms_path = "results/multi_seed/summary.csv"
    if os.path.exists(ms_path):
        rows = load_csv(ms_path)
        for variant in ("fixed", "curriculum"):
            rs = [r for r in rows if r["variant"] == variant and r["mean_reward"] != "None"]
            if rs:
                means = [float(r["mean_reward"]) for r in rs]
                multi_seed[variant] = {
                    "mean": round(float(np.mean(means)), 2),
                    "std": round(float(np.std(means)), 2),
                    "n_seeds": len(rs),
                    "per_seed": {f"seed_{r['seed']}": float(r["mean_reward"]) for r in rs},
                }

    # Sweep results
    sweep = {}
    sweep_path = "results/sweep/sweep_results.csv"
    if os.path.exists(sweep_path):
        rows = load_csv(sweep_path)
        rows.sort(key=lambda r: float(r["final_reward"]), reverse=True)
        sweep["n_configs"] = len(rows)
        sweep["top5"] = [
            {
                "clip_epsilon": float(r["clip_epsilon"]),
                "gae_lambda": float(r["gae_lambda"]),
                "actor_lr": float(r["actor_lr"]),
                "reward": float(r["final_reward"]),
            }
            for r in rows[:5]
        ]

    # Evaluation baselines
    eval_data = {}
    eval_path = "results/eval/comparison_table.txt"
    if os.path.exists(eval_path):
        lines = open(eval_path).read().strip().split("\n")
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0][0].isalpha():
                try:
                    name = " ".join(parts[:2]).rstrip(":")
                    mean_r = float(parts[-4].replace(",", ""))
                    std_r = float(parts[-3])
                    eval_data[name] = {"mean_reward": mean_r, "std_reward": std_r}
                except (ValueError, IndexError):
                    pass

    summary = {
        "generated": timestamp,
        "git_commit": commit,
        "multi_seed_ppo_200updates": multi_seed,
        "sweep": sweep,
        "evaluation_baselines": eval_data,
        "best_known_results": {
            "random_policy": -931.43,
            "a2c_200updates": -921.19,
            "ppo_fixed_200updates_avg": float(np.mean([float(r["mean_reward"]) for r in load_csv(ms_path) if r["variant"] == "fixed" and r["mean_reward"] != "None"])) if os.path.exists(ms_path) else None,
            "ppo_curriculum_200updates_avg": float(np.mean([float(r["mean_reward"]) for r in load_csv(ms_path) if r["variant"] == "curriculum" and r["mean_reward"] != "None"])) if os.path.exists(ms_path) else None,
            "ppo_fixed_500updates_seed0": -534.6,
            "ppo_curriculum_500updates_seed123": -587.0,
        },
    }

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Summary] Saved -> results/summary.json")
    return summary


if __name__ == "__main__":
    build_summary()
