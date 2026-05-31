"""
Cross-validation: evaluate trained PPO policy in both pure-Python and SUMO.

Usage:
    python eval_cross_validate.py --checkpoint checkpoints/ppo/ppo_final.pt
"""
import argparse, os, sys, json, time, numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.environ["SUMO_HOME"] = r"C:\Users\PC\AppData\Local\Temp\opencode\sumo_bin\sumo-1.21.0"

import torch
from envs.traffic_grid_env import MultiIntersectionEnv
from envs.sumo_adapter import SumoGridEnv, _SUMO_AVAILABLE
from models.actor_critic import SharedActorCritic

import sumo_rl

DEMAND_LEVELS = [0.3, 0.6, 1.0]
N_EVAL_EPISODES = 10
MAX_STEPS = 180  # 180 steps × 5s = 900s simulation


def make_sumo_env(demand: float):
    """Create SumoGridEnv at the given demand level."""
    d = os.path.dirname(sumo_rl.__file__)
    net = os.path.join(d, "nets", "RESCO", "grid4x4", "grid4x4.net.xml")

    demand_str = f"{demand:.1f}".replace(".", "")
    route_dir = os.path.join(os.path.dirname(__file__), "sumo", "routes")
    route = os.path.join(route_dir, f"grid4x4_{demand_str}.rou.xml")

    return SumoGridEnv(
        net_file=net, route_file=route,
        num_seconds=MAX_STEPS * 5, delta_time=5,
        use_gui=False,
    )


def make_python_env(demand: float):
    return MultiIntersectionEnv(demand_factor=demand, seed=42)


def evaluate(env, model, device, n_episodes, max_steps):
    """Run evaluation episodes; return mean ± std reward."""
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        for _ in range(max_steps):
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                action = model.act_deterministic(obs_t)
            action = action.cpu().numpy().squeeze(0)
            obs, r, term, trunc, _ = env.step(action)
            ep_reward += float(r)
            if term or trunc:
                break
        rewards.append(ep_reward)
    mean = float(np.mean(rewards))
    std = float(np.std(rewards))
    return mean, std, rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--n_episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SharedActorCritic().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("=" * 60)
    print(f"Cross-Validation: {args.checkpoint}")
    print(f"Device: {device}, Episodes: {args.n_episodes}")
    print("=" * 60)

    results = {}

    for demand in DEMAND_LEVELS:
        print(f"\n--- Demand {demand:.1f} ---")

        # Pure-Python eval
        py_env = make_python_env(demand)
        mean_py, std_py, _ = evaluate(
            py_env, model, device, args.n_episodes, args.max_steps
        )
        py_env.close()
        print(f"  Python:    {mean_py:8.2f} ± {std_py:.2f}")

        # SUMO eval
        if _SUMO_AVAILABLE:
            su_env = make_sumo_env(demand)
            mean_su, std_su, su_rewards = evaluate(
                su_env, model, device, args.n_episodes, args.max_steps
            )
            su_env.close()
            print(f"  SUMO:      {mean_su:8.2f} ± {std_su:.2f}")
        else:
            mean_su = std_su = None
            print(f"  SUMO:      SKIPPED (sumo-rl not available)")

        results[f"demand_{demand:.1f}"] = {
            "python_mean": mean_py, "python_std": std_py,
            "sumo_mean": mean_su, "sumo_std": std_su,
        }

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Demand':>8}  {'Python':>10}  {'SUMO':>10}  {'Delta':>10}")
    print("-" * 60)
    for demand in DEMAND_LEVELS:
        r = results[f"demand_{demand:.1f}"]
        delta = r["sumo_mean"] - r["python_mean"] if r["sumo_mean"] is not None else 0
        pystr = f"{r['python_mean']:7.2f}±{r['python_std']:.2f}"
        sust = f"{r['sumo_mean']:7.2f}±{r['sumo_std']:.2f}" if r['sumo_mean'] is not None else "  N/A"
        print(f"{demand:>8.1f}  {pystr:>10}  {sust:>10}  {delta:>+8.2f}")

    # Save results
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cross_validation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
