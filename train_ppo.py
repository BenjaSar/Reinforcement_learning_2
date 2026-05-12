"""
train_ppo.py — Phase 3 entry point
=====================================
Trains the PPO-GAE agent with all improvements over A2C.

Usage:
    python train_ppo.py [--n_updates 500] [--n_steps 128] [--n_envs 16]
                        [--gamma 0.99] [--gae_lambda 0.95]
                        [--clip_epsilon 0.2] [--lr 3e-4]
                        [--c_v 0.5] [--c_e_start 0.01] [--c_e_end 0.001]
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
from utils.curriculum import CurriculumScheduler


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO-GAE agent")
    p.add_argument("--n_updates",       type=int,   default=500)
    p.add_argument("--n_steps",         type=int,   default=128)
    p.add_argument("--n_envs",          type=int,   default=16)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--gae_lambda",      type=float, default=0.95)
    p.add_argument("--clip_epsilon",    type=float, default=0.2)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--lr_decay",        action="store_true", default=True)
    p.add_argument("--c_v",             type=float, default=0.5)
    p.add_argument("--c_e_start",       type=float, default=0.01)
    p.add_argument("--c_e_end",         type=float, default=0.001)
    p.add_argument("--n_epochs",        type=int,   default=4)
    p.add_argument("--batch_size",      type=int,   default=256)
    p.add_argument("--grad_clip",       type=float, default=0.5)
    p.add_argument("--kl_target",       type=float, default=0.02)
    p.add_argument("--device",          type=str,   default="cpu")
    p.add_argument("--seed",            type=int,   default=0)
    p.add_argument("--curriculum",      action="store_true", default=False,
                   help="Enable curriculum learning (Phase 5)")
    p.add_argument("--save_freq",       type=int,   default=50)
    p.add_argument("--no_norm_rewards", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  Phase 3 — PPO-GAE Training")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k:20s}: {v}")
    print("=" * 60)

    # --- Curriculum or fixed demand ---
    if args.curriculum:
        curriculum = CurriculumScheduler(
            stage_demands    = [0.3, 0.6, 1.0],
            stage_thresholds = [-5.5, -3.5],
            update_budgets   = [150, 150, args.n_updates],
        )
        initial_demand = 0.3
        print("[PPO] Curriculum learning ENABLED")
    else:
        curriculum = None
        initial_demand = 1.0

    # --- Environment ---
    vec_env = make_vec_env(
        n_envs        = args.n_envs,
        demand_factor = initial_demand,
        base_seed     = args.seed,
    )

    # --- Model ---
    model = SharedActorCritic()

    # --- Trainer ---
    trainer = PPOTrainer(
        env               = vec_env,
        model             = model,
        n_steps           = args.n_steps,
        n_updates         = args.n_updates,
        gamma             = args.gamma,
        gae_lambda        = args.gae_lambda,
        clip_epsilon      = args.clip_epsilon,
        lr                = args.lr,
        lr_decay          = args.lr_decay,
        c_v               = args.c_v,
        c_e_start         = args.c_e_start,
        c_e_end           = args.c_e_end,
        n_epochs          = args.n_epochs,
        batch_size        = args.batch_size,
        max_grad_norm     = args.grad_clip,
        kl_target         = args.kl_target,
        normalize_rewards = not args.no_norm_rewards,
        device            = args.device,
        log_dir           = "results/ppo",
        save_dir          = "checkpoints/ppo",
        save_freq         = args.save_freq,
    )

    # Curriculum integration: patch the trainer's train loop
    if curriculum is not None:
        history = _train_with_curriculum(trainer, vec_env, curriculum, args)
    else:
        history = trainer.train()

    # --- Save history ---
    os.makedirs("results/ppo", exist_ok=True)
    np.save("results/ppo/history.npy", history)

    # --- Plots ---
    trainer.logger.plot_training_curves(
        save_path = "results/ppo/training_curves.png",
        metrics   = ["p_loss", "v_loss", "entropy", "kl_div",
                     "grad_norm", "explained_var", "avg_reward"],
        smooth    = 20,
    )

    # --- Final checkpoint ---
    trainer.save("checkpoints/ppo/ppo_final.pt")

    # --- TorchScript export ---
    trainer.export_torchscript("checkpoints/ppo/policy_scripted.pt")

    print("\n[PPO] Done. Logs at results/ppo/")
    vec_env.close()
    return history


def _train_with_curriculum(trainer, vec_env, curriculum, args):
    """
    Runs PPO training loop with curriculum demand scheduling.
    Calls vec_env.call('set_demand_factor', demand) each update cycle.
    """
    import time
    import torch
    from algorithms.rollout_buffer import RolloutBuffer

    print(f"[Curriculum] Stage 1: demand={curriculum.current_demand:.0%}")

    # We monkey-patch trainer's train() to inject curriculum steps
    # by running the training manually here using the trainer's internal API.
    history = {
        "p_loss": [], "v_loss": [], "entropy": [], "kl_div": [],
        "grad_norm": [], "explained_var": [], "avg_reward": [], "avg_len": [],
        "c_e": [], "demand_factor": [],
    }

    obs_np, _ = vec_env.reset()
    device = trainer.device
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    import numpy as np
    done_np = np.zeros(trainer.n_envs, dtype=np.float32)
    t0 = time.time()

    for update in range(1, args.n_updates + 1):
        # Update demand in all envs
        demand = curriculum.get_demand_factor()
        vec_env.call("set_demand_factor", demand)

        c_e = trainer._get_entropy_coeff(update)
        trainer.buffer.reset()

        for _ in range(trainer.n_steps):
            with torch.no_grad():
                actions, log_probs, values = trainer.model.act(obs)
            actions_np = actions.cpu().numpy()
            obs_next_np, rewards_np, term_np, trunc_np, _ = vec_env.step(actions_np)
            done_np = np.logical_or(term_np, trunc_np).astype(np.float32)

            if trainer.reward_normalizer is not None:
                trainer.reward_normalizer.update(rewards_np)
                rewards_np = rewards_np / (trainer.reward_normalizer.std + 1e-8)

            trainer._ep_rewards += rewards_np
            trainer._ep_lengths += 1
            for i, d in enumerate(done_np):
                if d:
                    trainer._completed_rewards.append(float(trainer._ep_rewards[i]))
                    trainer._completed_lengths.append(int(trainer._ep_lengths[i]))
                    trainer._ep_rewards[i] = 0.0
                    trainer._ep_lengths[i] = 0

            trainer.buffer.add(
                obs=obs, actions=actions,
                rewards=torch.tensor(rewards_np, dtype=torch.float32),
                values=values,
                log_probs=log_probs,
                dones=torch.tensor(done_np, dtype=torch.float32),
            )
            obs = torch.tensor(obs_next_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            last_value = trainer.model.get_value(obs)
        trainer.buffer.compute_gae(last_value,
                                   torch.tensor(done_np, dtype=torch.float32))
        trainer.buffer.normalize_advantages()
        metrics = trainer._ppo_update(c_e)

        if trainer.scheduler:
            trainer.scheduler.step()

        window = trainer.n_envs * 4
        avg_r = (float(np.mean(trainer._completed_rewards[-window:]))
                 if trainer._completed_rewards else 0.0)
        avg_l = (float(np.mean(trainer._completed_lengths[-window:]))
                 if trainer._completed_lengths else 0.0)

        for k, v in metrics.items():
            history[k].append(v)
            trainer.logger.log_scalar(k, v, update)
        history["avg_reward"].append(avg_r)
        history["avg_len"].append(avg_l)
        history["c_e"].append(c_e)
        history["demand_factor"].append(demand)
        trainer.logger.log_scalar("Avg_Reward",    avg_r,  update)
        trainer.logger.log_scalar("Avg_Len",       avg_l,  update)
        trainer.logger.log_scalar("Demand_Factor", demand, update)

        if update % 10 == 0:
            print(f"[PPO+Curr] iter {update:4d} | demand={demand:.0%} | "
                  f"R={avg_r:.3f} | {curriculum}")

        # Curriculum advance
        curriculum.step(avg_r)

        # Checkpointing
        if update % args.save_freq == 0:
            trainer.save(f"checkpoints/ppo/ppo_update_{update}.pt")
        if avg_r > trainer._best_reward and trainer._completed_rewards:
            trainer._best_reward = avg_r
            trainer.save("checkpoints/ppo/ppo_best.pt")

    trainer.logger.flush()
    print(f"[PPO+Curr] Done in {time.time()-t0:.1f}s")
    return history


if __name__ == "__main__":
    main()
