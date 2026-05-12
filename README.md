# PPO-GAE Agent for Adaptive Traffic Signal Control

**Reinforcement Learning II — TP Final**
**Algorithm:** Proximal Policy Optimization with Generalized Advantage Estimation. This project is based on [all-rl-algorithms](https://github.com/FareedKhan-dev/all-rl-algorithms/blob/master/08_a2c.ipynb)
**Environment:** 4x4 multi-intersection grid (pure Python, no external simulator required)

---

## Overview

This project implements and compares two actor-critic algorithms — **baseline A2C** and **PPO-GAE** — on an adaptive traffic signal control problem. A centralized agent controls 16 intersections simultaneously, learning to minimize vehicle queue lengths and waiting times across a 4x4 grid.

The work is structured in five phases following the proposal in `propose_tp_final.md`:

| Phase | Description | Entry Point |
|-------|-------------|-------------|
| 1 | Custom traffic environment (144-dim state, 16x Discrete(4) actions) | `envs/traffic_grid_env.py` |
| 2 | Baseline A2C with n-step returns, single-epoch updates | `train_a2c.py` |
| 3 | PPO-GAE with 8 algorithmic improvements over A2C | `train_ppo.py` |
| 4 | Multi-seed evaluation, baseline comparison, hyperparameter sweep | `evaluate.py`, `sweep.py` |
| 5 | Curriculum learning, TensorBoard, checkpointing, TorchScript export | `train_ppo.py --curriculum` |

---

## Architecture

### Neural Network (SharedActorCritic)

```
Input: state in R^144 (16 intersections x 9 features)
       |
       v
Linear(144, 256) + ReLU
       |
       v
Linear(256, 256) + ReLU   <-- Shared trunk
       |
   +---+---+
   |       |
   v       v
[Actor]  [Critic]
16 x Linear(256, 4) + Softmax    Linear(256, 128) + ReLU
= 16 independent Categorical     Linear(128, 1)
  distributions (one per         = scalar V(s)
  intersection)
```

### Environment

| Property | Value |
|----------|-------|
| Grid | 4x4 = 16 intersections |
| Observation dim | 144 (16 intersections x 9 features) |
| Features per intersection | 4 lane queue lengths + 4-dim phase one-hot + 1 time scalar |
| Action space | `MultiDiscrete([4] * 16)` |
| Episode length | 720 steps (~1 simulated hour at 5s/step) |
| Reward | `r = -(sum_queues / N_lanes) - 0.1 * (sum_waiting / N_lanes)` |
| Vehicle arrivals | Independent Poisson process per lane |
| Min green constraint | 5 steps before phase switch is allowed |

### PPO-GAE Loss Function

```
L = -L_CLIP(theta) + c_v * L_VF - c_e * H(pi_theta)

L_CLIP = E[ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]
A_t    = GAE:  sum_{l>=0} (gamma * lambda)^l * delta_{t+l}
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
c_e    = linear decay from 0.01 to 0.001 over training
```

---

## Project Structure

```
tp_final/
|
|-- envs/
|   |-- traffic_grid_env.py   # MultiIntersectionEnv: 144-dim obs, Poisson arrivals, min-green constraint
|   |-- sumo_adapter.py       # Optional drop-in SUMO wrapper (requires sumo-rl installed)
|
|-- models/
|   |-- actor_critic.py       # SharedActorCritic: shared trunk + 16 actor heads + scalar critic
|
|-- algorithms/
|   |-- rollout_buffer.py     # RolloutBuffer: stores transitions, computes GAE / n-step returns
|   |-- a2c.py                # A2CTrainer: baseline A2C (n-step returns, 1 epoch, no clipping)
|   |-- ppo.py                # PPOTrainer: full PPO-GAE with all 8 improvements
|
|-- utils/
|   |-- running_stats.py      # RunningMeanStd: online reward normalization (Welford algorithm)
|   |-- curriculum.py         # CurriculumScheduler: 3-stage demand scaling (30% -> 60% -> 100%)
|   |-- logger.py             # MetricLogger: TensorBoard writer + matplotlib plot helpers
|
|-- train_a2c.py              # Phase 2: train baseline A2C
|-- train_ppo.py              # Phase 3+5: train PPO-GAE (optional --curriculum flag)
|-- evaluate.py               # Phase 4: evaluate all policies, produce comparison table + plots
|-- sweep.py                  # Phase 4: grid search over epsilon, lambda, lr
|-- tp_final_notebook.ipynb   # Summary notebook: all phases in one place
|-- requirements.txt
|-- propose_tp_final.md       # Original project proposal
|
|-- checkpoints/
|   |-- a2c/                  # A2C checkpoints (a2c_final.pt)
|   |-- ppo/                  # PPO checkpoints (ppo_best.pt, ppo_update_N.pt, policy_scripted.pt)
|
|-- results/
    |-- a2c/                  # TensorBoard logs + training_curves.png + history.npy
    |-- ppo/                  # TensorBoard logs + training_curves.png + history.npy
    |-- eval/                 # comparison_table.txt + reward_comparison.png + queue_comparison.png
    |-- sweep/                # sweep_results.csv + sweep_heatmap.png
```

---

## Requirements

**Python:** 3.9 or higher (tested on 3.12)

**Core dependencies:**

```
gymnasium>=0.29.0
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
tensorboard>=2.14.0
```

**Optional (Jupyter notebook):**

```
jupyter>=1.0.0
ipykernel>=6.0.0
```

**Optional (SUMO real simulator):**

```
sumo-rl>=1.4.0
```
SUMO also requires the SUMO binary installed separately (see [SUMO section](#optional-sumo-simulator)).

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train PPO-GAE (recommended, full settings)

```bash
python train_ppo.py --n_updates 500 --n_steps 128 --n_envs 16 --seed 0
```

### 3. Evaluate and compare against baselines

```bash
python evaluate.py --n_seeds 5 --n_eval_episodes 20
```

Results are printed to stdout and saved to `results/eval/`.

---

## Usage

All scripts are run from the project root directory (`tp_final/`).

### train_a2c.py — Phase 2: Baseline A2C

Trains the A2C baseline with n-step returns and a single gradient update per rollout.

```bash
python train_a2c.py [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_updates` | `200` | Number of training iterations |
| `--n_steps` | `128` | Rollout steps per environment per update |
| `--n_envs` | `16` | Number of parallel environment instances |
| `--gamma` | `0.99` | Discount factor |
| `--lr` | `3e-4` | Adam learning rate |
| `--c_v` | `0.5` | Value loss coefficient |
| `--c_e` | `0.01` | Entropy coefficient (static) |
| `--grad_clip` | `0.5` | Gradient clipping max norm |
| `--demand` | `1.0` | Traffic demand factor [0.0, 1.0] |
| `--seed` | `0` | Random seed |
| `--device` | `cpu` | Torch device (`cpu` or `cuda`) |
| `--normalize_rewards` | off | Enable reward normalization |

**Outputs:**
- `results/a2c/training_curves.png` — P_Loss, V_Loss, Entropy, Avg Reward plots
- `results/a2c/history.npy` — Training metrics array
- `results/a2c/events.out.tfevents.*` — TensorBoard logs
- `checkpoints/a2c/a2c_final.pt` — Final model checkpoint

**Example (fast demo run):**
```bash
python train_a2c.py --n_updates 50 --n_steps 64 --n_envs 4
```

---

### train_ppo.py — Phase 3+5: PPO-GAE

Trains the full PPO agent with GAE, reward normalization, entropy decay, and optional curriculum learning.

```bash
python train_ppo.py [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_updates` | `500` | Number of PPO update cycles |
| `--n_steps` | `128` | Rollout steps per environment per update |
| `--n_envs` | `16` | Number of parallel environment instances |
| `--gamma` | `0.99` | Discount factor |
| `--gae_lambda` | `0.95` | GAE lambda (bias-variance tradeoff) |
| `--clip_epsilon` | `0.2` | PPO clipping threshold |
| `--lr` | `3e-4` | Adam initial learning rate |
| `--lr_decay` | on | Linear LR decay to 0 over training |
| `--c_v` | `0.5` | Value loss coefficient |
| `--c_e_start` | `0.01` | Initial entropy coefficient |
| `--c_e_end` | `0.001` | Final entropy coefficient (linear decay) |
| `--n_epochs` | `4` | PPO update epochs per rollout (K) |
| `--batch_size` | `256` | Mini-batch size within PPO update |
| `--grad_clip` | `0.5` | Gradient clipping max norm |
| `--kl_target` | `0.02` | Approx KL threshold for early epoch stopping |
| `--save_freq` | `50` | Checkpoint every N updates |
| `--curriculum` | off | Enable 3-stage curriculum learning |
| `--no_norm_rewards` | off | Disable reward normalization |
| `--seed` | `0` | Random seed |
| `--device` | `cpu` | Torch device (`cpu` or `cuda`) |

**Outputs:**
- `results/ppo/training_curves.png` — 6-panel metrics plot
- `results/ppo/history.npy` — Training metrics array
- `results/ppo/events.out.tfevents.*` — TensorBoard logs
- `checkpoints/ppo/ppo_best.pt` — Best model by avg reward
- `checkpoints/ppo/ppo_update_N.pt` — Periodic checkpoints (every `--save_freq` updates)
- `checkpoints/ppo/policy_scripted.pt` — TorchScript export of the value function

**Example (full training with curriculum):**
```bash
python train_ppo.py --n_updates 500 --n_steps 128 --n_envs 16 --curriculum --seed 0
```

**Example (fast demo run):**
```bash
python train_ppo.py --n_updates 50 --n_steps 64 --n_envs 4 --batch_size 128
```

---

### evaluate.py — Phase 4: Evaluation and Benchmarking

Evaluates all trained policies against five baselines and generates comparison plots.

```bash
python evaluate.py [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_seeds` | `5` | Number of independent seeds for PPO multi-seed eval |
| `--n_eval_episodes` | `20` | Episodes per policy per evaluation |
| `--a2c_ckpt` | `checkpoints/a2c/a2c_final.pt` | Path to A2C checkpoint |
| `--ppo_ckpt` | `checkpoints/ppo/ppo_best.pt` | Path to PPO checkpoint |
| `--demand` | `1.0` | Traffic demand factor for evaluation |
| `--device` | `cpu` | Torch device |

**Evaluated policies:**

| Policy | Description |
|--------|-------------|
| Random | Uniform random phase selection |
| Fixed-Time | Cycle through all phases every 6 steps (30s green equivalent) |
| Actuated | Extend green if unserved queue exceeds threshold |
| A2C (Phase 2) | Trained A2C baseline |
| PPO-GAE (Phase 3) | Trained PPO agent |

**Outputs:**
- `results/eval/comparison_table.txt` — Mean reward, std, queue, waiting time per policy
- `results/eval/reward_comparison.png` — Bar chart with error bars
- `results/eval/queue_comparison.png` — Mean queue length comparison
- `results/eval/ppo_seed_distribution.png` — Per-seed reward scatter plot

---

### sweep.py — Phase 4: Hyperparameter Sweep

Runs a grid search over PPO clip epsilon, GAE lambda, and learning rate.

```bash
python sweep.py [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_updates_sweep` | `100` | Training updates per configuration |
| `--n_steps` | `64` | Rollout steps (reduced for sweep speed) |
| `--n_envs` | `4` | Parallel environments |
| `--seed` | `42` | Random seed |
| `--device` | `cpu` | Torch device |

**Search grid:**

| Hyperparameter | Values searched |
|----------------|-----------------|
| `clip_epsilon` (ε) | {0.1, 0.2, 0.3} |
| `gae_lambda` (λ) | {0.90, 0.95, 0.99} |
| `lr` | {3e-4, 1e-3} |

Total configurations: 3 x 3 x 2 = **18 runs**

**Outputs:**
- `results/sweep/sweep_results.csv` — All configurations ranked by final reward
- `results/sweep/sweep_heatmap.png` — ε vs λ heatmap (best lr per cell)

---

### tp_final_notebook.ipynb — Summary Notebook

Runs all five phases interactively with inline plots and explanations.

```bash
jupyter notebook tp_final_notebook.ipynb
```

The notebook uses reduced settings (50 A2C updates, 100 PPO updates, 4 envs) for fast execution. For full-scale training use the scripts above.

---

## Hyperparameter Reference

| Hyperparameter | Symbol | Default | Recommended Range | Notes |
|----------------|--------|---------|-------------------|-------|
| Discount factor | γ | 0.99 | 0.97–0.99 | Higher for long signal cycles |
| GAE lambda | λ | 0.95 | 0.90–0.99 | Bias-variance tradeoff |
| PPO clip threshold | ε | 0.2 | 0.1–0.3 | Reduce if policy oscillates |
| Learning rate | lr | 3e-4 | 3e-4–1e-3 | Decayed linearly to 0 |
| Value loss coeff | c_v | 0.5 | 0.5–1.0 | Increase to 1.0 early if critic diverges |
| Entropy coeff (start) | c_e | 0.01 | 0.01–0.05 | Higher = more early exploration |
| Entropy coeff (end) | c_e | 0.001 | 0.0001–0.005 | Annealed linearly |
| Rollout length | N | 128 | 64–512 | Per environment per update |
| PPO epochs per update | K | 4 | 4–10 | Monitor KL; stop early if KL > 0.02 |
| Mini-batch size | B | 256 | 128–512 | Total batch = N * n_envs |
| Gradient clip norm | — | 0.5 | 0.5–1.0 | Prevents actor/critic divergence |
| KL early stop threshold | — | 0.02 | 0.01–0.05 | Per-epoch stopping criterion |

---

## Improvements Over Baseline A2C

Each improvement directly addresses a limitation identified in Section 2 of the proposal.

| # | Improvement | Implementation | Limitation Addressed |
|---|-------------|----------------|----------------------|
| 1 | **PPO clipped surrogate** (ε=0.2) | `algorithms/ppo.py:_ppo_update()` | Unbounded policy updates / collapse |
| 2 | **GAE** (λ=0.95) | `algorithms/rollout_buffer.py:compute_gae()` | High advantage variance (n-step returns) |
| 3 | **Reward normalization** (RunningMeanStd) | `utils/running_stats.py` | High early V_Loss (critic overwhelmed) |
| 4 | **Gradient clipping** (max_norm=0.5) | `algorithms/ppo.py:_ppo_update()` | Actor/critic gradient explosion |
| 5 | **Entropy coefficient decay** (0.01 → 0.001) | `algorithms/ppo.py:_get_entropy_coeff()` | Static entropy over-suppresses late policy |
| 6 | **K=4 PPO epochs** per rollout | `algorithms/ppo.py:_ppo_update()` | Sample inefficiency (1 update per rollout) |
| 7 | **Vectorized environments** (16 parallel) | `envs/traffic_grid_env.py:make_vec_env()` | Single-worker throughput bottleneck |
| 8 | **Curriculum learning** (30%→60%→100%) | `utils/curriculum.py` | Cold-start on high-complexity environment |

---

## Monitoring with TensorBoard

After running any training script, launch TensorBoard from the project root:

```bash
tensorboard --logdir results
```

Then open `http://localhost:6006` in a browser.

**Available metrics:**

| Tag | Description | Target |
|-----|-------------|--------|
| `PPO/p_loss` | Clipped policy loss L^CLIP | Stable, near zero |
| `PPO/v_loss` | Value loss L^VF | Decreasing monotonically after early transient |
| `PPO/entropy` | Policy entropy H(π) | Gradual decrease; flag if < 0.1 |
| `PPO/kl_div` | Approximate KL divergence | Should stay below 0.02 |
| `PPO/grad_norm` | Gradient norm (before clipping) | Flag if consistently > 2.0 |
| `PPO/explained_var` | Explained variance of value function | Should approach 1.0 |
| `PPO/Avg_Reward` | Rolling episode reward | Should increase over training |
| `PPO/Entropy_Coeff` | Current c_e value | Decreasing linearly |
| `PPO/LR` | Current learning rate | Decreasing linearly |
| `A2C/P_Loss` | A2C policy gradient loss | — |
| `A2C/V_Loss` | A2C value loss | — |
| `A2C/Entropy` | A2C policy entropy | — |

---

## Generated Outputs Summary

```
results/
  a2c/
    training_curves.png     -- P_Loss, V_Loss, Entropy, Avg Reward, Avg Len
    history.npy             -- Loadable dict of metric lists
    events.out.tfevents.*   -- TensorBoard log file
  ppo/
    training_curves.png     -- 6-panel: P_Loss, V_Loss, Entropy, KL, ExplVar, Avg Reward
    history.npy
    events.out.tfevents.*
  eval/
    comparison_table.txt    -- Policy vs. mean reward / std / queue / waiting
    reward_comparison.png   -- Bar chart comparison
    queue_comparison.png    -- Queue length bar chart
    ppo_seed_distribution.png -- Per-seed scatter plot
  sweep/
    sweep_results.csv       -- Ranked configurations
    sweep_heatmap.png       -- epsilon x lambda heatmap

checkpoints/
  a2c/
    a2c_final.pt            -- Final A2C weights + optimizer state
  ppo/
    ppo_best.pt             -- Best PPO model by rolling avg reward
    ppo_update_N.pt         -- Periodic checkpoints (every --save_freq updates)
    policy_scripted.pt      -- TorchScript export (inference-only deployment)
```

**Loading a checkpoint:**

```python
import torch
from models.actor_critic import SharedActorCritic

model = SharedActorCritic()
ckpt = torch.load("checkpoints/ppo/ppo_best.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

**Loading the TorchScript export (no model code needed):**

```python
import torch
policy = torch.jit.load("checkpoints/ppo/policy_scripted.pt")
value = policy.get_value(torch.zeros(1, 144))
```

---

## Reproducibility

All scripts accept a `--seed` argument that seeds both NumPy and PyTorch.

```bash
# Reproduce the exact training run
python train_ppo.py --seed 42 --n_updates 500 --n_steps 128 --n_envs 16
```

**Pinned versions used during development:**

```
Python    3.12
gymnasium 0.29.x
torch     2.x
numpy     2.x
matplotlib 3.x
tensorboard 2.x
```

---

## Optional: SUMO Simulator

The project uses a self-contained pure-Python environment by default. To use the real SUMO traffic simulator:

**Step 1 — Install SUMO:**

- Windows: download the `.msi` installer from https://eclipse.dev/sumo/
- Set the `SUMO_HOME` environment variable to the SUMO installation directory

**Step 2 — Install the Python wrapper:**

```bash
pip install sumo-rl
```

**Step 3 — Swap the environment:**

```python
from envs.sumo_adapter import SumoGridEnv

env = SumoGridEnv(
    net_file="sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml",
    route_file="sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml",
    num_seconds=3600,
)
```

The adapter exposes the same `observation_space` and `action_space` as `MultiIntersectionEnv`, so all training scripts work without modification.

---

## References

- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
- Schulman et al. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016
- Mnih et al. (2016). *Asynchronous Methods for Deep Reinforcement Learning.* ICML 2016
- FareedKhan-dev/all-rl-algorithms — original `8_a2c.ipynb` analyzed in Section 1 of the proposal
