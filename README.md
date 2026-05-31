![header](doc/imgs/LogoHeader.png)

# PPO-GAE Agent for Adaptive Traffic Signal Control

**Reinforcement Learning II — Final Project**

A comparative implementation of **A2C** and **PPO-GAE** on a custom 4×4 multi-intersection traffic grid environment with 144-dimensional state and a factored 16×Discrete(4) action space.

---

## Project Description

This project addresses adaptive traffic signal control in a multi-intersection urban grid using deep reinforcement learning. A centralized agent with a factored multi-head policy controls 16 intersections simultaneously, learning to minimize vehicle queue lengths and waiting times across the network.

The work is organized in five phases: environment construction, baseline A2C, PPO-GAE with 8 algorithmic improvements, multi-seed evaluation against classical baselines, and curriculum learning with TorchScript deployment export.

---

## Features

- **Custom 4×4 Traffic Environment** — Pure-Python `MultiIntersectionEnv` with Poisson vehicle arrivals, lane-level queue/waiting tracking, min-green phase constraint, and 720-step episodic horizon (~1 simulated hour at 5 s/step)
- **Baseline A2C** — Synchronous Advantage Actor-Critic with n-step returns, single-epoch updates, gradient clipping, and optional reward normalization
- **PPO-GAE with 8 Improvements** — Clipped surrogate objective (ε=0.2), Generalized Advantage Estimation (λ=0.95), RunningMeanStd reward normalization, gradient clipping, entropy coefficient decay (0.05→0.01), K=4 PPO epochs with mini-batching, value function clipping, and separate actor/critic optimizers
- **Curriculum Learning** — 3-stage demand scaling (30 % → 60 % → 100 %) with callback-based integration
- **3-Stage Evaluation** — Comparison against Random, Fixed-Time, and Actuated baselines; multi-seed PPO evaluation; hyperparameter grid search over ε, λ, lr
- **Logging & Visualization** — TensorBoard integration, matplotlib training curves, comparison bar charts, per-seed distribution plots
- **Streamlit UI** — 5-page interactive dashboard for training, evaluation, live grid simulation, checkpoint comparison, and embedded TensorBoard
- **Deployment** — TorchScript policy export for inference-only use
- **Optional SUMO Adapter** — Drop-in `SumoGridEnv` wrapper for the Eclipse SUMO traffic simulator

---

## Repository Structure

```
tp_final/
│
├── envs/
│   ├── traffic_grid_env.py      # MultiIntersectionEnv: 144-dim obs, Poisson arrivals, min-green
│   └── sumo_adapter.py          # Optional drop-in SUMO wrapper
│
├── models/
│   └── actor_critic.py          # SharedActorCritic: shared trunk → 16 actor heads + critic
│
├── algorithms/
│   ├── base_trainer.py          # ActorCriticTrainer(ABC) + TrainingCallback hook system
│   ├── rollout_buffer.py        # RolloutBuffer: GAEsment, n-step returns, mini-batch iteration
│   ├── a2c.py                   # A2CTrainer(ActorCriticTrainer): n-step returns, 1 epoch
│   └── ppo.py                   # PPOTrainer(ActorCriticTrainer): clipped surrogate, VF clip, separate optimizers
│
├── utils/
│   ├── running_stats.py         # RunningMeanStd: Welford online reward normalization
│   ├── curriculum.py            # CurriculumScheduler: 3-stage demand scaling
│   └── logger.py                # MetricLogger: TensorBoard writer + matplotlib plots
│
├── app/
│   ├── streamlit_app.py         # Streamlit entry point
│   ├── components/
│   │   ├── model_loader.py      # Checkpoint discovery and loading
│   │   └── grid_renderer.py     # Plotly 4x4 grid renderer
│   └── pages/
│       ├── 1_Train.py           # Launch training from UI
│       ├── 2_Evaluate.py        # Evaluate checkpoints
│       ├── 3_Live_Simulation.py # Step-by-step grid animation
│       ├── 4_Compare.py         # Multi-checkpoint comparison
│       └── 5_TensorBoard.py     # Embedded TensorBoard viewer
│
├── train_a2c.py                 # Phase 2: A2C training entry point
├── train_ppo.py                 # Phase 3+5: PPO-GAE training (uses callback for curriculum)
├── evaluate.py                  # Phase 4: multi-seed evaluation and benchmarking
├── sweep.py                     # Phase 4: hyperparameter grid search
├── tp_final_notebook.ipynb      # Interactive summary notebook (all 5 phases)
│
├── checkpoints/
│   ├── a2c/                     # A2C checkpoints (a2c_final.pt)
│   └── ppo/                     # PPO checkpoints (best, periodic, TorchScript)
│
├── results/
│   ├── a2c/                     # A2C TensorBoard logs, training curves, history
│   ├── ppo/                     # PPO TensorBoard logs, training curves, history
│   ├── eval/                    # Comparison tables and plots
│   └── sweep/                   # Sweep CSV and heatmap
│
├── requirements.txt
├── propose_tp_final.md          # Original project proposal
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.9+ (tested on 3.12)
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd tp_final

# Create a virtual environment (recommended)
python -m venv rl
source rl/bin/activate        # Linux/Mac
# .\rl\Scripts\Activate.ps1   # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:** gymnasium, torch, numpy, matplotlib, tensorboard
**Optional:** streamlit + plotly (UI), jupyter (notebook), sumo-rl (real simulator)

---

## Usage

All scripts run from the project root (`tp_final/`).

### Train A2C Baseline

```bash
python train_a2c.py --n_updates 200 --n_steps 128 --n_envs 16 --seed 0 --normalize_rewards
```

Output: `results/a2c/training_curves.png`, `checkpoints/a2c/a2c_final.pt`, TensorBoard logs.

### Train PPO-GAE

```bash
python train_ppo.py --n_updates 500 --n_steps 128 --n_envs 16 --seed 0
```

Output: Periodic checkpoints (`checkpoints/ppo/ppo_update_N.pt`), best model (`ppo_best.pt`), TorchScript export (`policy_scripted.pt`), training curves, TensorBoard logs.

### Train with Curriculum Learning

```bash
python train_ppo.py --n_updates 500 --n_steps 128 --n_envs 16 --curriculum --seed 0
```

Three-stage demand: 30 % → 60 % → 100 %, with reward-threshold-based stage advancement.

### Evaluate Policies

```bash
python evaluate.py --n_seeds 5 --n_eval_episodes 20
```

Compares Random, Fixed-Time, Actuated, A2C, and PPO. Saves `results/eval/comparison_table.txt` and comparison plots.

### Hyperparameter Sweep

```bash
python sweep.py --n_updates_sweep 100 --n_envs 4
```

Grid search over ε ∈ {0.1, 0.2, 0.3}, λ ∈ {0.9, 0.95, 0.99}, lr_a ∈ {3e-4, 1e-3}, lr_c ∈ {1e-3}. Saves CSV and heatmap to `results/sweep/`.

### Launch Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Interactive dashboard with live grid simulation, training launcher, checkpoint comparison, and embedded TensorBoard.

### Launch TensorBoard

```bash
tensorboard --logdir results
```

---

## Architecture Overview

### Agent Flow

1. **Observation** — 144-dim vector: per intersection, 4 normalized queue lengths + 4-dim phase one-hot + 1 normalized time scalar
2. **Shared Trunk** — Two-layer MLP (144→256→256) with ReLU activations, orthogonal weight initialization
3. **Multi-Head Actor** — 16 independent linear heads (256→4) producing categorical distributions per intersection
4. **Critic** — Two-layer MLP (256→128→1) producing scalar state-value V(s)
5. **Action Selection** — Independent categorical sampling per intersection → 16-dim discrete action

### Training Pipeline (PPO, one iteration)

1. **Warmup** (first update only) — 512 random-action steps seed the reward normalizer via RunningMeanStd
2. **Rollout Collection** — Run current policy π_θ for N=128 steps across E=16 vectorized envs → collect (s, a, r, log π, V(s), done) tuples
3. **Bootstrap + GAE** — δ_t = r_t + γV(s_{t+1}) − V(s_t); Â_t = Σ_{l≥0} (γλ)^l δ_{t+l}
4. **Advantage Normalization** — Standardize Â_t to zero mean, unit variance
5. **PPO Update (K=4 epochs)** — For each mini-batch: compute clipped surrogate policy loss, value function clipping, separate actor/critic optimizer steps, entropy bonus, KL-based early stopping
6. **LR Decay** — Linear schedule to 5 % of initial value
7. **Logging & Checkpointing** — TensorBoard metrics, periodic .pt saves, best-model tracking on raw reward

### Architecture: Clean Layer Separation

| Layer | Directory | Responsibility | Dependency Direction |
|-------|-----------|----------------|---------------------|
| **Domain** | `models/`, `algorithms/base_trainer.py` | Core RL abstractions: Agent, Policy, Environment interfaces | — |
| **Application** | `algorithms/a2c.py`, `algorithms/ppo.py` | Training orchestration, loss computation, checkpointing | → Domain |
| **Infrastructure** | `envs/`, `utils/`, `app/components/` | Gym wrappers, logging backends, storage | → Application |
| **Interface** | `train_a2c.py`, `train_ppo.py`, `app/` | CLI parsers, Streamlit UI, notebook | → All layers |

---

## CLI Reference

### train_a2c.py

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
| `--device` | `cpu` | Torch device |
| `--normalize_rewards` | off | Enable reward normalization |

### train_ppo.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_updates` | `500` | Number of PPO update cycles |
| `--n_steps` | `128` | Rollout steps per environment per update |
| `--n_envs` | `16` | Number of parallel environment instances |
| `--gamma` | `0.99` | Discount factor |
| `--gae_lambda` | `0.95` | GAE lambda (bias-variance tradeoff) |
| `--clip_epsilon` | `0.2` | PPO clipping threshold |
| `--actor_lr` | `3e-4` | Actor (policy) Adam learning rate |
| `--critic_lr` | `1e-3` | Critic (value) Adam learning rate |
| `--lr_decay` | on | Linear LR decay to 5 % of initial |
| `--lr_min_frac` | `0.05` | LR decay floor fraction |
| `--c_v` | `0.5` | Value loss coefficient |
| `--c_e_start` | `0.05` | Initial entropy coefficient |
| `--c_e_end` | `0.01` | Final entropy coefficient (linear decay) |
| `--n_epochs` | `4` | PPO update epochs per rollout (K) |
| `--batch_size` | `256` | Mini-batch size within PPO update |
| `--grad_clip` | `0.5` | Gradient clipping max norm |
| `--kl_target` | `0.02` | Approx KL threshold for early epoch stopping |
| `--warmup_steps` | `512` | Random steps to seed reward normalizer |
| `--save_freq` | `50` | Checkpoint every N updates |
| `--curriculum` | off | Enable 3-stage curriculum learning |
| `--no_norm_rewards` | off | Disable reward normalization |
| `--seed` | `0` | Random seed |
| `--device` | `cpu` | Torch device |

### evaluate.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_seeds` | `5` | Number of independent seeds for multi-seed eval |
| `--n_eval_episodes` | `20` | Episodes per policy per evaluation |
| `--a2c_ckpt` | `checkpoints/a2c/a2c_final.pt` | Path to A2C checkpoint |
| `--ppo_ckpt` | `checkpoints/ppo/ppo_best.pt` | Path to PPO checkpoint |
| `--demand` | `1.0` | Traffic demand factor for evaluation |
| `--device` | `cpu` | Torch device |

---

## RL Methodology

### Algorithms

| Algorithm | Advantage Estimation | Policy Update | Key Features |
|-----------|---------------------|---------------|--------------|
| **A2C** | n-step returns | Single full-batch gradient step | Static entropy (c_e=0.01), single optimizer |
| **PPO-GAE** | GAE (λ=0.95) | Clipped surrogate, K=4 epochs, mini-batches | Entropy decay (0.05→0.01), separate optimizers, VF clip, KL early-stop |

### Reward Strategy

```
r_t = -( Σ queue_length(i) / N_lanes ) - 0.1 × ( Σ waiting_time(i) / N_lanes )
```

Rewards are normalized by a running estimate of standard deviation (Welford's algorithm) to stabilize critic learning. Raw (un-normalized) rewards are tracked separately for evaluation comparison.

### Exploration

- **Entropy regularization** with linear decay (0.05 → 0.01) encourages broad phase exploration early and policy commitment late
- **Curriculum learning** (optional) starts at 30 % traffic demand, giving the agent a simpler learning task before facing peak congestion

### Optimization

- **PPO clipped surrogate** prevents destructively large policy updates
- **Value function clipping** mirrors PPO's clipping for the critic, bounding per-mini-batch value updates
- **Separate optimizers** (actor lr=3e-4, critic lr=1e-3) let the critic converge faster without dominating the actor gradient
- **LR decay with floor** (5 % of initial) avoids complete learning rate shutdown
- **KL early stopping** per epoch (threshold=0.02) prevents overfitting on stale data

---

## Results & Metrics

### Best Known Results (20 episodes, demand=1.0, raw reward)

| Policy | Mean Reward | ±Std | Mean Queue | Mean Waiting |
|--------|------------|------|------------|-------------|
| Random | −925.5 | 19.4 | 1.015 | 2.70 |
| Fixed-Time | −1008.8 | 12.9 | 1.116 | 2.85 |
| Actuated | −1879.5 | 10.3 | 1.925 | 6.86 |
| A2C (200 updates) | −2539.9 | 303.6 | 1.951 | 15.77 |
| **PPO-GAE fixed (500 updates, seed 0)** | **−534.6** | **5.3** | **0.616** | **1.26** |
| PPO-GAE curriculum (500 updates, seed 123) | −587.0 | 14.7 | 0.674 | 1.46 |

PPO-GAE achieves a **42 % improvement over the Random baseline** at 500 updates. The curriculum variant trails slightly (−587 vs −535) because it only had 200 updates at 100% demand (300 updates were spent at 30%/60% demand). With more updates at full demand, curriculum would likely surpass fixed-demand PPO.

### Multi-Seed Robustness (3 seeds × 200 updates)

| Variant | Mean ± Std | Seed 0 | Seed 1 | Seed 2 |
|---------|-----------|--------|--------|--------|
| Fixed-demand PPO | −830.1 ± 8.5 | −819.9 | −840.8 | −829.5 |
| Curriculum PPO | −852.2 ± 7.0 | −858.2 | −855.9 | −842.4 |

Low inter-seed variance (σ < 10) confirms training is stable across seeds. Curriculum trails at 200 updates because it only reaches stage 2 (60% demand) by then.

### Hyperparameter Sweep (100 updates × 4 envs)

Top configurations (sorted by final avg reward):

| ε (clip) | λ (GAE) | Actor LR | Final Reward |
|----------|---------|----------|-------------|
| 0.3 | 0.99 | 1e-3 | −3302 |
| 0.3 | 0.95 | 1e-3 | −3491 |
| 0.3 | 0.99 | 3e-4 | −3617 |
| 0.2 | 0.99 | 3e-4 | −3763 |
| 0.3 | 0.90 | 1e-3 | −3807 |

**Key takeaways:**
- Higher `actor_lr` (1e-3) wins in 4 of top 5 slots — current default of 3e-4 is conservative
- Higher λ (0.99) dominates the leaderboard — long-horizon advantage estimation helps
- Higher ε (0.3) appears in 4 of top 5 — more conservative clipping tolerates larger policy updates

All sweep runs degrade from ~−270 early to ~−3300+ final over 100 updates, indicating longer training is needed for convergence.

---

## Monitoring with TensorBoard

```bash
tensorboard --logdir results
```

**Available metrics:**

| Tag | Description | Target |
|-----|-------------|--------|
| `PPO/p_loss` | Clipped policy loss L^CLIP | Stable, near zero |
| `PPO/v_loss` | Value loss L^VF | Decreasing monotonically |
| `PPO/entropy` | Policy entropy H(π) | Gradual decrease; flag if < 0.1 |
| `PPO/kl_div` | Approximate KL divergence | Should stay below 0.02 |
| `PPO/grad_norm` | Gradient norm (before clipping) | Flag if consistently > 2.0 |
| `PPO/explained_var` | Explained variance of value function | Should approach 1.0 |
| `PPO/Avg_Reward` | Rolling episode reward (normalized) | Should increase over training |
| `PPO/Raw_Reward` | Rolling episode reward (raw, un-normalized) | For fair evaluation comparison |
| `PPO/Entropy_Coeff` | Current c_e value | Decreasing linearly |
| `PPO/Actor_LR` / `PPO/Critic_LR` | Current learning rates | Decreasing linearly |
| `A2C/P_Loss` | A2C policy gradient loss | — |
| `A2C/V_Loss` | A2C value loss | — |
| `A2C/Entropy` | A2C policy entropy | — |

---

## Reproducibility

All scripts accept a `--seed` argument that seeds NumPy, PyTorch, and the environment's internal RNG.

```bash
# Reproduce the exact training run
python train_ppo.py --seed 42 --n_updates 500 --n_steps 128 --n_envs 16

# Multi-seed evaluation
python multi_seed.py --n_updates 200 --seeds 0 1 2

# Hyperparameter sweep
python sweep.py --n_updates_sweep 100 --n_envs 4
```

**Experiment manifests**: Each training run saves a JSON manifest to `results/*/manifests/` containing the full config, seed, timestamp, git commit, and final metrics for full reproducibility.

**Pinned versions used during development:**

```
Python     3.12
gymnasium  0.29.x
torch      2.x
numpy      2.x
matplotlib 3.x
tensorboard 2.x
```

---

## Optional: SUMO Simulator

The project uses a self-contained pure-Python environment by default. To use the Eclipse SUMO traffic simulator:

**Step 1 — Install SUMO:**
- Windows: download the `.msi` installer from [eclipse.dev/sumo](https://eclipse.dev/sumo/)
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

Custom route files at multiple demand levels (0.3, 0.6, 1.0) can be generated:

```bash
python sumo/generate_routes.py --demand 0.6 --output_dir sumo/routes
```

### Cross-Validation Results

Evaluation of a PPO policy (trained purely in the Python simulator) transferred to SUMO:

| Demand | Python | SUMO | Delta |
|--------|--------|------|-------|
| 0.3 | −68.1 ± 13.2 | −157.8 ± 0.1 | −89.6 |
| 0.6 | −124.2 ± 19.3 | −237.6 ± 0.3 | −113.3 |
| 1.0 | −209.9 ± 33.6 | −355.0 ± 0.4 | −145.1 |

The policy qualitatively transfers (all phases active, vehicles handled), but absolute rewards differ due to:
- **Traffic dynamics**: SUMO models 3-lane approaches with car-following, lane-changing; Python is single-lane Poisson
- **Reward scale**: SUMO agent rewards use different normalization than the Python env's queue+waiting penalty
- **Observation mapping**: 8-phase/12-lane SUMO intersection mapped to 4-phase/4-lane Python format

The delta grows with demand (more vehicles ⇒ more dynamics differences), confirming the policy learned the right behavioral patterns but in a simplified dynamics model.

---

## Future Improvements

1. **Complete hyperparameter sweep** — Re-run `sweep.py` under the refactored trainer to validate optimal ε/λ/lr configuration
2. **Multi-seed PPO training** — Run 5 independent seeds end-to-end for statistical significance
3. **Increase rollout length** — Test N=512 (vs current 128) for improved GAE advantage estimation
4. **Train directly in SUMO** — Train PPO within the `SumoGridEnv` adapter (already compatible) and compare performance
5. **End-to-end tests** — Add GitHub Actions timing test that runs full 500-update PPO training
6. **Experiment dashboard** — Aggregate all manifest.json results into a single Streamlit comparison view

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. **Open an issue** to discuss proposed changes before submitting a PR
2. **Maintain the existing modular structure** — new algorithms go in `algorithms/`, new utilities in `utils/`
3. **Extend the base class** — new on-policy algorithms should extend `ActorCriticTrainer` from `algorithms/base_trainer.py`
4. **Add tests** for new components in the `tests/` directory
5. **Update the README** if you add or modify entry points or change default hyperparameters
6. **Preserve reproducibility** — all training scripts must accept a `--seed` argument

---

## License

Not specified in repository.

---

## References

- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
- Schulman et al. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016
- Mnih et al. (2016). *Asynchronous Methods for Deep Reinforcement Learning.* ICML 2016
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*
- FareedKhan-dev/all-rl-algorithms — Reference A2C notebook analyzed in project proposal

---

![footer](doc/imgs/LogoFooter.png)
