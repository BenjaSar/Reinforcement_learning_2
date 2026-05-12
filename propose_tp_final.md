# A2C Notebook Analysis & Alternative RL Project Proposal

---

## Section 1 — Repository Summary

The `8_a2c.ipynb` notebook from `FareedKhan-dev/all-rl-algorithms` implements  **Advantage
Actor-Critic (A2C)** , the synchronous single-worker variant of the actor-critic family. The
implementation trains an agent across  **400 iterations** , logging policy loss (`P_Loss`),
value loss (`V_Loss`), and entropy per iteration. The combined loss function is defined as:

> L^{A2C}(θ, φ) = E_t[−log π_θ(a_t|s_t) · Â_t^{detached} − c_e · H(π_θ(·|s_t)) + c_v · (R_t − V_φ(s_t))²]

The notebook covers the theoretical derivation of A2C including n-step returns and GAE
formulation, the pseudocode for the training loop (collect N steps → compute advantages →
single gradient update), and an optional policy visualization section. It also documents
"Common Challenges and Solutions in A2C."

**Main RL Concepts Implemented:**

* Synchronous advantage actor-critic (A2C) with a shared rollout buffer
* Advantage estimation (n-step returns; GAE formulation is documented)
* Entropy regularization via coefficient `c_e`
* Combined actor + critic loss with value coefficient `c_v`
* Episode-level reward and length tracking

**Key Architectural Components (as observed):**

* Single actor network outputting a policy distribution
* Single critic network outputting a scalar state-value estimate V(s)
* Shared training loop with one gradient update per N-step rollout
* Training metrics: `P_Loss`, `V_Loss`, `Entropy`, `Avg Reward`, `Avg Len`

> **[ASSUMPTION]** The exact network layer sizes, activation functions, and optimizer type
> are not fully visible in the fetched content. Based on the repository's custom environment
> (avg reward peaking at ~7.89 with episodes of ~18 steps), the environment appears to be
> a small custom task — not a standard Gym environment such as CartPole-v1 or LunarLander.

---

## Section 2 — Identified Limitations or Extension Opportunities

### Scalability

The notebook trains a **single synchronous worker** over 400 fixed iterations on what
appears to be a low-complexity custom environment. The architecture does not implement
vectorized or parallel environments, meaning wall-clock training time scales poorly as
environment complexity increases. There is no observable checkpoint saving, making it
impractical to resume training on longer-horizon tasks.

### Environment Complexity

Observable reward values (max ~7.89, episode length ~18 steps at convergence) indicate
a **low-dimensional, short-horizon task** — likely a discrete-action custom environment.
The implementation is not validated on continuous action spaces, pixel-based observations,
or sparse-reward settings. Generalization to harder benchmarks (e.g., MuJoCo, Atari) is
not demonstrated.

### Exploration

Entropy regularization is present in the loss formulation (coefficient `c_e`), and the
training logs confirm entropy decreases from ~1.31 to ~0.63 over 400 iterations — a
healthy decay pattern. However, **no entropy coefficient schedule** (e.g., linear or
exponential decay) is observable. A static `c_e` may over-suppress entropy in later
training phases or fail to encourage adequate exploration early on.

### Sample Efficiency

A2C is on-policy by construction: each batch of N-step rollouts is discarded after a
single gradient update. The notebook performs  **one update per rollout collection** , making
no use of multiple update epochs on the same data (as PPO does). This is the primary
structural source of sample inefficiency in the current implementation.

### Stability

The training log shows **high value loss early in training** (`V_Loss: 14.36` at iter 50,
`V_Loss: 20.64` at iter 100), which then decays significantly by iter 400 (`V_Loss: 0.21`).
This pattern suggests early critic instability — a known risk when the actor begins updating
before the critic has produced reliable value estimates. No gradient clipping, reward
normalization, or value loss clipping is observable in the fetched content.

### PPO / Enhanced A2C Potential

The notebook documents GAE in the theory section but does not confirm its use in the
implementation. Replacing n-step returns with **full GAE (λ-returns)** would immediately
reduce advantage estimation variance. Replacing the A2C update with a **PPO clipped
surrogate objective** (ε-clipping on the probability ratio) would prevent destructively
large policy updates — a key weakness of vanilla A2C on harder tasks.

---

## Section 3 — Proposed Alternative Project

| Field                             | Description                                                                                                                                                                                                                                                                                                                                                                                                             |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Project Title**           | PPO-GAE Agent for Adaptive Traffic Signal Control in a Multi-Intersection Grid                                                                                                                                                                                                                                                                                                                                          |
| **Problem Description**     | Urban traffic congestion causes significant economic and environmental costs. Fixed-time traffic signal controllers cannot adapt to dynamic vehicle flow. An RL agent that learns to minimize vehicle wait times and queue lengths across a multi-intersection network represents a meaningful real-world optimization problem.                                                                                         |
| **Suitability for A2C/PPO** | Traffic signal control requires sequential decision-making under partial observability, with delayed consequences spanning multiple signal cycles. Actor-critic methods are well-suited because the critic provides a learned baseline that reduces gradient variance across long rollouts. PPO's clipping mechanism prevents policy collapse when reward shaping is imperfect — a common issue in traffic simulation. |
| **Proposed Environment**    | [SUMO](https://eclipse.dev/sumo/)(Simulation of Urban MObility) with the `sumo-rl`Python wrapper, using a 4×4 grid of intersections. Alternatively,`CityFlow`provides a lighter-weight simulation with OpenAI Gym compatibility.                                                                                                                                                                                      |
| **State Space**             | Per-intersection observation: queue length per lane (4 values), current phase (one-hot, 4 values), time since last phase change (1 scalar). For a 4×4 grid: 16 intersections × 9 features =**144-dimensional continuous vector** .                                                                                                                                                                              |
| **Action Space**            | Discrete: one action per intersection selecting the next active signal phase (4 phases × 16 intersections). In the centralized setting: a joint action of dimension 4^16 (impractical); in the decentralized setting: 16 independent agents each choosing 1 of 4 phases.**Recommended** : single centralized agent with a factored action space of 16 × 4 = 64 discrete outputs via multi-head policy.          |
| **Reward Function**         | r_t = −(Σ_i queue_length_i(t)) / N_lanes, where the sum is over all lanes across all intersections and N_lanes normalizes the magnitude. A secondary shaping term of −0.1 × Σ_i waiting_time_i(t) penalizes long individual waits. No reward is given for phase changes that are shorter than a minimum green time threshold (to prevent oscillation).                                                             |
| **Learning Objective**      | The policy should converge to a phase-switching strategy that minimizes mean vehicle queue length and waiting time per cycle, outperforming both a fixed-time baseline and an actuated signal controller on a 1-hour simulated traffic episode with stochastic vehicle demand.                                                                                                                                          |

---

## Section 4 — Technical Architecture

### Agent Architecture

A **shared-backbone design** with separate actor and critic output heads is recommended.
Sharing the lower layers forces the network to learn a common traffic-state representation
useful for both policy and value estimation, reducing total parameter count. The shared
trunk processes the 144-dimensional state vector into a 256-dimensional latent embedding.

### Actor Network

```
Input: state ∈ R^144
→ Linear(144, 256) + ReLU
→ Linear(256, 256) + ReLU
→ [Head i: Linear(256, 4) + Softmax] × 16 intersections
Output: 16 categorical distributions, one per intersection
```

The multi-head actor produces independent phase-selection distributions per intersection,
enabling a factored joint policy without exponential action-space blowup.

### Critic Network

```
Input: state ∈ R^144 (shared backbone output: R^256)
→ Linear(256, 128) + ReLU
→ Linear(128, 1) (no activation)
Output: scalar V(s) ∈ R
```

A single global critic estimates the centralized value function, which is appropriate for
cooperative multi-intersection optimization.

### Training Loop (One Iteration)

1. **Rollout collection** : Run the current policy π_θ for N=2048 environment steps across
   a vectorized environment (16 parallel SUMO instances). Collect (s_t, a_t, r_t, s_{t+1},
   done_t, log π_θ(a_t|s_t)) tuples.
2. **Bootstrap terminal value** : If the last state s_N is non-terminal, compute V_φ(s_N)
   as the bootstrap value.
3. **GAE computation** : Compute δ_t = r_t + γ V_φ(s_{t+1}) − V_φ(s_t). Compute
   Â_t^{GAE} = Σ_{l≥0} (γλ)^l δ_{t+l}. Compute returns R_t = Â_t + V_φ(s_t).
4. **Normalize advantages** : Standardize Â_t to zero mean and unit variance across the batch.
5. **PPO update (K=4 epochs)** : For each epoch, iterate over mini-batches of size 256:

* Compute probability ratio r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
* Clipped surrogate loss: L^{CLIP} = E_t[min(r_t Â_t, clip(r_t, 1−ε, 1+ε) Â_t)]
* Value loss: L^{VF} = E_t[(R_t − V_φ(s_t))²]
* Entropy bonus: L^{ENT} = E_t[H(π_θ(·|s_t))]
* Total loss: L = −L^{CLIP} + c_v · L^{VF} − c_e · L^{ENT}
* Gradient step with Adam; apply global gradient clipping (max norm = 0.5).

1. **Update θ_old ← θ** for next rollout.

### Experience Collection

* **16 parallel SUMO environments** (vectorized), each running an independent 1-hour
  traffic episode with stochastic demand.
* Rollout length N = 2048 steps total (128 steps per environment instance).
* On-policy: rollout buffer is discarded after each PPO update cycle.

### PPO Adaptation Details

The clipped surrogate objective prevents the updated policy from deviating excessively from
π_{θ_old}:

> L^{CLIP}(θ) = E_t[min(r_t(θ) Â_t, clip(r_t(θ), 1−ε, 1+ε) Â_t)]

where ε = 0.2 is the clipping threshold. This replaces the unconstrained A2C policy gradient
and is the key stability improvement over the analyzed notebook.

### Hyperparameter Considerations

| Hyperparameter               | Recommended Range | Notes                                     |
| ---------------------------- | ----------------- | ----------------------------------------- |
| `γ`(discount factor)      | 0.97–0.99        | Higher γ for long signal cycles          |
| `λ`(GAE lambda)           | 0.90–0.95        | Bias-variance trade-off in advantage      |
| `ε`(PPO clip)             | 0.1–0.2          | Start at 0.2; reduce if policy oscillates |
| `lr`(Adam)                 | 3e-4 – 1e-3      | Use linear decay schedule                 |
| `c_v`(value loss coeff)    | 0.5               | Standard from PPO paper                   |
| `c_e`(entropy coeff)       | 0.01 → 0.001     | Decay linearly over training              |
| `N`(rollout length)        | 1024–4096        | Longer for slower-changing traffic        |
| `K`(PPO epochs per update) | 4–10             | Monitor KL divergence; stop early if high |
| `batch size`               | 256–512          | Mini-batch size within PPO update         |
| `grad_clip`                | 0.5               | Prevents actor/critic divergence          |

---

## Section 5 — Recommended Improvements Over the Original Repository

### 1. PPO Clipped Surrogate Objective

 **What it is** : Replaces the unbounded A2C policy gradient with a clipped probability-ratio
objective that limits how much π_θ can shift from π_{θ_old} in a single update.
 **Why it addresses a limitation** : The original notebook shows high early value loss and
does not bound policy update magnitude. On a harder environment (144-dim state, multi-head
action), unconstrained A2C updates risk policy collapse. PPO clipping directly mitigates the
training instability risk identified in Section 2.

### 2. Generalized Advantage Estimation (GAE, λ-returns)

 **What it is** : A weighted average of n-step advantage estimates controlled by λ ∈ [0,1],
giving a tunable bias-variance trade-off: λ=0 → pure TD(0) (low variance, high bias);
λ=1 → Monte Carlo (low bias, high variance).
 **Why it addresses a limitation** : The notebook documents GAE theoretically but its use in
the actual implementation is not confirmed. For a multi-intersection environment with
non-stationary rewards across a 1-hour episode, standard n-step returns introduce
high-variance advantages. GAE with λ=0.95 provides a stable, practically proven trade-off.

### 3. Entropy Coefficient Scheduling

 **What it is** : Linear or exponential decay of `c_e` from an initial value (e.g., 0.01)
toward a floor (e.g., 0.001) over training.
 **Why it addresses a limitation** : The original notebook uses a static `c_e`. Entropy
should be high early (encouraging exploration of phase-switching strategies) and low late
(allowing the policy to commit to learned behaviors). A fixed coefficient cannot satisfy
both requirements simultaneously.

### 4. Vectorized / Parallel Environments

 **What it is** : Running N independent environment instances in parallel, collecting N
rollouts simultaneously using `gym.vector.SyncVectorEnv` or equivalent.
 **Why it addresses a limitation** : The original notebook operates a single-worker rollout.
For traffic simulation (SUMO is computationally expensive per step), parallel environments
are essential for collecting sufficient data within a practical wall-clock budget. 16
parallel instances provide a 16× throughput improvement.

### 5. Reward Normalization

 **What it is** : Normalizing rewards by a running estimate of their standard deviation
(e.g., using a `RunningMeanStd` buffer over recent episodes).
 **Why it addresses a limitation** : The original notebook shows raw V_Loss of 14.36 in early
training, suggesting the critic was initially overwhelmed by unscaled reward magnitudes.
Normalizing rewards to unit variance stabilizes value function learning and reduces critic
divergence risk.

### 6. Curriculum Learning

 **What it is** : Beginning training with a lower vehicle-demand scenario (e.g., 30%
of peak traffic) and progressively increasing demand as the agent's average reward
surpasses a threshold.
 **Why it addresses a limitation** : The proposed environment is more complex than the
original notebook's custom task. A cold-start on full peak-traffic demand produces very
sparse learning signal early on. Curriculum learning ensures the agent first learns basic
phase-switching logic before facing peak congestion.

### 7. Checkpointing and Experiment Tracking

 **What it is** : Saving model weights at fixed intervals (e.g., every 50 PPO update cycles)
using `torch.save`, and logging all metrics to **Weights & Biases (WandB)** or
 **TensorBoard** .
 **Why it addresses a limitation** : The original notebook has no observable checkpointing.
Training a SUMO simulation for hundreds of iterations is costly; losing progress due to an
unhandled exception or hardware failure is unacceptable. WandB also enables hyperparameter
sweep tracking across seeds.

### 8. Parallel Rollout Workers

 **What it is** : Beyond vectorized environments, dedicated rollout worker processes that
asynchronously push experience into a shared queue while the learner updates.
 **Why it addresses a limitation** : On CPU-bound environments like SUMO, data collection
(simulation) and learning (GPU gradient computation) can be pipelined, masking the
simulation latency and further increasing throughput over the single-thread design in the
original notebook.

---

## Section 6 — Implementation Roadmap

### Phase 1 — Environment Setup

**Deliverables:**

* Install `sumo-rl` or `cityflow` and verify SUMO 1.x compatibility
* Implement a `MultiIntersectionEnv` wrapper conforming to the `gymnasium.Env` interface
* Validate observation shape (144-dim), action space (16 × Discrete(4)), and reward signal
  on a single-step rollout
* Implement vectorized environment using `gymnasium.vector.SyncVectorEnv` with 16 instances
* Log a 10-episode random-policy baseline: mean queue length, mean waiting time, mean
  episode reward

### Phase 2 — Baseline A2C Implementation

**Deliverables:**

* Implement `SharedActorCritic` PyTorch module (shared trunk + multi-head actor + scalar critic)
* Implement n-step return advantage computation (matching the original notebook's approach)
* Run 200 PPO-style iterations with A2C update rule (no clipping, single epoch)
* Confirm learning signal: reward should improve vs. random-policy baseline within 50 iterations
* Establish baseline metrics: convergence curve, P_Loss, V_Loss, Entropy

### Phase 3 — PPO Migration / Improvement

**Deliverables:**

* Replace A2C update with PPO clipped surrogate (ε = 0.2, K = 4 epochs, mini-batch = 256)
* Replace n-step returns with GAE (λ = 0.95, γ = 0.99)
* Add reward normalization (RunningMeanStd)
* Add gradient clipping (max norm = 0.5)
* Add entropy decay schedule (linear from 0.01 → 0.001)
* Run 500 update cycles; compare convergence speed and final reward vs. Phase 2 baseline

### Phase 4 — Evaluation & Benchmarking

**Deliverables:**

* Define evaluation protocol: 20 held-out episodes with fixed random seed per evaluation
* Run 5 independent training seeds; report mean ± std of final reward
* Compare against: (a) random policy, (b) fixed-time controller, (c) Phase 2 A2C baseline,
  (d) `stable-baselines3` PPO with default hyperparameters
* Plot convergence curves: episode reward, queue length, waiting time over training steps
* Document hyperparameter sensitivity: sweep ε ∈ {0.1, 0.2, 0.3}, λ ∈ {0.9, 0.95, 0.99}

### Phase 5 — Optimization & Deployment

**Deliverables:**

* Enable curriculum learning: stage 1 (30% demand), stage 2 (60%), stage 3 (100%)
* Add WandB experiment tracking with full hyperparameter logging
* Add model checkpointing every 50 update cycles with best-model saving
* Export final policy as a TorchScript module for inference-only deployment
* Write a reproducibility README with exact version pins and seed instructions

---

## Section 7 — Evaluation Strategy

### Training Metrics (logged per update step)

* **Policy Loss** (`L^{CLIP}`): should stabilize and trend toward zero
* **Value Loss** (`L^{VF}`): should decrease monotonically after early transient
* **Entropy** (`H`): should decrease with training; watch for premature collapse (<0.1)
* **KL Divergence** (approx.): `E_t[log π_old − log π_new]`; stop PPO epoch early if > 0.02
* **Gradient Norm** : confirm clipping is active; flag if norm > 2.0 consistently
* **Explained Variance** : `1 − Var(R_t − V(s_t)) / Var(R_t)`; should approach 1.0

### Reward Convergence Analysis

Compute a 50-update rolling mean of episode reward. Declare convergence when the rolling
mean has not improved by more than 1% for 100 consecutive update cycles. Compare convergence
iteration count across seeds to assess consistency.

### Policy Stability Analysis

Track the standard deviation of episode reward across the 5 training seeds. A stable policy
should have inter-seed std < 15% of the mean final reward. Monitor entropy for oscillation
(rising entropy after declining indicates policy instability); flag as unstable if entropy
rises >20% from its minimum across two consecutive evaluation windows.

### Episode Performance Tracking

Per evaluation episode (not per update step):

* Mean vehicle queue length per intersection
* Mean vehicle waiting time (seconds)
* Total vehicles that completed their journey
* Throughput (vehicles cleared / simulated hour)

### Baseline Comparison

| Baseline                        | Description                                                     |
| ------------------------------- | --------------------------------------------------------------- |
| **Random Policy**         | Uniform random phase selection at each step                     |
| **Fixed-Time Controller** | Phase cycles with 30s green per phase, no adaptation            |
| **Actuated Controller**   | Simple rule: extend green if queue > threshold                  |
| **Phase 2 A2C**           | Baseline A2C from this project, no PPO enhancements             |
| **SB3 PPO (default)**     | `stable-baselines3`PPO with default hyperparameters, same env |

---

## Section 8 — Risks and Technical Challenges

### 1. Training Instability (Actor/Critic Divergence)

 **Nature** : The actor updates before the critic has converged to accurate value estimates,
causing gradient feedback based on unreliable advantages — a cycle that can escalate.
 **Likelihood** : Medium-high. The original notebook exhibits V_Loss of 14–20 in early
training on a simpler task. A 144-dim state space increases this risk.
 **Mitigation** : Gradient clipping (max norm 0.5), reward normalization, and increasing the
value loss coefficient `c_v` to 1.0 during early training phases.

### 2. Sparse or Deceptive Rewards

 **Nature** : Traffic queue length may remain consistently high for many steps during early
training (agent has not learned any useful phase logic), producing a nearly constant reward
signal with no gradient signal.
 **Likelihood** : Medium. Fixed-time controllers outperform random policies, but poorly
initialized RL agents may underperform even random at the start.
 **Mitigation** : Reward shaping: add a small positive reward for reducing queue length by
any amount (delta reward), which provides a denser training signal. Remove shaping term
once learning stabilizes (use a schedule).

### 3. Sample Inefficiency

 **Nature** : PPO is on-policy; each rollout is discarded after K update epochs. For a
computationally expensive SUMO simulation, this requires many simulation steps.
 **Likelihood** : High — this is a structural property of PPO, not a pathology.
 **Mitigation** : Maximize rollout reuse by increasing K (PPO epochs per update) up to 10,
monitor KL divergence to ensure the data remains on-policy, and use vectorized environments
for parallel data collection.

### 4. Sim-to-Real Gap

 **Nature** : SUMO traffic models make simplifying assumptions (deterministic vehicle
following, idealized lane changes) that may not reflect real signal controller behavior.
 **Likelihood** : Medium, if real-world deployment is a goal.
 **Mitigation** : Validate the trained policy in SUMO under multiple demand scenarios and
random seeds before any physical deployment. Use domain randomization (vary arrival rates,
vehicle mixes) during training to improve robustness.

### 5. Hyperparameter Sensitivity

 **Nature** : PPO performance is sensitive to the interaction of `lr`, `ε`, `λ`, `γ`, and
rollout length N. The original notebook does not demonstrate systematic hyperparameter
search.
 **Likelihood** : High — traffic control is a non-stationary problem where mistuned
hyperparameters cause oscillation.
 **Mitigation** : Run a structured grid search across ε ∈ {0.1, 0.2}, λ ∈ {0.9, 0.95},
and lr ∈ {3e-4, 1e-3} using WandB Sweeps. Adopt the best configuration for Phase 4
benchmarking.

### 6. Overfitting to a Fixed Environment Seed

 **Nature** : If training always uses the same vehicle demand pattern or random seed, the
policy may memorize demand timing rather than learning generalizable signal control.
 **Likelihood** : High, if stochastic demand is not implemented.
 **Mitigation** : Randomize vehicle arrival rates (±20% around a mean) and random seeds
across episodes. Evaluate on 20 held-out seeds that were never seen during training.

---

## Section 9 — Conclusion

The proposed project — applying PPO with GAE to multi-intersection adaptive traffic signal
control in SUMO — is technically viable: the environment provides a well-defined discrete
action space, dense reward signal, and clear convergence criteria, all of which are
prerequisites for reliable actor-critic training. PPO with GAE is the correct algorithmic
choice because it directly addresses the core limitations of the analyzed A2C notebook:
unconstrained policy updates, static entropy regularization, single-worker rollouts, and
absent reward normalization — each replaced by a principled, empirically validated
mechanism. Completing this project would deliver both engineering value (a deployable,
reproducible PPO pipeline with curriculum learning and experiment tracking) and research
value (a rigorous comparison of A2C vs. PPO stability and sample efficiency on a real-world
combinatorial control problem that extends well beyond the simple custom environment in the
original repository).

---

*Analysis grounded in observable content from `8_a2c.ipynb` (FareedKhan-dev/all-rl-algorithms).
Items not directly confirmed in the notebook are labeled  **[ASSUMPTION]** .*
