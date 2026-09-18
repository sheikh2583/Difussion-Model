# Flow Matching & Mean Flow — Comparative Study

A reproducible research codebase comparing six generative image-generation
algorithms on a **shared backbone, data pipeline, training loop, sampler, and
evaluation stack**. Built as a thesis project exploring the family of
flow-based generative models, from vanilla Flow Matching to one-step
Mean-Flow distillation.

Supports **CIFAR-10** (32 × 32) and **CelebA** (64 × 64), saves resumable
checkpoints, provides an automated full-tournament runner, and includes a
local browser UI for inspecting trained models.

---

## Table of Contents

1. [Quickstart](#quickstart)
2. [The Core Idea](#the-core-idea)
3. [Repository Map](#repository-map)
4. [Shared Foundation](#shared-foundation)
5. [Algorithms](#algorithms)
   - [Flow Matching (FM)](#1-flow-matching-fm)
   - [Flow Matching — Logit-Normal Time (FM-LN)](#2-flow-matching--logit-normal-time-fm-ln)
   - [Mean Flow (MF)](#3-mean-flow-mf)
   - [Mean Flow Distillation (MF-Distill)](#4-mean-flow-distillation-mf-distill)
   - [Consistency Models](#5-consistency-models)
   - [Rectified Flow Reflow](#6-rectified-flow-reflow)
6. [Unique Thesis Contributions](#unique-thesis-contributions)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Outputs & Inference UI](#outputs--inference-ui)
10. [Setup](#setup)
11. [Config Reference](#config-reference)

---

## Quickstart

**Windows (no command-line required):**

1. Double-click **`INIT_ALL.cmd`** — installs Python if needed, creates the
   environment, installs dependencies, and downloads CIFAR-10. CelebA is an
   optional large download so a quota failure cannot break first-time setup.
2. Double-click **`TRAIN.cmd`** — choose a model, then choose whether to
   continue its existing run or preserve it and start fresh.

**Linux:**

```bash
./init_all.sh
./train_interactive.sh
```

To initialize both datasets explicitly, run `INIT_ALL.cmd -Datasets all` on
Windows or `./init_all.sh --datasets all` on Linux. The checked-in
`requirements_frozen.txt` is the CUDA workstation snapshot, not a portable
installer; always use the initialization script on a new machine.

**Command-line (after setup):**

```bash
# Smoke test — CPU safe, ~30 s
python train.py --algorithm mock --config config/smoke_fast.json --mode fresh

# Train the three core thesis algorithms
python train.py --algorithm fm         --config config/fm_full.json --mode fresh
python train.py --algorithm fm_lognorm --config config/fm_lognorm_full.json --mode fresh
python train.py --algorithm mf         --config config/mf_full.json --mode fresh

# Full tournament — trains all six algorithms in dependency order
.\scripts\run_full_tournament.ps1 -Dataset cifar10   # Windows
./scripts/run_full_tournament.sh  --dataset cifar10  # Linux
```

---

## The Core Idea

All algorithms in this project are **flow-based generative models**. They learn
to transform Gaussian noise into realistic images by learning a vector field
over a linear probability path.

The high-level contrast between the three main thesis algorithms:

| | **FM** | **FM-LN** | **MF** |
|---|---|---|---|
| What the network predicts | Instantaneous velocity v(z_t, t) | Same as FM | Average velocity u(z_t, r, t) over [r, t] |
| Time sampling | Uniform t ~ U(0,1) | Logit-normal t = sigmoid(u), u~N | Uniform t, plus sampled r in [0, t] |
| Training target | epsilon - x_0 | epsilon - x_0 | Mean Flow Identity (JVP/FD) |
| Min useful NFE | ~5 | ~5 | **1** |
| Extra parameters | 0 | 0 | 323 (the r-embed MLP) |

---

## Repository Map

```text
train.py                 CLI entry point: --algorithm, --config, --mode
evaluate.py              Re-evaluate any checkpoint at arbitrary NFE values
bootstrap.py             Cross-platform environment & dataset setup

algorithms/
  base.py                BaseAlgorithm ABC: training_step() + sample()
  r_embed.py             Shared r-conditioning MLP (Linear->SiLU->Linear, 323 params)
  flow_matching.py       FM: CFM / Rectified Flow, uniform time
  flow_matching_lognorm.py  FM-LN: logit-normal time sampling (SD3 trick)
  mean_flow.py           MF: Mean Flow with FD-JVP + two-path stochastic training
  mean_flow_distill.py   MF-Distill: FM-teacher rollout -> student, no JVP
  mean_flow_adaptive_nfe.py  Adaptive per-sample early-exit sampler
  mean_flow_multiscale.py    Coarse-to-fine cascade sampling pipeline
  consistency.py         Consistency Models (Song et al. 2023)
  reflow.py              Rectified Flow Reflow (Liu et al. 2022)
  mock.py                Smoke-test stub

models/backbone.py       Shared time-conditioned SimpleUNet (6,352,899 params)
training/trainer.py      Generic training loop — zero algorithm-specific math
sampling/sampler.py      Generic sampler — calls algorithm.sample()
evaluation/
  evaluator.py           FID reference cache + per-NFE evaluation
  metrics.py             FID (torchmetrics) + Inception Score
experiments/runner.py    Builds shared objects; runs train->sample->evaluate

config/
  smoke_fast.json          2-epoch CPU smoke test
  fm_full.json             FM on CIFAR-10, 100 epochs, batch 128
  fm_lognorm_full.json     FM-LN on CIFAR-10, 100 epochs, batch 128
  mf_full.json             MF on CIFAR-10, 100 epochs
  mf_distill_full.json     MF-Distill on CIFAR-10, requires FM teacher
  consistency_full.json    Consistency Models on CIFAR-10
  reflow_full.json         Reflow on CIFAR-10, requires pairs artifact
  mf_coarse16.json         Coarse 16x16 MF model for multiscale pipeline
  *_celeba64.json          CelebA 64x64 variants of each algorithm

scripts/
  run_full_tournament.*        Full six-algorithm run with dependency ordering
  train_all.*                  Batch training helper
  evaluate_all.*               Batch evaluation helper
  generate_checkpoint_samples.py  Sample grids from every checkpoint
  generate_reflow_pairs.py        Generate 50k Reflow (z1, x0) pairs
  sample_mean_flow_extensions.py  Adaptive + multiscale inference CLI
  aggregate_results.py            Combine JSONL metrics -> CSV + comparison plots
  benchmark_training_flows.py     Pre-run GPU memory & time estimates
  verify_workflow.py              End-to-end sanity checks before a long run
  interactive_train.py            Non-interactive training menu

web/inference_server.py  Local HTTP inference server + browser UI

docs/
  THEORY_NOTES.md                Mathematical derivations and references
  PROJECT_INTERVIEW_GUIDE.md     Full algorithm walk-through for thesis defense
  IMPLEMENTATION_CHANGES.md      Engineering changelog
  TRAINING_TIME_ESTIMATES.md     GPU-specific time budgets
  CONSISTENCY_TUNING_NOTES.md    Consistency Model training tips

results/                 Generated outputs (gitignored)
data/                    CIFAR-10 / CelebA (gitignored)
```

---

## Shared Foundation

### Probability Path

Every algorithm is built on the same **linear interpolation path** between real
data and Gaussian noise:

```
z_t = (1 - t) * x_0 + t * epsilon,   epsilon ~ N(0, I),   t in [0, 1]
```

- `t = 0` is the **data** end.
- `t = 1` is the **noise** end (the known prior).

The instantaneous velocity along this path is:

```
v = d/dt [z_t] = epsilon - x_0
```

This is **not** a diffusion model. It does not involve a stochastic
reverse-time SDE or a DDPM noise schedule. It learns a **deterministic ODE**
whose vector field maps noise to data. Generation is: start from
`z_1 ~ N(0, I)` and integrate the ODE *backward* from `t=1` to `t=0`.

All images are normalized to `[-1, 1]` throughout.

### Shared SimpleUNet Backbone

`models/backbone.py` defines a single `SimpleUNet` used by **every** algorithm:

```
f_θ(x, t) : R^(3×H×W) × [0,1]  →  R^(3×H×W)
```

The network is intentionally **semantically neutral** — it does not know whether
its output is an instantaneous velocity, an average velocity, or a clean image
prediction. That meaning is assigned entirely by the algorithm wrapper.

**Architecture:**
- Sinusoidal time embedding → 2-layer MLP → injected into each ResBlock via
  a learned projection
- 3-level encoder/decoder with GroupNorm-8 + SiLU residual blocks
- Skip connections at every resolution scale
- Strided convolution downsampling; transposed-convolution upsampling
- **6,352,899 parameters** at the standard 32×32 CIFAR-10 configuration

All algorithms obtain their model through the single `build_backbone()` function
so architecture and parameter count **can never silently diverge** between methods.

### Fairness Enforcement

The runner rejects any key in `algorithm_kwargs` that would shadow a shared
control (learning rate, batch size, epochs, backbone, dataset, optimizer,
evaluation settings). This is enforced at runtime via
`ExperimentConfig._protected_keys`. An algorithm cannot, even accidentally,
improve its results by secretly adjusting optimizer hyperparameters.

---

## Algorithms

### 1. Flow Matching (FM)

> **Source:** `algorithms/flow_matching.py` · **Config:** `config/fm_full.json`
> **Reference:** Lipman et al. 2022; Liu et al. 2022 (Rectified Flow)

**What it learns:** The conditional instantaneous velocity `v = epsilon - x_0`
along the linear probability path.

#### Training Objective

```
L_FM(θ) = E[|| f_θ(z_t, t) - (epsilon - x_0) ||²],   t ~ U(0,1)
```

Steps per training batch:
1. Sample Gaussian noise `epsilon ~ N(0, I)`.
2. Sample time `t ~ U(0, 1)`, one per image.
3. Interpolate: `z_t = (1-t)*x_0 + t*epsilon`.
4. Compute target velocity: `v = epsilon - x_0`.
5. Predict: `v_hat = f_θ(z_t, t)`.
6. Loss: `MSE(v_hat, v)`.

#### Sampling (Reverse Euler)

Start from `z_1 ~ N(0, I)`, take `nfe` equal steps from `t=1` to `t=0`:

```
z ← z - f_θ(z, t_cur) * Δt
```

One backbone evaluation per step, so **NFE = number of Euler steps**.
Outputs are clamped to `[-1, 1]`.

---

### 2. Flow Matching — Logit-Normal Time (FM-LN)

> **Source:** `algorithms/flow_matching_lognorm.py`
> **Config:** `config/fm_lognorm_full.json`
> **Inspired by:** Esser et al. 2024 (Stable Diffusion 3)

**What it changes:** One line — the distribution from which training times are
drawn. Everything else (path, target, sampler, backbone) is identical to FM.

#### Logit-Normal Time Sampling

Instead of `t ~ U(0,1)`, draw:

```
u ~ N(μ, σ²),   t = sigmoid(u) = 1 / (1 + exp(-u))
```

The induced density on `t` is the **logit-normal**:

```
p(t) = 1 / (sqrt(2π) * σ * t*(1-t)) * exp(-(logit(t) - μ)² / (2σ²))
```

With default `μ=0, σ=1`, this **concentrates training in the intermediate
region** of the path (roughly `t ∈ [0.1, 0.9]`), where the regression problem
is most informative, and reduces the fraction of nearly degenerate training
samples near `t=0` and `t=1`.

**Key insight for the thesis:** FM-LN is not a new ODE or a new path. It is a
**non-uniform importance weighting** of the same flow-matching objective.
A model trained with FM-LN still uses the identical reverse-Euler sampler as FM.
Any improvement in FID comes from the network learning the vector field more
accurately, not from a better integration scheme.

Both `logit_mean` and `logit_std` are configurable via `algorithm_kwargs`.

---

### 3. Mean Flow (MF)

> **Source:** `algorithms/mean_flow.py` · **Config:** `config/mf_full.json`
> **Reference:** Geng et al. 2025, arXiv:2505.13447

Mean Flow is the **central algorithm** of this thesis. Instead of predicting an
*instantaneous* velocity at a point, MF predicts an **average velocity over an
interval [r, t]**:

```
u(z_t, r, t) = 1/(t-r) * integral_r^t v(z_τ, τ) dτ
```

This directly gives the **displacement identity**:

```
z_r = z_t - (t-r) * u(z_t, r, t)
```

At inference with `nfe=1`, this is the headline **one-step jump**:

```
z_0 = z_1 - u(z_1, 0, 1)
```

#### The Mean Flow Identity (Training Target)

Differentiating the displacement relation along the trajectory gives:

```
u(z_t, r, t) = v(z_t, t) - (t-r) * d/dt[u(z_t, r, t)]
```

where the total time derivative along the trajectory is:

```
d/dt[u] = (∂u/∂z_t)*v + ∂u/∂t
```

This requires a **Jacobian-vector product (JVP)** with tangent vector `(v, 0, 1)`.

#### The r-Embedding Module

The backbone signature is `f_θ(x, t)` — it takes one time input. MF needs two
inputs: `z_t`, `r`, and `t`. The solution is a small dedicated
**r-embedding MLP** (`algorithms/r_embed.py`):

```
REmbed:  Linear(1 → 64) → SiLU → Linear(64 → C)
```

Its output is a per-channel additive bias broadcast over (H, W) and added to
the image tensor **before** it enters the shared backbone:

```python
def forward(self, z, r):
    r_signal = self.net(r.unsqueeze(-1))    # (B, C)
    return z + r_signal[:, :, None, None]   # broadcast over H, W
```

This adds **323 parameters** (the only addition to the shared backbone count),
and is the sanctioned extension point for dual-time conditioning.

#### Sampling (Displacement Identity)

For `nfe` steps, construct `nfe + 1` times linearly from 1 to 0:

```
z ← z - (t_cur - t_next) * u(z, r=t_next, t=t_cur)
```

At `nfe=1`, this is the single noise-to-data jump. There is **no Euler
sub-step inside each interval** — the network directly predicts the displacement
over the whole requested range.

---

### 4. Mean Flow Distillation (MF-Distill)

> **Source:** `algorithms/mean_flow_distill.py`
> **Config:** `config/mf_distill_full.json`

**What it solves:** The standard Mean Flow Identity requires a JVP, which
constrains AMP usage and roughly doubles training cost per step. MF-Distill
eliminates this by using a **frozen FM teacher** to produce supervision targets.

#### Target Construction

Instead of the JVP-based Mean Flow Identity, the training target is computed
empirically via teacher rollout:

```python
z_r = frozen_FM.euler(z_t, from=t, to=r, steps=teacher_nfe)
u_tgt = (z_t - z_r) / (t - r)
```

- **Diagonal case** (`r ≈ t`, controlled by `p_same`): target degenerates to
  the teacher's instantaneous velocity (single forward pass).
- **Interval case** (`r < t`): teacher Euler-integrates from `t` down to `r`
  in `teacher_nfe` steps; the displacement is divided by the interval length.

The teacher is loaded frozen from a trained FM checkpoint; only the **student**
(same architecture, same `r_embed`) is optimized. At inference, the teacher is
never called — sampling uses the student's displacement identity, identical to
standard MF.

---

### 5. Consistency Models

> **Source:** `algorithms/consistency.py`
> **Config:** `config/consistency_full.json`
> **Reference:** Song et al. 2023, arXiv:2303.01469

Consistency Models learn a **consistency function** `f_θ(x_t, t) → x_0` that
maps *any* point on *any* trajectory directly to the clean image at `t=0`.

#### Boundary-Enforcing Parameterization

```
f_θ(x, t) = c_skip(t)*x + c_out(t)*F_θ(x, t)

c_skip(t) = 1 / (1 + t²)        ensures f_θ(x, 0) = x exactly
c_out(t)  = t / sqrt(1 + t²)
```

#### Training: Self-Consistency Distillation

```
L_CD = E[ || f_θ(x_t, t) - f_θ-(x_{t-Δt}, t-Δt) ||² ]
```

where:
- `x_{t-Δt} = x_t - Δt * v_teacher(x_t, t)` (one frozen-teacher Euler step)
- `f_θ-` is an **EMA** of the student network (updated after every optimizer step)

At inference: one-step sampling is `f_θ(z_1, 1)` — pure noise to image in a
single forward pass. Multi-step refinement alternates network application with
noise re-injection.

---

### 6. Rectified Flow Reflow

> **Source:** `algorithms/reflow.py` · **Config:** `config/reflow_full.json`
> **Reference:** Liu et al. 2022, arXiv:2209.14577

A trained FM model's learned paths are **curved** in practice. Reflow
straightens them.

#### Two-Stage Process

**Offline** (run `scripts/generate_reflow_pairs.py` first):
```
z_1 ~ N(0, I)
x_hat_0 = FM.euler(z_1, nfe=50)      <- full FM inference
save (z_1, x_hat_0) pairs to disk
```

**Training** (this file):
```
x_t = (1-t)*x_hat_0 + t*z_1          <- interpolate the SAME pair
v   = z_1 - x_hat_0                   <- constant velocity along this pair
loss = MSE(v_θ(x_t, t), v)
```

Because `z_1` and `x_hat_0` are now **deterministically paired**, the resulting
ODE path is straighter, requiring fewer Euler steps at inference. The training
loop is identical to FM — the data source changes, not the objective.

---

## Unique Thesis Contributions

This section documents the **novel implementation decisions** that distinguish
this codebase from a straightforward reproduction of published methods.

---

### A. Finite-Difference JVP for AMP-Compatible Mean Flow

**Problem:** The Mean Flow Identity requires `d/dt[u(z_t, r, t)]`, computed via
`torch.func.jvp`. PyTorch's functional JVP does not compose with
`torch.cuda.amp.autocast`, forcing either disabled AMP (roughly doubling training
time) or a precision mismatch that corrupts the target.

**Solution:** A **finite-difference approximation** of the JVP using two
standard forward passes — fully compatible with AMP:

```python
delta    = self.jvp_delta          # annealed scalar (see §C)
z_pert   = z_t + delta * v         # perturb along trajectory tangent
t_pert   = (t + delta).clamp(max=1.0)

u_pred      = self._forward(z_t,   r, t)
u_pred_pert = self._forward(z_pert, r, t_pert)
dudt        = (u_pred_pert - u_pred) / delta   # O(δ) bias
```

The FD step introduces `O(delta)` bias in the target. This is controlled by
annealing `delta` toward zero over training (see §C below).

---

### B. Two-Path Stochastic Training

Each Mean Flow training step is **stochastically routed** to one of two paths,
controlled by `p_fd_step` (default 0.5):

| Path | Probability | Method | Bias | Cost |
|---|---|---|---|---|
| **FD path** | `p_fd_step` | Two forward passes; finite-difference du/dt | O(δ) | 2× passes, full AMP |
| **Diagonal path** | `1 − p_fd_step` | Force r = t; target is exactly v | Zero | 1 pass, cheapest |

The diagonal path is **not** the same as the per-sample `p_same` diagonal
forcing inside the r-sampling. Both mechanisms are independent and complementary:

- `p_same` decides whether individual **samples within a batch** get `r = t`.
- `p_fd_step` decides whether the **entire step** uses the FD path or the exact
  diagonal target.

This combination provides an unbiased gradient signal on half the steps at zero
extra cost, while the FD steps provide the full Mean Flow Identity signal.

---

### C. Annealed FD Perturbation

The finite-difference step size `delta` is **linearly annealed** from
`jvp_delta_start` to `jvp_delta_end` over the full training run via an
`on_epoch_end` hook called by the trainer:

```python
@property
def jvp_delta(self) -> float:
    progress = self._epoch / max(self._total_epochs, 1)
    return self.jvp_delta_start + (self.jvp_delta_end - self.jvp_delta_start) * progress
```

| Parameter | Default | Meaning |
|---|---|---|
| `jvp_delta_start` | `1e-2` | Large delta at epoch 0 → stable early targets |
| `jvp_delta_end` | `1e-4` | Small delta at final epoch → low-bias targets |

Early in training, a large delta provides a stable (if biased) gradient signal
while the network is far from convergence. As training progresses and the
network stabilises, delta shrinks to reduce approximation bias for final-quality
targets.

---

### D. JVP-Free Distillation via Teacher Rollout

**Problem:** Computing the JVP requires differentiating through the network,
which is expensive and AMP-incompatible. Can a student learn Mean Flow's
interval-average prediction **without any JVP**?

**Solution:** Replace the analytical JVP target with an **empirical teacher rollout**:

```python
z_r   = frozen_FM_teacher.euler(z_t, from=t, to=r, steps=teacher_nfe)
u_tgt = (z_t - z_r) / (t - r)
```

The student sees the same training interface as MF (predicts `u(z_t, r, t)`,
uses the same `r_embed` module, same displacement-identity sampler at inference),
but its training targets come from running the frozen teacher rather than from
differentiating the network.

**Properties:**
- Zero JVP cost → full AMP compatibility, roughly half the training time per step.
- Student is **independent** of the teacher at inference.
- Teacher rollout is wrapped in `torch.no_grad()` → no memory overhead from the
  teacher's computation graph.
- The `_forward` alias makes MF-Distill fully compatible with
  `AdaptiveMeanFlowSampler`.

---

### E. Adaptive Per-Sample NFE Allocation

> **Source:** `algorithms/mean_flow_adaptive_nfe.py`
> **CLI:** `python scripts/sample_mean_flow_extensions.py adaptive --help`

A post-training inference wrapper that allocates a **different number of network
calls to each image** based on estimated generation difficulty:

```python
# At each step, predict clean image from current state:
velocity_to_zero = algorithm._forward(z[active], r=0, t=t_current)
prediction = z[active] - t_current * velocity_to_zero

# Measure change from previous step:
relative_change = ||prediction - prev_prediction|| / ||prediction||

# Retire this sample if converged:
if relative_change < confidence_threshold and nfe_used >= min_nfe:
    output[sample] = prediction
    active[sample] = False
```

Properties:
- Evaluates only **active (un-converged) samples** at each step, reducing GPU
  work when most samples converge early.
- Records `last_nfe_per_sample` (a per-image tensor) and `last_average_nfe`
  for analysis.
- The `r=0` clean-image prediction at every step means **early-exited samples
  never remain at `t > 0`** — all outputs are genuine clean-image candidates.
- Min/max NFE bounds and the confidence threshold are configurable.

---

### F. Coarse-to-Fine Multiscale Sampling

> **Source:** `algorithms/mean_flow_multiscale.py`
> **CLI:** `python scripts/sample_mean_flow_extensions.py multiscale --help`
> **Coarse config:** `config/mf_coarse16.json`

A two-stage generation pipeline that exploits Mean Flow's interval conditioning:

```
Stage 1 (coarse model, 16x16):
    z_coarse = MF_16x16.sample(n, nfe=coarse_nfe)

Stage 2 (fine model, 32x32, starting from t_start instead of t=1):
    z_fine = bilinear_upsample(z_coarse, 32x32)
    for t in linspace(t_start, 0.0, fine_nfe+1):
        displacement = MF_32x32._forward(z_fine, r=t_next, t=t_cur)
        z_fine = z_fine - (t_cur - t_next) * displacement
```

The key insight: because MF's displacement identity works for **any sub-interval
[r, t]**, the fine model can start at `t = 0.5` (partway through the trajectory)
with a bilinearly upsampled coarse draft, rather than from pure noise at `t = 1`.
This reuses coarse-level structure and focuses fine-model compute on the detail
refinement portion of the path.

---

### G. Configurable Logit-Normal Time Sampling

Standard logit-normal implementations hard-code `μ=0, σ=1`. This project
exposes both parameters as `algorithm_kwargs`:

```json
"algorithm_kwargs": {
  "logit_mean": 0.0,
  "logit_std":  1.5
}
```

A wider `logit_std` concentrates training further toward intermediate times;
a non-zero `logit_mean` shifts the concentration left or right along the path.
Non-positive `logit_std` is rejected at construction time with a clear error.

---

### H. Fairness-Enforcing Pipeline Design

A contribution to **research methodology**, not just software engineering:

1. **Single construction point** — `build_backbone()` is the only way to create
   a network. FM and MF cannot accidentally diverge in architecture or parameter
   count.

2. **Protected keys guard** — `algorithm_kwargs` is filtered at runtime against
   a set of shared controls. An algorithm cannot silently change batch size,
   learning rate, optimizer, backbone, or evaluation settings through its own
   configuration dictionary.

3. **Metadata-validated FID cache** — the reference statistics file is checked
   against a `.meta.json` sidecar recording the exact image count, resolution,
   and channel count. A run requesting a different configuration raises an error
   immediately rather than computing biased FID silently.

4. **Algorithm-agnostic trainer and sampler** — `trainer.py` contains zero
   algorithm-specific math. It calls `algorithm.training_step(batch)` and reads
   `out["loss"]`. Every algorithm is trained identically.

---

## Training

```bash
# Start a new single-algorithm run. Any existing canonical run is first moved
# into results/history/<run>_<timestamp>/ so it remains recoverable.
python train.py --algorithm fm          --config config/fm_full.json          --mode fresh
python train.py --algorithm fm_lognorm  --config config/fm_lognorm_full.json  --mode fresh
python train.py --algorithm mf          --config config/mf_full.json          --mode fresh
python train.py --algorithm mf_distill  --config config/mf_distill_full.json  --mode fresh
python train.py --algorithm consistency --config config/consistency_full.json --mode fresh

# Continue the latest checkpoint to its configured target
python train.py --algorithm mf --config config/mf_full.json --mode continue

# Extend a completed epoch-100 run to a total of 110 epochs
python train.py --algorithm mf --config config/mf_full.json --mode continue --epochs 110

# Full tournament (all six algorithms, dependency order, skips completed runs)
.\scripts\run_full_tournament.ps1 -Dataset cifar10   # Windows
./scripts/run_full_tournament.sh  --dataset cifar10  # Linux

# Start the whole tournament fresh while preserving old runs and Reflow pairs
.\scripts\run_full_tournament.ps1 -Dataset cifar10 -Mode fresh
./scripts/run_full_tournament.sh  --dataset cifar10 --mode fresh

# Dry run — prints plan, starts nothing
.\scripts\run_full_tournament.ps1 -DryRun

# Manual Reflow pair generation (required before training reflow)
python scripts/generate_reflow_pairs.py ^
  --checkpoint results/fm_cifar10/checkpoints/run_1/FlowMatchingAlgorithm_epoch100.pt ^
  --config config/fm_full.json ^
  --n-pairs 50000 ^
  --output data/reflow_pairs_cifar10.pt
```

Checkpoints are grouped by attempt under
`results/<experiment>_<dataset>/checkpoints/run_1/`, `run_2/`, and so on. Each
run directory contains its `.pt` files plus `archive/` with the self-contained
ZIP files. The underscore avoids quoting problems on Windows and Linux. Run:

```bash
python scripts/organize_checkpoints.py          # preview legacy migration
python scripts/organize_checkpoints.py --apply  # move flat files into run_1
```

Migration refuses name collisions and never overwrites a checkpoint. A fresh
run archives the previous logs/config/metrics under `results/history/` but keeps
the complete numbered checkpoint tree in the canonical experiment directory,
then writes into the next number. Resumption uses only the latest numbered run,
preventing checkpoints from separate attempts from being mixed.
Each `.pt` file is also packaged as a self-contained ZIP with weights, config,
and metadata. Resumption restores
model state, optimizer, scheduler, AMP scaler, counters, and RNG state.
If a canonical run directory is non-empty, direct `train.py` calls require an
explicit `--mode continue` or `--mode fresh`; this prevents accidental metric
mixing and checkpoint replacement. The beginner menu asks the same question.

### Mean Flow v2 error-fix workflow

Agents only prepare and statically validate this workflow; they do not execute
training or evaluation. After `config/mf_full_v2.json` is supplied by the
trainer/config track, run the static preflight first:

```powershell
# Windows
venv\Scripts\python.exe scripts\preflight_mf_v2.py --strict-evidence
venv\Scripts\python.exe train.py --algorithm mf --config config\mf_full_v2.json `
  --experiment-name mf_v2_probe --epochs 15 --mode fresh --train-only
```

```bash
# Linux
venv/bin/python scripts/preflight_mf_v2.py --strict-evidence
venv/bin/python train.py --algorithm mf --config config/mf_full_v2.json \
  --experiment-name mf_v2_probe --epochs 15 --mode fresh --train-only
```

Omit `--strict-evidence` on a clean clone, where the gitignored original
`results/mf_cifar10` directory is expected to be absent. `--train-only`
disables FID reference preparation, periodic evaluation, and final sampling.
The probe and full run use separate directories and scheduler horizons; never
continue the 15-epoch probe as the 100-epoch experiment.

### Latest local tournament status

The six-algorithm CIFAR-10 tournament completed on 18 September 2026, including
epoch-100 checkpoints, evaluation, and aggregation. Its local transcript is
`results/tournament_run_20260917_143512_pid10672.log`. These generated artifacts
remain gitignored and therefore are not included in a fresh clone. Completion
does not imply convergence: the MF run showed late loss divergence and remains
an active analysis item in `PLAN.md`.

---

## Evaluation

```bash
# Re-evaluate a checkpoint at all configured NFE values
python evaluate.py --algorithm fm \
  --checkpoint results/fm_cifar10/checkpoints/run_1/FlowMatchingAlgorithm_epoch100.pt \
  --config results/fm_cifar10/config.json --make-plots

# Evaluate all available epoch-100 checkpoints
.\scripts\evaluate_all.ps1 -Dataset cifar10   # Windows
./scripts/evaluate_all.sh  --dataset cifar10  # Linux

# Aggregate all JSONL metrics into a comparison CSV and plots
python scripts/aggregate_results.py
```

**Metrics recorded per NFE value:**
- FID (Frechet Inception Distance) — lower is better
- Inception Score (mean ± std) — higher is usually better
- Sampling time (wall clock, seconds)
- Time per image, images per second
- Peak GPU memory (MB)

**FID reference:** the first algorithm to run builds and caches a reference
statistics file from real dataset images. Every subsequent algorithm reuses the
same cache. The cache is validated against a metadata sidecar; a mismatch (e.g.,
different `num_images`) raises an error before training begins.

---

## Outputs & Inference UI

Each run writes to `results/<experiment>_<dataset>/`:

```text
config.json           Resolved configuration snapshot
logs/                 Training log
checkpoints/          Resumable .pt files + per-checkpoint ZIPs
samples/              Image grids at each evaluated NFE
metrics/              JSONL with training and evaluation records
```

```bash
# Visual grids from every discovered checkpoint
python scripts/generate_checkpoint_samples.py

# Adaptive Mean Flow inference (per-sample NFE allocation)
python scripts/sample_mean_flow_extensions.py adaptive \
  --checkpoint results/mf_cifar10/checkpoints/run_1/MeanFlowAlgorithm_epoch100.pt \
  --config results/mf_cifar10/config.json

# Coarse-to-fine cascade
python scripts/sample_mean_flow_extensions.py multiscale --help

# Local browser inference UI
python web/inference_server.py
# Open http://127.0.0.1:8000
```

---

## Setup

**Windows (automatic):**

```
Double-click INIT_ALL.cmd
```

**Linux (automatic):**

```bash
./init_all.sh
```

**Manual / selective:**

```bash
# CIFAR-10 only
python bootstrap.py --yes

# Both datasets
python bootstrap.py --yes --datasets all

# CelebA only, CPU build
python bootstrap.py --yes --datasets celeba --gpu cpu

# Override GPU detection
python bootstrap.py --yes --gpu cuda128   # CUDA 12.8
python bootstrap.py --yes --gpu rocm      # AMD ROCm
```

After setup, activate the environment:

```bash
source venv/bin/activate          # Linux/macOS
.\venv\Scripts\Activate.ps1      # Windows PowerShell
```

Before a long run, verify the full workflow:

```bash
python scripts/verify_workflow.py --dataset cifar10
python scripts/verify_workflow.py --strict-prerequisites
```

---

## Config Reference

Every JSON config has the same structure. `algorithm_kwargs` is the only section
that differs between algorithms.

```jsonc
{
  "experiment_name": "mf",
  "output_dir":      "./results",
  "seed":            0,
  "device":          "cuda",
  "amp":             true,
  "batch_size":      128,
  "epochs":          100,
  "checkpoint_frequency_epochs": 10,

  "dataset": {
    "name":       "cifar10",
    "root":       "./data/raw",
    "image_size": 32,
    "num_workers": 4
  },

  "backbone": {
    "name":           "simple_unet",
    "in_channels":    3,
    "base_channels":  64,
    "channel_mults":  [1, 2, 2],
    "num_res_blocks": 2,
    "time_embed_dim": 256
  },

  "optim": {
    "optimizer":     "adamw",
    "learning_rate": 0.0002,
    "weight_decay":  0.0,
    "scheduler":     "none"
  },

  "evaluation": {
    "metrics":               ["fid", "is"],
    "num_generated_samples": 5000,
    "nfe_values":            [1, 5, 10, 20, 50, 100],
    "eval_frequency_epochs": 25,
    "fid_reference_cache":   "./results/metrics/fid_reference_stats.npz"
  },

  // Algorithm-specific only — cannot shadow any key above
  "algorithm_kwargs": {}
}
```

**Algorithm-specific `algorithm_kwargs`:**

| Algorithm | Key | Default | Meaning |
|---|---|---|---|
| `fm_lognorm` | `logit_mean` | `0.0` | Mean of Gaussian before sigmoid |
| `fm_lognorm` | `logit_std` | `1.0` | Std of Gaussian before sigmoid (must be > 0) |
| `mf` | `p_same` | `0.25` | Probability of forcing r = t per sample |
| `mf` | `jvp_delta_start` | `1e-2` | Initial FD perturbation size |
| `mf` | `jvp_delta_end` | `1e-4` | Final FD perturbation size (after annealing) |
| `mf` | `p_fd_step` | `0.5` | Probability of FD path per training step |
| `mf_distill` | `teacher_checkpoint` | *(required)* | Path to trained FM .pt checkpoint |
| `mf_distill` | `teacher_nfe` | `4` | Euler steps for teacher rollout per step |
| `mf_distill` | `p_same` | `0.25` | Probability of diagonal shortcut |
| `consistency` | `teacher_checkpoint` | *(required)* | Path to trained FM .pt checkpoint |
| `consistency` | `ema_decay` | `0.999` | EMA decay for target network |
| `consistency` | `consistency_weight` | `1.0` | Loss scale |
| `consistency` | `n_timesteps` | `18` | Discrete time schedule steps |
| `reflow` | `pairs_path` | *(required)* | Path to .pt file of (z1, x0) pairs |

---

## Further Reading

| Document | Contents |
|---|---|
| `docs/THEORY_NOTES.md` | Mathematical derivations, proofs, and literature references |
| `docs/PROJECT_INTERVIEW_GUIDE.md` | Full algorithm walkthrough for thesis defense/interview |
| `docs/IMPLEMENTATION_CHANGES.md` | Engineering changelog and verification record |
| `docs/TRAINING_TIME_ESTIMATES.md` | Per-GPU time and memory budgets |
| `docs/CONSISTENCY_TUNING_NOTES.md` | Consistency Model training tips |
