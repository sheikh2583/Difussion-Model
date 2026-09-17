# Theory Notes
# Mathematical reference for thesis: "How Fast Can Generative Models Get?"
# Updated as each algorithm is implemented — cite this in thesis chapters.

---

## Shared mathematical setup (all algorithms)

All methods operate on the same probability path between data and noise:

```
x_0 ~ p_data    (real image, t=0)
x_1 ~ N(0, I)   (pure noise, t=1)
x_t = (1-t)·x_0 + t·x_1    t ∈ [0,1]   (linear interpolation)
```

The instantaneous velocity along this path:
```
v = dx_t/dt = x_1 - x_0    (constant along the path — key property of linear paths)
```

All methods train a neural network f_θ(·) on this shared backbone (SimpleUNet,
6.35M parameters) and differ only in: what f_θ predicts, what loss is used,
and what is done at inference time.

---

## §1 — Flow Matching (FM)

**Paper:** Lipman et al. 2022, "Flow Matching for Generative Modeling"
          arXiv:2210.02747
          Liu et al. 2022, "Rectified Flow: A Marginal Preserving Approach to
          Optimal Transport" arXiv:2209.14577

**What the network predicts:** instantaneous velocity v_θ(x_t, t)

**Training loss:**
```
L_FM = E_{t~U(0,1), x_0~p_data, x_1~N(0,I)} [ ||v_θ(x_t, t) - v||² ]
     where x_t = (1-t)x_0 + t·x_1,  v = x_1 - x_0
```

**Inference:** Euler ODE integration from t=1 to t=0:
```
x_{t-Δt} = x_t - v_θ(x_t, t) · Δt      (NFE steps, step size Δt = 1/NFE)
```

**Key property:** Linear paths mean v is constant along each trajectory,
so in theory NFE=1 suffices if the model is perfect. In practice, model
error accumulates across the distribution, requiring NFE>>1.

**In codebase:** `algorithms/flow_matching.py`

---

## §2 — FM + Logit-Normal time sampling (FM-LN)

**Paper:** Esser et al. 2024, "Scaling Rectified Flow Transformers for
          High-Resolution Image Synthesis" (Stable Diffusion 3)
          arXiv:2403.03206

**What changes from FM:** only the time-sampling distribution during training.
```
FM:    t ~ Uniform(0, 1)
FM-LN: t = sigmoid(u),  u ~ N(0, 1)     (logit-normal distribution)
```

**Why:** The logit-normal distribution concentrates samples near t=0.5, the
region where the velocity field is hardest to learn (neither "mostly noise"
nor "mostly data"). Empirically improves FID for a fixed number of training
steps.

**Everything else** (loss, inference, backbone) identical to FM.

**In codebase:** `algorithms/flow_matching_lognorm.py`

---

## §3 — Mean Flow (MF)

**Paper:** Geng et al. 2025, "Mean Flows for One-Step Generative Modeling"
          arXiv:2505.13447

**What the network predicts:** average velocity over an interval [r, t]:
```
u(z_t, r, t) = (1/(t-r)) · ∫_r^t v(z_τ, τ) dτ
```

**Displacement identity** (key property enabling one-step generation):
```
z_r = z_t - (t-r) · u(z_t, r, t)
```
With r=0, t=1: one network call maps z_1 (noise) → z_0 (image).

**Mean Flow Identity** (training target, derived by differentiating the
displacement identity along the trajectory):
```
u_tgt(z_t, r, t) = v(z_t, t) - (t-r) · d/dt[u(z_t, r, t)]
```
where d/dt[u] = JVP of u w.r.t. (z_t, r, t) with tangent (v, 0, 1).

**Training loss:**
```
L_MF = E[ ||u_θ(z_t, r, t) - u_tgt||² ]
     where r ~ U(0, t),  u_tgt computed via JVP (exact) or FD (approximate)
```

**Our implementation uses FD-JVP** (finite-difference approximation):
```
d/dt[u] ≈ (u(z_t + δv, r, t+δ) - u(z_t, r, t)) / δ
         δ annealed 1e-2 → 1e-4 over training
```
Bias: O(δ). Advantage: runs under AMP, ~2× faster than exact JVP.

**Stochastic routing:** with probability (1 - p_fd_step), forces r=t and uses
exact target u_tgt=v (zero bias, single forward pass).

**r-conditioning:** backbone only accepts (x, t). r is encoded via a small
extra module REmbed: Linear(1→64) → SiLU → Linear(64→C), added as per-channel
spatial bias to x before the backbone. This module is reported separately
(not counted in the 6.35M shared backbone).

**In codebase:** `algorithms/mean_flow.py`, `algorithms/r_embed.py`

---

## §4 — Mean Flow Distillation (MF-Distill)

**Paper basis:** Combines Mean Flow (Geng et al. 2025) with the general
principle of score/flow distillation:
- Salimans & Ho 2022, "Progressive Distillation for Fast Sampling of
  Diffusion Models" arXiv:2202.00512
- Luo et al. 2023, "Diff-Instruct" arXiv:2305.18455

**Core idea:** Replace the expensive JVP in Mean Flow training with a
regression target derived from a frozen teacher FM model's Euler rollout:

```
z_r^teacher = FM_teacher.euler(z_t, t→r, NFE=4 steps)
u_tgt = (z_t - z_r^teacher) / (t - r)       [no derivatives needed]
loss  = MSE(u_θ(z_t, r, t), u_tgt.detach())
```

**Why this works:** The displacement identity says the true u_tgt is exactly
(z_t - z_r) / (t-r). The teacher's multi-step rollout approximates z_r
accurately (teacher_nfe=4 steps, inference-only, no grad). The student then
learns to reproduce this one-step.

**Fairness note:** Teacher (FM, 6.35M params, 100 epochs) cost is additive
and must be reported separately. Student backbone parameter count is identical
to FM/MF for the backbone comparison. Label all results "mf_distill" not "mf".

**In codebase:** `algorithms/mean_flow_distill.py`

---

## §5 — Consistency Models (CM)

**Paper:** Song et al. 2023, "Consistency Models"
          arXiv:2303.01469
          Song et al. 2023, "Improved Techniques for Training Consistency Models"
          arXiv:2310.14189

**What the network predicts:** a consistency function f_θ(x_t, t) that maps
any point on any trajectory directly to its origin x_0:
```
f_θ(x_0, 0) = x_0    (boundary condition: at t=0, output = input)
```

**Self-consistency constraint** (the key training signal):
Points on the same trajectory must map to the same x_0:
```
f_θ(x_t, t) ≈ f_θ(x_{t-Δt}, t-Δt)   for all t, Δt
```

**Training loss (consistency distillation):**
```
L_CD = E[ d(f_θ(x_t, t),  f_θ-(x_{t-Δt}, t-Δt)) ]
```
where:
- x_{t-Δt} = x_t - Δt · v_teacher(x_t, t)   [one Euler step with frozen FM teacher]
- f_θ-  is an EMA (exponential moving average) of f_θ with decay ρ=0.999
- d(·,·) is a perceptual distance (LPIPS or simple L2; we use L2 for simplicity)

**Boundary condition enforcement:**
```
f_θ(x, t) = c_skip(t)·x + c_out(t)·F_θ(x, t)
c_skip(t) = 1/(1 + t²),    c_out(t) = t/√(1 + t²)   [so f_θ(x,0) = x exactly]
```

**Inference (one step):**
```
x_0_hat = f_θ(x_1, 1)    [single network call from pure noise]
```
Multi-step: alternately apply f_θ and re-inject noise for refinement.

**Known training difficulties:**
- Loss scale: consistency_weight must be tuned per dataset.
- EMA decay: too slow → unstable; too fast → doesn't stabilize.
- Boundary condition: c_skip/c_out parameterization is critical for stability.

**In codebase:** `algorithms/consistency.py`
**Status:** implemented; empirical loss tuning is still required (see
`docs/CONSISTENCY_TUNING_NOTES.md`).

---

## §6 — Rectified Flow Reflow

**Paper:** Liu et al. 2022, "Flow Straight and Fast: Learning to Generate
          and Transfer Data with Rectified Flow"
          arXiv:2209.14577   (Section 3: Reflow)

**Core idea:** A trained FM model traces curved paths in practice (because
the learned field is the average of many straight paths, which curves). Reflow
straightens these paths by:

1. Generate N (noise, image) pairs using the trained FM model:
   ```
   z_1 ~ N(0,I)
   x_0_hat = FM.euler(z_1, NFE=50)   [run full integration]
   store (z_1, x_0_hat) as a training pair
   ```

2. Train a new FM model on these generated pairs instead of random pairing:
   ```
   x_t = (1-t)·x_0_hat + t·z_1      [interpolate between generated pair]
   v   = z_1 - x_0_hat               [constant velocity along this pair]
   L   = MSE(v_θ(x_t, t), v)
   ```

**Why it works:** The original FM pairs random noise z_1 with random data x_0.
After reflow, z_1 is paired with *its own generated x_0_hat* — a pair that
the model already knows how to connect — so the resulting path is straighter
and can be integrated with fewer Euler steps.

**Expected result:** FID vs. NFE curve shifts left (same FID at fewer steps).
This is the "clean before/after" result described in the thesis plan.

**In codebase:** `algorithms/reflow.py`
**Pair generator:** `scripts/generate_reflow_pairs.py` (run with `--help` for
the complete command interface)

---

## §7 — Comparison summary table

| Algorithm | Predicts | Loss involves | One-step? | Extra params | Paper |
|---|---|---|---|---|---|
| FM | v(x_t, t) | MSE velocity | No (approx) | None | Lipman 2022 |
| FM-LN | v(x_t, t) | MSE velocity | No (approx) | None | Esser 2024 |
| MF (FD-JVP) | u(z_t,r,t) | Mean Flow Identity (FD) | Yes (exact) | REmbed ~600p | Geng 2025 |
| MF-Distill | u(z_t,r,t) | Displacement target from teacher | Yes (exact) | REmbed + teacher | Geng+Salimans |
| CM | f(x_t, t) → x_0 | Self-consistency + EMA | Yes | c_skip/c_out | Song 2023 |
| Reflow | v(x_t, t) | MSE velocity on straight pairs | Closer to yes | None (FM retrain) | Liu 2022 §3 |

---

## §8 — Shared evaluation protocol

All methods evaluated identically:
- Dataset: CIFAR-10 32×32 (Phase 1), then CelebA 64×64 (Phase 1+)
- NFE sweep: [1, 2, 5, 10, 20] function evaluations
- FID: computed against 5000 real images (same reference set across all methods)
- IS: Inception Score on same 5000 generated samples
- Training cost: wall-clock time and peak GPU memory, logged per epoch via
  `utils/timing.py` and `utils/results.py`
- Backbone: SimpleUNet, 6.35M parameters, identical across FM/FM-LN/MF/CM/Reflow.
  MF-Distill: student is 6.35M, teacher is additional 6.35M (reported separately).

---

## §9 — Papers to read before implementing each phase

Phase 2 (MF-Distill): Geng 2025 §3, Salimans 2022 §2
Phase 3 (CM): Song 2023 §3–4 (full paper), Song 2023 "Improved" §2–3
Phase 4 (Reflow): Liu 2022 §3 (3 pages — very short, read all of it)
Phase 5 (write-up): All of the above, plus:
  - Ho et al. 2020 "DDPM" (background context for intro chapter)
  - Song et al. 2020 "Score SDE" (background context)
