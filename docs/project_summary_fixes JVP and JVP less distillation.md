# DiffusionProject — What We Built & Achieved

## Project Goal
A clean, reproducible research codebase for comparing **Flow Matching (FM)** and **Mean Flow (MF)** generative models on CIFAR-10, with a shared training/evaluation pipeline that enforces apples-to-apples comparisons between algorithms.

---

## Architecture

```
DiffusionProject/
├── algorithms/          # All generative model logic lives here
│   ├── base.py              # BaseAlgorithm ABC (training_step + sample)
│   ├── r_embed.py           # Shared r-conditioning module (Linear→SiLU→Linear)
│   ├── flow_matching.py     # FM: CFM / Rectified Flow, uniform-t
│   ├── flow_matching_lognorm.py  # FM-LN: SD3 logit-normal t trick
│   ├── mean_flow.py         # MF: Mean Flow + FD-JVP, one/few-step sampling
│   ├── mean_flow_distill.py # MF-Distill: FM-teacher → student distillation
│   ├── mean_flow_adaptive_nfe.py  # [STUB] Future Dir 01: per-sample early exit
│   ├── mean_flow_multiscale.py    # [STUB] Future Dir 02: coarse-to-fine cascade
│   └── mock.py              # Smoke-test algorithm
├── models/backbone.py   # Shared SimpleUNet — FM and MF use identical architecture
├── training/trainer.py  # Generic training loop (zero algorithm-specific math)
├── sampling/sampler.py  # Generic sampler
├── evaluation/          # FID + IS evaluation
├── config/              # fm_full, fm_lognorm, mf_full, mf_distill_full JSONs
├── train.py             # CLI: python train.py --algorithm mf --config config/mf_full.json
├── evaluate.py          # CLI: re-evaluate any checkpoint at any NFE
├── train_all.ps1 / .sh  # One-command train-all convenience scripts
└── inference_server.py  # Local HTTP server + UI for interactive generation
```

---

## Algorithms Implemented

| Key | Algorithm | NFE at inference | Key idea |
|-----|-----------|-----------------|----------|
| `fm` | Flow Matching | 5–100 | Linear interpolant, Euler ODE |
| `fm_lognorm` | FM + Logit-Normal t | 5–100 | Better t-sampling (SD3 trick) |
| `mf` | Mean Flow (FD-JVP) | **1–4** | Learns average velocity; one-step capable |
| `mf_distill` | MF Distillation | **1–4** | FM teacher rollout → student supervision; no JVP needed |

### What makes `mf` special
- Implements the **Mean Flow Identity** (Geng et al. 2025): `u_tgt = v − (t−r)·du/dt`
- Uses **stochastic FD-JVP** instead of `torch.func.jvp`: 50% FD path (2 forward passes, no AMP restrictions), 50% diagonal path (`r=t`, exact target, 1 forward pass)
- Delta anneals linearly `1e-2 → 1e-4` over training via `on_epoch_end` hook

### What makes `mf_distill` special
- **No JVP at all** — teacher Euler rollout produces the target displacement directly
- Teacher (FM, 6.35M params) is loaded frozen from a checkpoint; only the student is optimised
- At inference: pure student displacement identity — teacher never called

---

## Engineering Achievements

### Fairness-enforcing pipeline
- `ExperimentConfig._protected_keys` — runtime assertion prevents `algorithm_kwargs` from shadowing shared controls (LR, batch size, epochs, etc.)
- `build_backbone()` — single construction point; FM and MF can never silently diverge in parameter count

### Clean separation of concerns
- `trainer.py` contains **zero** algorithm-specific mathematics
- Algorithms expose `training_step(batch) → {"loss": scalar}` + `sample(n, nfe, device)`
- `on_epoch_end(epoch, total)` hook for algorithm-side state updates (delta annealing)

### Checkpoint format
New format saves `model_state` + `extra_module_N_state` separately, so `evaluate.py` can load any algorithm's checkpoint without knowing its internal structure upfront.

### Inference UI
`inference_server.py` — local HTTP server with `inference_ui.html` for interactive side-by-side generation from any trained checkpoint.

---

## Future Directions (Stubbed, Ready to Implement)

| Module | Idea | Prerequisite |
|--------|------|-------------|
| `AdaptiveMeanFlowSampler` | Per-sample early exit based on velocity confidence | None — wraps any MF checkpoint |
| `MultiScaleMeanFlowPipeline` | Coarse 16×16 → fine 32×32 cascade | Two separate training runs + `z_init` in Sampler |

---

## To Train

```powershell
# All 4 algorithms (FM must finish before mf_distill)
.\train_all.ps1

# Skip already-trained ones
.\train_all.ps1 -SkipFm -SkipFmLognorm -SkipMf

# Just distillation
.\train_all.ps1 -Only mf_distill
```
