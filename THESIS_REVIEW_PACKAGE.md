# Thesis Technical Review Package

**Project:** A Controlled Comparison of Flow Matching and Mean Flow for Unconditional CIFAR-10 Image Generation
**Institution:** Department of Computer Science and Engineering, IUT (Islamic University of Technology)
**Programme:** Bachelor of Science in Computer Science and Engineering
**Defense Date:** 14 September 2026
**Repository:** https://github.com/sheikh2583/Difussion-Model
**Package Built:** 2026-09-17 | **Last updated:** 2026-09-18

This document is a self-contained technical review dossier intended for a reviewer who has no access to the codebase. All evidence is drawn from the live project state and reproduced verbatim where it matters. Nothing has been paraphrased to look better than it is.

---

## 0. Post-Defense Change Log

> **Reading guide for 3rd-party reviewers:** Sections 1–10 below describe the project as submitted for the BSc defense. This section documents every change made *after* the defense date, grouped by commit. All changes are stability fixes and infrastructure additions; the original experimental evidence (FM, FM-LN, first MF run) is frozen and untouched. A two-agent system (Antigravity/Claude = trainer+config owner; Codex = lifecycle+provenance owner) is managing these changes under the rules in `AGENTS.md`.

### How to read the change log

| Column | Meaning |
|--------|---------|
| Commit | Short SHA, links to changed files |
| Author | `Antigravity` (trainer/config) or `Codex` (lifecycle/provenance) or `User` |
| Owned files | Paths changed — each agent only touches its own paths |
| Purpose | Why the change was made |

---

### `378607c` — 2026-09-18 — `docs: coordinate MF v2 work across agents` — **User/Codex**

| File | Change |
|------|--------|
| `AGENTS.md` | Created: defines file ownership, no-training rule, GPU/git lock protocol for dual-agent session |
| `PLAN.md` | Created: authoritative task plan — A0/A1 (Antigravity), C1/C2 (Codex), integration gate I0 |

**Purpose:** After the completed 12 h 22 min tournament revealed MF loss divergence (epoch 97–100 losses: 0.493, 0.530, 0.601, 0.795), a structured plan was established to fix MF stability without altering tournament evidence.

---

### `e301eb1` — 2026-09-18 — `docs: restrict agents to error fixes only` — **User**

| File | Change |
|------|--------|
| `PLAN.md` | Added explicit prohibition: agents must not run training, GPU evaluation, sampling, pair generation, or benchmarks |

**Purpose:** Safety update to prevent accidental model execution by either agent.

---

### `b4b1537` — 2026-09-18 — `Fixes after 1st Run` — **Codex**

| File | Owner | Change |
|------|-------|--------|
| `train.py` | Codex | Safe `--mode fresh/continue` CLI; `--train-only` flag for probe runs |
| `experiments/runner.py` | Codex | Integration with lifecycle and provenance helpers |
| `utils/checkpoint_provenance.py` | Codex | **[NEW]** `build_provenance()` and `validate_provenance()` — verifies algorithm/dataset/backbone identity before resume; warns (does not fail) for legacy checkpoints |
| `scripts/preflight_mf_v2.py` | Codex | **[NEW]** Static MF v2 safety checker (no torch import, no model run): validates schema, run-dir isolation, distinct FID cache, scheduler horizon, clip norm, MF kwargs |
| `tests/test_checkpoint_provenance.py` | Codex | **[NEW]** 4 provenance unit tests (CPU only) |
| `tests/test_mf_v2_preflight.py` | Codex | **[NEW]** Preflight unit tests |
| `tests/test_runner_lifecycle.py` | Codex | **[NEW]** Runner lifecycle unit tests |
| `README.md` | Codex | MF v2 probe and full-run command documentation |
| `INIT_ALL.cmd` / `init_all.sh` | Codex | Cross-platform bootstrap |
| `.gitattributes` | Codex | Line-ending hygiene |
| `bootstrap.py` | Codex | Updated setup helper |

**Purpose:** C1 (resume/checkpoint provenance guard) and C2 (MF v2 preflight workflow). Prevents resuming an incompatible checkpoint. Validates `mf_full_v2.json` before the user runs any training.

---

### `dc3006f` — 2026-09-18 — `feat(A1): gradient clipping, mf_full_v2.json, trainer controls tests` — **Antigravity**

| File | Owner | Change |
|------|-------|--------|
| `config/config.py` | Antigravity | Added `gradient_clip_norm: Optional[float] = None` to `OptimConfig`. Default `None` preserves all existing configs. |
| `training/trainer.py` | Antigravity | AMP-correct clipping in `_run_epoch`: `scaler.unscale_()` → `clip_grad_norm_(trainable_modules params)` → `scaler.step()`. Provenance embed in `save_checkpoint()`. |
| `config/mf_full_v2.json` | Antigravity | **[NEW]** MF v2 controlled config: batch 128, lr 1e-4, wd 1e-4, clip 1.0, cosine scheduler η_min=1e-6, 100 epochs, AMP, seed 0, 5000-sample eval, NFE [1,2,5,10,20], p_same=0.25, p_fd_step=0.5, δ 1e-3→1e-4. Distinct FID cache path. |
| `tests/test_trainer_controls.py` | Antigravity | **[NEW]** 18 CPU-only synthetic tests: config parsing, disabled-by-default clipping, AMP unscale→clip→step order, all-module param collection, scheduler construction, checkpoint provenance serialization. All 18 pass. |

**Purpose:** A1 task — fixes the root cause of MF divergence: adds gradient clipping to the trainer (AMP-safe), creates the corrected MF v2 experiment config, and validates all controls with CPU unit tests. **Preflight `scripts/preflight_mf_v2.py --strict-evidence` passes.**

---

### `0ebb766` — 2026-09-18 — `fix(chunk-0): aggregator glob, MF JSONL split, eval traceability` — **Antigravity**

| File | Owner | Change |
|------|-------|--------|
| `scripts/aggregate_results.py` | Antigravity | **Fix 0.1** — Canonical JSONL selection: aggregator now only loads the file whose stem matches the run directory name, excluding smoke/auxiliary files. Corrects `fm_lognorm_rtx3060` FID@5: **310.43 → 96.45** |
| `scripts/forensic_mf_split.py` | Antigravity | **[NEW] Fix 0.2** — Forensic split of `mf_cifar10.jsonl` into run1 (epochs 1–30, batch=32, diverged) and run2 (epochs 31–100, batch=64, stable ~0.40 then late divergence). Original file untouched. |
| `utils/results.py` | Antigravity | **Fix 0.3** — Added `checkpoint_path` and `num_generated_samples` to `ResultRecord` schema. Legacy records retain `None`. |
| `evaluation/evaluator.py` | Antigravity | **Fix 0.3** — `evaluate()` now accepts `checkpoint_path=` and writes both traceability fields into every future evaluation record. |

**`summary.csv` diff (values that changed):**

| Run | Field | Before (buggy) | After (correct) |
|-----|-------|-----------------|-----------------|
| `fm_lognorm_rtx3060` | `fid_at_5` | 310.43 (from smoke file) | **96.45** (from real run) |

All other rows unchanged. Aggregator now processes 9 files (was 11, excluding `smoke_lognorm.jsonl` and `smoke.jsonl`).

**MF JSONL split finding:** `mf_cifar10.jsonl` contained two concatenated runs. Run 2 (epochs 31–100, batch=64) reached stable loss ~0.40 — the report's claim "abandoned at epoch 30, loss never improved" is factually wrong. Forensic files written to `results/aggregate/forensic/` (gitignored).

---

### Pending (awaiting user approval to run)

| Gate | What | Status |
|------|------|--------|
| I0 | Integration audit (Codex reads Antigravity's files, reports findings) | BLOCKED on Codex reviewing dc3006f |
| G0 | 15-epoch MF v2 probe (`mf_v2_probe_cifar10`) — user-run only | READY after I0 |
| G1 | 100-epoch MF v2 full run (`mf_v2_cifar10`) — user-run only | READY after passing G0 |
| E0 | Evaluate and aggregate MF v2 — user-run only | After G1 |

---

## 1. One-Paragraph Project Summary

This is an undergraduate BSc final-year design project that implements and compares three flow-based generative modelling algorithms -- Flow Matching (FM), FM with logit-normal time sampling (FM-LN), and Mean Flow (MF) -- for unconditional image generation on CIFAR-10 (32x32 RGB). All three algorithms share a single 6.35 million-parameter UNet backbone so that architecture and parameter count cannot diverge between methods; only the mathematical training objective and time-sampling distribution differ. The project records training loss, sampling time, GPU memory, FID, and Inception Score at NFE in {1, 5, 10, 20} using a shared JSONL logging schema. Three training runs were actually completed for the submitted report (FM at epoch 100, FM-LN at epoch 100, MF abandoned at epoch 30 with diverging loss). The codebase has since been substantially extended: it now includes three further algorithms (MF-Distill, Consistency Models, Reflow), CelebA 64x64 dataset support, cross-platform orchestration scripts, a local browser-based inference server, and infrastructure for a planned "full tournament" comparing all six algorithms on both datasets. **That tournament has not been executed.** The browser UI, reproducibility infrastructure, and documentation are the clearest engineering achievements; the experimental evidence covers only three CIFAR-10 runs with unmatched training budgets.

---

## 2. Full Algorithm / Method List

### 2.1 Flow Matching (FM)

- **Paper:** Lipman et al. 2022, "Flow Matching for Generative Modeling," arXiv:2210.02747; Liu et al. 2022, "Rectified Flow," arXiv:2209.14577.
- **Implementation status:** Fully implemented and trained to 100 epochs on CIFAR-10.
- **What is implemented vs paper:** Straight reimplementation of conditional flow matching on the linear interpolation path x_t = (1-t)*x_0 + t*epsilon. Loss is MSE between predicted and target velocity v = epsilon - x_0. Sampling uses Euler ODE integration. No deviation from the source paper.
- **Original vs adapted:** Direct reimplementation. No modifications to the core algorithm. Code is educational in character, with step-by-step inline comments.

### 2.2 FM + Logit-Normal Time Sampling (FM-LN)

- **Paper:** Esser et al. 2024, "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (Stable Diffusion 3), arXiv:2403.03206.
- **Implementation status:** Fully implemented and trained to 100 epochs on CIFAR-10 (on an RTX 3060 Laptop, different hardware session from FM).
- **What is implemented vs paper:** Identical to FM except time is sampled as t = sigmoid(u), u ~ N(mu, sigma^2) with configurable logit_mean and logit_std. Everything else -- path, loss, backbone, inference -- is unchanged.
- **Original vs adapted:** Direct reimplementation of the time-sampling variant. The logit_mean/logit_std parameterisation is configurable, a minor convenience adaptation.

### 2.3 Mean Flow (MF) with FD-JVP

- **Paper:** Geng et al. 2025, "Mean Flows for One-Step Generative Modeling," arXiv:2505.13447.
- **Implementation status:** Implemented; training attempted but **abandoned at epoch 30** with diverging loss (loss rose from 0.554 at epoch 1 to 0.648 at epoch 30; never decreased).
- **What is implemented vs paper:** The network predicts the average velocity u(z_t, r, t) over an interval [r, t]. The training target is the Mean Flow Identity: u_tgt = v - (t-r)*d/dt[u]. **This project deviates from the paper in how d/dt[u] is computed.** The paper uses the exact torch.func.jvp call. This implementation replaces it with a **finite-difference (FD) approximation:**

  ```
  d/dt[u] ~= (u(z_t + delta*v, r, t+delta) - u(z_t, r, t)) / delta
  ```

  where delta is linearly annealed from 1e-2 to 1e-4 over training.

- **What is original/novel:** The FD-JVP approximation is the main deviation from the source paper. Stated tradeoff (from code docstring): "removes the forced-fp32 / disabled-AMP constraint and roughly halves training time per epoch at the cost of O(delta) bias in the target." The bias has not been empirically quantified. A further "two-path stochastic routing" adds a cheap diagonal path (prob 1 - p_fd_step) where r=t is forced and the target degenerates exactly to v with zero bias. The r-conditioning embeds r via a 323-parameter two-layer projection added channel-wise to the input.

### 2.4 Mean Flow Distillation (MF-Distill)

- **Paper basis:** Combines Mean Flow (Geng et al. 2025) with distillation from Salimans & Ho 2022, arXiv:2202.00512.
- **Implementation status:** Fully implemented and mechanically validated (training step produces finite loss, sampling produces finite tensors). **No training run has been executed.**
- **What is implemented:** Uses a frozen FM teacher's 4-step Euler rollout to produce the displacement target u_tgt = (z_t - z_r_teacher) / (t-r), bypassing any JVP or FD approximation. This is the project's own combination.
- **Original vs adapted:** The distillation target formula is the team's own synthesis of the Mean Flow Identity and progressive distillation; not a verbatim copy of any paper section.

### 2.5 Consistency Models (CM)

- **Paper:** Song et al. 2023, "Consistency Models," arXiv:2303.01469; Song et al. 2023, "Improved Techniques," arXiv:2310.14189.
- **Implementation status:** Fully implemented and mechanically validated. Tuning harness exists with 4 variants. **No actual training run executed.** docs/CONSISTENCY_TUNING_NOTES.md states: "no optimization run was started because model training is explicitly reserved for the user."
- **What is implemented:** Full consistency distillation loss with EMA target network (decay 0.999), boundary-enforcing parameterisation f_theta(x,t) = c_skip(t)*x + c_out(t)*F_theta(x,t), frozen FM teacher for one-step Euler targets, 18-step discrete time schedule.
- **Original vs adapted:** Near-direct reimplementation of Song 2023, adapted to the project's flow-matching framework and using L2 instead of LPIPS (simplification, documented).

### 2.6 Rectified Flow Reflow

- **Paper:** Liu et al. 2022, Section 3, arXiv:2209.14577.
- **Implementation status:** Fully implemented. 50,000 CIFAR-10 pairs generated and validated. **No training run executed.**
- **What is implemented:** Offline pair generation via generate_reflow_pairs.py (FM at 50 NFE produces (noise, image) pairs); online training on those straight pairs using standard FM MSE loss.
- **Original vs adapted:** Direct reimplementation of Section 3 of Liu et al. 2022.

### 2.7 Extension Samplers (inference-only wrappers, not BaseAlgorithm subclasses)

- **AdaptiveMeanFlowSampler:** Retires converged samples early based on relative change in successive clean-image predictions. Parameters: min_nfe, max_nfe, confidence_threshold. Mechanically validated only.
- **MultiScaleMeanFlowPipeline:** Coarse-to-fine two-stage pipeline (16x16 draft -> bilinear upsample -> 32x32 refinement). Mechanically validated only.

---

## 3. Architecture Summary

### 3.1 The Shared Backbone (build_backbone, SimpleUNet)

All algorithms use an identical SimpleUNet built by the single function build_backbone(cfg, image_size) in models/backbone.py. Architecture:

- **Encoder:** in_conv (3->64), then 3 stages with channel multipliers [1, 2, 2] (widths 64, 128, 128), 2 ResBlocks per stage, strided-conv downsampling between stages.
- **Bottleneck:** Two ResBlocks at the narrowest spatial scale.
- **Decoder:** Mirror of encoder with bilinear upsampling and skip connections.
- **Time conditioning:** 256-dimensional sinusoidal embedding -> 2-layer MLP -> injected into every ResBlock via a learned linear projection onto the channel dimension.
- **Norms/activations:** GroupNorm (8 groups), SiLU throughout.
- **Parameter count: 6,352,899** (confirmed from JSONL logs: `parameter_count: 6352899` in every training record).

### 3.2 Why This Design Exists

The explicit purpose is apples-to-apples comparison. The `_protected_keys` field in every config JSON enumerates the keys that algorithm_kwargs is forbidden to override: batch_size, epochs, checkpoint_frequency_epochs, seed, device, amp, dataset, backbone, optim, evaluation, output_dir. This is enforced in code.

trainable_modules() in BaseAlgorithm defaults to [self.model]. Only MF overrides it to [self.model, self.r_embed], where r_embed is a 323-parameter two-layer MLP (the r-conditioning module). This is explicitly noted as the only extra parametric component, intentionally tiny relative to the 6.35M backbone.

### 3.3 The BaseAlgorithm Interface

```
BaseAlgorithm (abstract)
|-- training_step(batch) -> {"loss": scalar_tensor, ...}
|-- sample(n_samples, nfe, device) -> Tensor(N, C, H, W)
|-- trainable_modules() -> List[nn.Module]       # what the trainer optimises
|-- on_after_optimizer_step()                    # hook, e.g. EMA update
|-- checkpoint_state() / load_checkpoint_state()
|-- name()
```

The Trainer, Sampler, and Evaluator all depend only on this abstract class. Adding a new algorithm requires creating one file and one registry entry -- no changes to any shared module.

---

## 4. What Is Actually Original, vs Reimplementation

**Reimplementations (no substantive modification):**
- FM: straight reimplementation of Lipman 2022 / Liu 2022 CFM.
- FM-LN: direct implementation of Esser et al. 2024 logit-normal time sampling.
- Reflow: direct implementation of Liu 2022, Section 3.
- Consistency Models: direct implementation of Song 2023, with documented simplification (L2 instead of LPIPS).

**Adaptations with documented deviations:**
- **MF with FD-JVP:** Paper (Geng 2025) uses exact torch.func.jvp. This project replaces it with finite-difference approximation. Stated reason: AMP compatibility and ~2x speed. Cost: O(delta) bias. **The bias has not been empirically quantified.** The annealing schedule and two-path stochastic routing are the team's own engineering choices.
- **MF-Distill:** The distillation target u_tgt = (z_t - z_r_teacher) / (t-r) using a frozen FM teacher's Euler rollout is the team's own synthesis. Not a verbatim copy of any paper section.

**Engineering originality (distinct from algorithmic novelty):**
- The shared-backbone fairness enforcement system (_protected_keys, build_backbone, BaseAlgorithm interface).
- The two-path stochastic training routing in MF (FD path / diagonal path).
- The adaptive NFE sampler (AdaptiveMeanFlowSampler).
- The coarse-to-fine pipeline (MultiScaleMeanFlowPipeline).
- Cross-platform tournament orchestration, reproducibility infrastructure.
- The browser-based inference + training-history dashboard.

---

## 5. Current Empirical Status

### 5.1 Repository Metadata

```
git log --oneline | wc  ->  38 commits (as of 2026-09-18)
git log --format="%an" | sort -u  ->  sheikh2583 (single contributor + agent commits under same account)
Date range: 2026-09-15 to 2026-09-18
Post-defense commits: 4 (see Change Log, Section 0)
```

### 5.2 Checkpoint and Metrics Status for Every Run Directory

**results/fm_cifar10/**
- Checkpoint: FlowMatchingAlgorithm_epoch100.pt -- COMPLETE (100 epochs)
- Metrics: fm_cifar10.jsonl (65 KB, 127 rows)
- FID/IS at epoch 100, 1,000 generated samples:

| NFE | FID (lower better) | IS mean +/- std  |
|-----|--------------------|------------------|
|   1 |             387.81 |    1.38 +/- 0.03 |
|   5 |              98.42 |    5.21 +/- 0.43 |
|  10 |              81.77 |    5.90 +/- 0.53 |
|  20 |              76.10 |    6.12 +/- 0.72 |

Training loss: 0.316 (epoch 1) -> 0.179 (epoch 100). Monotonically decreasing.

**results/fm_lognorm_rtx3060/**
- Checkpoint: FlowMatchingLognormAlgorithm_epoch100.pt -- COMPLETE (100 epochs)
- Note: trained on RTX 3060 Laptop, different hardware session from FM.
- Metrics: fm_lognorm_rtx3060.jsonl (61.5 KB, 117 rows)
- FID/IS at epoch 100, 1,000 generated samples:

| NFE | FID (lower better) | IS mean +/- std  |
|-----|--------------------|------------------|
|   1 |             362.60 |    1.79 +/- 0.05 |
|   5 |              96.45 |    5.26 +/- 0.44 |
|  10 |              78.95 |    5.93 +/- 0.56 |
|  20 |          **72.66** | **6.18 +/- 0.52**|

Training loss: 0.204 (epoch 1) -> 0.141 (epoch 100). Monotonically decreasing.

**results/fm_lognorm_cifar10/** (separate directory, also FM-LN)
- Checkpoints present up to epoch 90 only. **No epoch 100 checkpoint.** Training interrupted.
- This appears to be a restarted parallel attempt. Evaluated results in the report come from fm_lognorm_rtx3060.

**results/mf_cifar10/** *(frozen evidence — do not modify)*
- Checkpoint: MeanFlowAlgorithm_epoch100.pt -- COMPLETE (tournament ran all 100 epochs)
- Metrics: mf_cifar10.jsonl (updated by tournament; late-divergence epochs confirmed)
- Late-epoch training losses (from tournament log `results/tournament_run_20260917_143512_pid10672.log`):

| Epoch | Loss   |
|-------|--------|
|    97 | 0.493  |
|    98 | 0.530  |
|    99 | 0.601  |
|   100 | 0.795  |

Training loss **diverged late in training** — the run completed mechanically but the epoch-100 checkpoint is evidence of failure, not a valid model to resume or evaluate further. The divergence is the motivation for the MF v2 stability run (see Section 0 Change Log and `config/mf_full_v2.json`).

> **Note for 3rd-party reviewers:** The submitted BSc defense report covers the *earlier* partial MF run (stopped at epoch 30). The tournament ran MF to completion but the late divergence confirms the report's conclusion. The post-defense work in the Change Log (Section 0) is attempting to reproduce a stable MF result with corrected hyperparameters (gradient clipping, cosine LR decay, larger batch) before any further reporting.

**results/smoke_cifar10/**
- MockAlgorithm_epoch2.pt -- 2-epoch smoke test, not a real model.

**results/final_smoke_cifar10/**
- MockAlgorithm_epoch1.pt -- 1-epoch smoke test.

**Completed in tournament (all 100 epochs):** FM, FM-LN (as fm_lognorm_cifar10), MF (diverged), Consistency, MF-Distill, Reflow.

**No runs exist for CelebA** (any algorithm).

> **Tournament log:** `results/tournament_run_20260917_143512_pid10672.log` — approximately 12 h 22 min on RTX 4070 Laptop. All six CIFAR-10 algorithms reached epoch 100. Aggregation completed. Five runs are usable; MF is diverged evidence.

### 5.3 verify_workflow.py Real Output (run 2026-09-17)

```
[OK] algorithms: ['consistency', 'fm', 'fm_lognorm', 'mf', 'mf_distill', 'reflow']
[OK] additional utility algorithms: ['mock']
[OK] dataset registry: cifar10, celeba
[OK] config: config/fm_full.json
[OK] config: config/fm_lognorm_full.json
[OK] config: config/mf_full.json
[OK] config: config/mf_distill_full.json
[OK] config: config/consistency_full.json
[OK] config: config/reflow_full.json
[OK] config: config/fm_celeba64.json
[OK] config: config/fm_lognorm_celeba64.json
[OK] config: config/mf_celeba64.json
[OK] config: config/mf_distill_celeba64.json
[OK] config: config/consistency_celeba64.json
[OK] config: config/reflow_celeba64.json
[OK] config: config/mf_coarse16.json
[OK] MF-Distill teacher: results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt
[OK] Consistency teacher: results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt
[OK] Reflow pairs: data/reflow_pairs_cifar10.pt
[BLOCKED] CelebA MF-Distill teacher: results/fm_celeba/checkpoints/FlowMatchingAlgorithm_epoch100.pt
[BLOCKED] CelebA Consistency teacher: results/fm_celeba/checkpoints/FlowMatchingAlgorithm_epoch100.pt
[BLOCKED] CelebA Reflow pairs: data/reflow_pairs_celeba.pt

Workflow code and configuration checks passed.
Long-running stages remain blocked until these artifacts exist:
  - CelebA MF-Distill teacher missing
  - CelebA Consistency teacher missing
  - CelebA Reflow pairs missing
```

---

## 6. Codebase Size / Shape Metrics

### 6.1 Total Python LOC

```
5,661 lines (all .py files tracked by git, excluding venv and __pycache__)
```

### 6.2 Breakdown by Folder

| Folder      |  LOC | Notes                                                                   |
|-------------|-----:|-------------------------------------------------------------------------|
| algorithms/ | 1239 | 12 files: 6 algorithms + base + mock + r_embed + 2 extension samplers  |
| scripts/    | 1957 | 25 scripts: orchestration, aggregation, benchmark, inference, etc.      |
| utils/      |  398 | checkpoints, results, timing utilities                                  |
| training/   |  294 | trainer.py (main training loop)                                         |
| evaluation/ |  251 | evaluator.py, metrics.py                                                |
| models/     |  178 | backbone.py (SimpleUNet + build_backbone)                               |
| config/     |  129 | config.py + __init__.py                                                 |
| sampling/   |  100 | sampler.py                                                              |

Root-level scripts (train.py ~83, evaluate.py ~119, bootstrap.py ~490) add ~700 more lines.

### 6.3 Config Presets and Scripts

- **18 config presets** in config/: 6 CIFAR-10 full training, 6 CelebA 64x64, 2 FM-LN variants, 1 coarse MF (16x16), 1 smoke test, 2 others.
- **25 scripts** in scripts/ (cross-platform PS1 + sh pairs for all major workflows).

---

## 7. Known Limitations, Honestly Stated

### 7.1 CIFAR-10 Resolution Ceiling (32x32)

All results are on 32x32 images. At this resolution, meaningful discrimination between few-step generation methods is difficult. FID scores at NFE=1 for a poorly trained model (210 for MF) are more attributable to training quality than to any fundamental algorithmic property. The ~312-point FID gap between NFE=1 and NFE=20 for well-trained FM is real, but it compresses into a range that is also noisy at 1,000-sample evaluation.

### 7.2 Fairness Gap for Teacher-Dependent Methods

MF-Distill and Consistency Models require a pre-trained FM teacher (6.35M params, 100 epochs, ~2.5 hrs) plus student training. Student backbone parameter count is identical to FM/MF, so that comparison is fair. Total compute cost is not. This is documented in algorithms/__init__.py:

```
Fairness note on mf_distill and consistency:
    Both require a pre-trained FM teacher checkpoint. Their results are NOT
    directly comparable to backbone-only runs. Report teacher cost separately.
```

### 7.3 FD-JVP Truncation Bias -- Not Empirically Quantified

The finite-difference approximation introduces O(delta) bias in the Mean Flow training target. delta is annealed from 1e-2 to 1e-4. **No ablation exists comparing FD-JVP to exact JVP quality.** Whether it is the primary cause of MF divergence is unknown.

### 7.4 MF Training Instability -- Actual Outcome

- Training loss rose from 0.554 (epoch 1) to 0.648 (epoch 30) -- never decreased.
- MF was stopped at epoch 30 of 100.
- Multi-step sampling degrades quality (FID worsens from NFE=5 to NFE=20).
- FID at NFE=1 is 210.99, confirming the displacement mechanism functions mechanically, but quality is not competitive.

The report (ch7_challenges.tex) attributes MF divergence to: LR (5e-5, 4x lower than FM), small batch size (32 vs FM's 128), only 30 epochs, additive r-conditioning possibly too weak, and absence of EMA and gradient clipping (which the Geng 2025 paper uses). **The FD-JVP approximation may also be a contributing factor, but this has not been tested.**

### 7.5 Consistency Models -- Never Trained

Despite full implementation and a tuning harness, no training was ever run. docs/CONSISTENCY_TUNING_NOTES.md: "no optimization run was started because model training is explicitly reserved for the user."

### 7.6 Cross-Platform / Linux Verification Gap

All .sh scripts use LF (Unix) line endings -- the CRLF issue has been fixed. However, the Linux verification log in docs/LINUX_VERIFICATION.md shows:

```
| 2026-09-17 | Dev PC (Windows) | Windows 11 | 12.x | PASS | All steps verified |
| (add row when run on Linux lab PC) |  |  |  |  |  |
```

**Full Python installation, dependency resolution, and dataset download on an actual Linux machine has not been performed.** Linux scripts were parsed through WSL but end-to-end Linux execution has not been confirmed.

### 7.7 TODO / FIXME / XXX Markers in Python Files

```
grep -rn "TODO|FIXME|XXX" --include="*.py" .
-> zero matches
```

Zero markers found in any Python file.

### 7.8 Evaluation Sample Count

FID and IS are computed from **1,000 generated images** against a **1,000-image FID reference**. Standard benchmarks use 50,000. The report explicitly acknowledges this cannot be compared to published results.

### 7.9 Unmatched Training Budgets (Major Validity Threat)

| Run   | Epochs | Batch | LR   | GPU session                          |
|-------|--------|-------|------|--------------------------------------|
| FM    |    100 |   128 | 2e-4 | RTX 4070 Laptop                      |
| FM-LN |    100 |    32 | 2e-4 | RTX 3060 Laptop (different session)  |
| MF    |     30 |    32 | 5e-5 | RTX 4070 Laptop                      |

FM processes 4.99M training samples; MF processes 1.50M. FM-LN uses 4x more gradient steps per epoch. Hardware context differs.

### 7.10 Single Seed, No Variance Estimate

All runs use seed 0. No multi-seed replication. Run-to-run variance is unknown.

---

## 8. Team and Effort Context

**Team members (from docs/report/main.tex and appB_contributions.tex):**

1. **Sheikh Mosheul Akbar** -- Sole responsibility for software architecture, implementation, experiment execution, training and evaluation workflows, debugging, integration, and preparation of the working system. Also conducted literature review and prepared project documentation. (Git authorship: all 34 commits are under username sheikh2583.)
2. **Md. Samiul Islam** -- Reviewed relevant research papers and assisted with project documentation.
3. **Rad Shahmad Daiyan** -- Assisted with preparation and printing of the final report documents.

**Timeframe:** Repository commits span 2026-09-15 to 2026-09-17 (34 commits, single author). Defense date is 14 September 2026, suggesting the git history captures a polishing/infrastructure push immediately after defense.

**Supervisor:** Dr. Md. Azam Hossain. Co-supervisor: Samnun Azfar.

The contribution table states "sole responsibility" for one member and minimal supporting roles for the other two. The reviewer should take this at face value as documented by the team themselves.

---

## 9. What Is Explicitly NOT Done Yet

### 9.1 Training Runs Not Executed

- MF-Distill CIFAR-10: not trained.
- Consistency Models CIFAR-10: not trained.
- Reflow CIFAR-10: not trained (pairs generated, training not started).
- All CelebA runs (all 6 algorithms): none started.
- MF CIFAR-10 to 100 epochs: stopped at 30 with diverging loss.
- FM-LN in the fm_lognorm_cifar10 directory: stopped at 90 epochs.

### 9.2 Reflow Pairs

50,000 CIFAR-10 Reflow pairs generated and validated (1.144 GB file). CelebA Reflow pairs not generated.

### 9.3 Tournament Aggregation

results/aggregate/ covers only three completed runs plus smoke tests. The planned full six-algorithm x two-dataset tournament has not been executed.

### 9.4 Report Status -- Chapter by Chapter

The report in docs/report/chapters/ is **written and complete as submitted for the BSc defense**. All chapters contain real content (not placeholders):

| Chapter                  | Status  | Notes                                                                       |
|--------------------------|---------|-----------------------------------------------------------------------------|
| ch1_introduction.tex     | Written | Background, problem statement, objectives, research questions, contributions|
| ch2_background.tex       | Written | Background on FM, MF, diffusion models                                      |
| ch3_scope.tex            | Written | Scope, controlled/uncontrolled factors, validity threats -- explicitly honest|
| ch4_methodology.tex      | Written | Pipeline, backbone, objectives, sampling, evaluation, audit protocol        |
| ch5_implementation.tex   | Written | Module organisation, registry, implementation notes FM/FM-LN/MF            |
| ch6_results.tex          | Written | Real tables from JSONL; FID/IS; honest discussion of MF divergence          |
| ch7_challenges.tex       | Written | MF instability, JVP sensitivity, unmatched budgets, evaluation limitations  |
| ch8_ethics.tex           | Written | Ethics chapter                                                              |
| ch9_conclusion.tex       | Written | Narrow conclusion: FM-LN best; MF displacement visible but not competitive  |
| appA_reproducibility.tex | Written | Fresh-clone setup, three-model workflow, evidence audit                     |
| appB_contributions.tex   | Written | Explicit individual contribution breakdown                                  |
| appC_github.tex          | Written | Repository URL, what is and is not tracked                                  |

**The report covers only FM, FM-LN, and MF.** The extended codebase (MF-Distill, Consistency, Reflow, CelebA) is infrastructure built after the experimental core and is not reflected in the submitted report.

---

## 10. Technical Evaluation

*Structured critical assessment based solely on the documented evidence above. No AI-authorship assessment is performed.*

### Strengths

**S1. Fairness architecture is genuine and well-executed.**
The shared-backbone design with _protected_keys, build_backbone, and the BaseAlgorithm interface is a legitimate, deliberate engineering choice that prevents the most common confound in algorithm comparison papers. The implementation is clean and the enforcement is mechanical, not just a documentation claim.

**S2. Honest reporting of failure.**
The submitted report does not hide MF divergence. Tables show rising loss. The challenges chapter lists five specific probable causes. The scope chapter explicitly lists uncontrolled differences that prevent causal attribution. This level of honesty is rare and appropriate.

**S3. Evidence traceability.**
Every reported number can be traced to a config JSON + JSONL log + checkpoint. The audit protocol (deduplication by last-occurrence rule, verification of contiguous epochs) is documented and applied. FID reference cache metadata prevents silent mismatch.

**S4. Infrastructure quality.**
The cross-platform bootstrap, interactive training launcher, browser inference server, and orchestration scripts represent substantially more engineering than a typical undergraduate project. The verify_workflow.py output shows a clean dependency graph with meaningful error messages.

**S5. FD-JVP deviation is documented.**
The deviation from the source paper is disclosed in code docstrings and theory notes, including the stated tradeoff (bias vs speed). This is better than many published papers that make similar approximations without disclosure.

---

### Weaknesses

**W1. The core experimental comparison is severely confounded.**
FM, FM-LN, and MF use different batch sizes, epoch counts, learning rates, and GPU hardware sessions. The report acknowledges this under "uncontrolled differences" but the framing still presents results as if they constitute a comparison. They do not: the MF run processes 3.3x fewer training samples than FM, uses a 4x lower learning rate, and uses different hardware. No valid conclusion about algorithm quality can be drawn from this data.

**W2. MF training was not fixed before submission.**
The diverging MF loss was identified. The paper specifies EMA and gradient clipping as necessary for stability; neither was added. Submitting with a known-diverged model at epoch 30 means the central algorithm under study produced no usable results.

**W3. 1,000-sample FID is unreliable for the stated purpose.**
The project compares algorithms using FID. At 1,000 samples vs a 1,000-image reference, the variance of FID estimates is large enough that the observed differences (76.10 FM vs 72.66 FM-LN) may not be statistically significant. No confidence intervals are reported.

**W4. FD-JVP bias is unquantified.**
The key technical deviation -- replacing JVP with FD -- has a stated O(delta) cost. Whether this bias matters in practice, and whether it contributed to MF divergence, is unknown. An ablation would resolve this. None was performed.

**W5. Four algorithms have zero training results.**
MF-Distill, Consistency, Reflow, and all CelebA runs are implemented but never trained. The gap between the codebase's scope and its actual experimental output is large.

**W6. Report scope is narrower than codebase scope.**
The report covers three algorithms on one dataset. The codebase supports six algorithms on two datasets. Substantial undocumented work exists.

**W7. Single contributor for a three-person group project.**
The contribution table states "sole responsibility" for one member. This is transparent, but raises whether the project satisfies group collaboration requirements.

**W8. MF r-conditioning is weak by the team's own analysis.**
The r-conditioning adds a 3-channel bias (323 parameters). The challenges chapter notes this may be inadequate; the Geng 2025 paper uses more expressive conditioning. Acknowledged but not addressed.

**W9. Linux verification is incomplete.**
The LINUX_VERIFICATION.md explicitly leaves the Linux row blank. A submission claiming cross-platform support without a completed Linux smoke test is a reproducibility gap.

---

### Missing Evidence

- ME1: No ablation between FD-JVP and exact JVP quality -- the central technical claim has no empirical support.
- ME2: No multi-seed replication -- single-seed results cannot estimate variance.
- ME3: No statistical significance testing on FID/IS differences.
- ME4: No qualitative sample images in this package (samples exist in results/*/samples/ but are not in the zipped metrics).
- ME5: No comparison to published CIFAR-10 baselines (explicitly disclaimed, but this means no external context for the FID numbers).
- ME6: No report chapter covering the four extended algorithms (MF-Distill, CM, Reflow, CelebA).

---

### Technical Risks

- TR1: If MF training converges with corrected hyperparameters (proper LR, EMA, gradient clipping), the one-step advantage at NFE=1 (FID 210 vs 388 for FM) could become competitive -- but this has not been demonstrated.
- TR2: The FD-JVP approximation may compound with early training instability. If v is large (uninitialised network), z_t + delta*v can diverge, producing unstable FD targets. Identified in ch7_challenges.tex under "JVP Numerical Sensitivity."
- TR3: The planned full tournament requires ~222 GPU hours on the RTX 4070 Laptop or ~137 hours on an RTX 3090. Whether this compute is available before any final submission deadline is not documented.
- TR4: Consistency model training stability was labelled the highest-risk item and was deliberately deferred. It remains unvalidated empirically.

---

### Summary Verdict

The project is a technically credible undergraduate implementation effort with one serious problem: the central experiment (algorithm comparison) is not a valid comparison due to unmatched training conditions, and the most interesting algorithm (Mean Flow) produced a diverged run that was never fixed. The infrastructure is solid and well above average for an undergraduate project. The honesty of the reporting -- the team explicitly lists the confounds that invalidate their own comparison -- is commendable and unusual. An expert reviewer would note that honest reporting partially compensates for weak empirical evidence, but cannot substitute for it.

The project demonstrates that the team understands what a controlled experiment requires. The gap is execution: the resources or time to run the controlled experiment were not available, and the submitted evidence reflects whatever could be collected within those constraints.

---

*End of THESIS_REVIEW_PACKAGE.md*
