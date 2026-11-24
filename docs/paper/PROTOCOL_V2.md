# Protocol v2: standardized experiment specification

Status: **specification only, not yet implemented.** v1 results (everything in
`THESIS_SUMMARY.md`) remain thesis evidence and must not be mixed into
v2 tables. Rationale: [FEASIBILITY.md](FEASIBILITY.md).

## Principles

1. **One recipe per dataset.** Every method shares the backbone, optimizer,
   LR schedule, batch size, step budget, EMA, precision and evaluation.
   Only the training objective and its inherent hyperparameters differ.
2. **Pre-registered reporting.** The reported checkpoint is the final epoch
   (EMA weights). No picking the best epoch, NFE or seed after the fact.
3. **Three seeds per cell.** Report each seed and the mean ± 95% CI.
4. **Count all the compute.** Each method's cost includes its teacher
   training and any pair generation.
5. **Freeze after Gate G1.** Any later change creates v2.1 and reruns every
   affected cell.

## Shared settings

| Setting | CIFAR-10 (pixel 3×32×32) | CelebA latent (VQ-f4, 3×16×16) |
|---|---|---|
| Backbone | SimpleUNet, base 64, mults [1,2,2], 2 res blocks (6.35M) | SimpleUNet, base 128, mults [1,2,2], 2 res blocks (24.0M) |
| Time embedding | **t scaled by 1000 before the sinusoid** (new `time_embed_scale`); applies to r/Δt conditioning too | same |
| Batch | 128 | 128 |
| Budget | 100 epochs ≈ 39k steps | 100 epochs ≈ 127k steps (G1 may adjust; the same for all methods) |
| Optimizer | AdamW, β=(0.9, 0.999), wd 0 | same |
| LR | Single value for all methods, chosen at G1 from {1e-4, 2e-4} using FM and MF-JVP | same |
| Schedule | 1k-step linear warmup (new), then constant | same |
| Gradient clip | 1.0 | 1.0 |
| EMA | decay 0.9999, warmup `min(d, (1+n)/(10+n))` (new); evaluate EMA weights | same |
| Precision | fp16 AMP with GradScaler; MF targets and estimators in fp32 | same |
| Seeds | 0, 1, 2 (seed controls init, data order, noise) | same |

## Method-inherent settings (the only allowed differences)

| Method | Settings |
|---|---|
| FM | uniform t, Euler sampling |
| FM-LN | t ~ logit-normal(0, 1), Euler sampling |
| MF (`mf_est`, new) | t and r from logit-normal(−0.4, 1.0) with r=min and t=max; r≠t ratio 0.25 (per sample, not per batch); conditioning on (t, t−r); adaptive loss weight p=1, c=1e-3; target estimator ∈ {jvp, double_vjp, fd, hutch, hutch_cv}; FD step 1e-3 (fixed); probes ∈ {1, 4} |
| MF-Distill | the MF settings above, with the target from a frozen FM teacher's 4-step Euler rollout |
| Consistency | consistency distillation from the FM teacher, EMA target 0.999, 18 discretization steps |
| Reflow | pairs from the FM teacher, 50-step Euler, same pair count on both datasets |
| SplitMeanFlow (new baseline) | interval-splitting consistency, initialized from the FM teacher, as in arXiv 2507.16884 |

Teacher-based methods use the **seed-matched** v2 FM run as the teacher.

## Evaluation

| Item | Setting |
|---|---|
| Metric | FID-10k (required), IS (secondary). FD-DINOv2 is recommended once implemented. |
| Reference | CIFAR-10: the 50k training set (the standard; v1 used the test set). CelebA latent: raw 64×64 RGB validation pixels (the corrected v1 reference). |
| NFE grid | {1, 2, 4, 8, 16, 32} for every method |
| Checkpoints scored | epoch 50 (trajectory) and epoch 100 (reported) |
| Sampler seed | equal to the training seed |
| Latent extra | report codec rFID (10.06) next to every latent FID; report backbone time and decoder time separately |
| Statistics | mean ± 95% CI with t(2)=4.30 over 3 seeds; all individual seeds in the appendix |

## Experiment matrix

Run IDs: `<paper>-<dataset>-<method>`. Each ID is 3 seeds unless marked.

| ID | Dataset | Method / variant | Paper | Priority |
|---|---|---|---|---|
| G0 | latent + CIFAR | estimator study on existing checkpoints | 1 | P0 |
| G1 | latent | FM and MF-JVP pilot, seed 0, 30 epochs, LR ∈ {1e-4, 2e-4} | both | P0 |
| P1-L-FM | latent | FM (also the teacher) | both | P1 |
| P1-L-JVP | latent | MF exact JVP | both | P1 |
| P1-L-FD | latent | MF FD | 1 | P1 |
| P1-L-DVJP | latent | MF double-VJP (exact, reverse mode only) | 1 | P1 |
| P1-L-HCV1 | latent | MF Hutchinson-CV, 1 probe | 1 | P1 |
| P1-L-HCV4 | latent | MF Hutchinson-CV, 4 probes | 1 | P1 |
| P1-L-SPLIT | latent | SplitMeanFlow | 1 | P2 |
| P1-L-ABL | latent | ablations, seed 0: probes {2, 8}, FD step {1e-2, 1e-4}, r≠t {0.5}, plain Hutchinson | 1 | P2 |
| P1-C-* | CIFAR-10 | the same 6 variants + SPLIT (can run on the RTX 4070) | 1 | P2 |
| P1-PROF | DiT-S latent | step time and memory per estimator, no full training | 1 | P2 |
| P2-L-* | latent | FM-LN, MF-Distill, Consistency, Reflow | 2 | P3 |
| P2-C-* | CIFAR-10 | FM-LN, MF-Distill, Consistency, Reflow | 2 | P3 |

## Layout and naming

```
config/protocol_v2/<dataset>/<method>.json          # seed is set by the launcher
results/protocol_v2/<dataset>/<method>/seed<k>/     # one run directory per seed
results/protocol_v2/aggregate/                      # CI tables, figures, cost table
results/protocol_v2/estimator_study/                # G0 outputs
```

Experiment name: `<method>_s<k>`. Machine label: `3090-lab` or `4070-laptop`.

## Code changes required (not yet built)

| Change | Where |
|---|---|
| `time_embed_scale` field; pass it to the sinusoid in the backbone and RCond | `config/config.py`, `models/backbone.py`, `algorithms/r_embed.py` |
| `ema_decay` and EMA warmup; save and restore EMA state; evaluate with EMA weights | new `utils/ema.py`, `training/trainer.py`, `utils/checkpoints.py`, `evaluate.py` |
| LR warmup steps | `config/config.py`, `training/trainer.py` |
| Paper-faithful MF with a switchable target estimator (`mf_est`) | new `algorithms/mf_targets.py`, `algorithms/mean_flow_est.py`, registry |
| SplitMeanFlow baseline | new `algorithms/split_mean_flow.py` |
| Estimator study (G0) | new `scripts/estimator_study.py` |
| Seed-aware launcher with a priority queue and resume | new `scripts/protocol_v2.py` + Linux/Windows wrappers |
| Aggregation with CIs and teacher-inclusive cost | extend `scripts/aggregate_results.py` (separate v2 output) |
| CIFAR-10 50k-train FID reference | `evaluation/evaluator.py` (new cache name, so v1 caches stay valid) |
| Optional DiT-S backbone for profiling | `models/` |

Legacy algorithms (`mf`, `mf_hutchinson`, `mf_distill`) stay unchanged, so v1
checkpoints keep loading.
