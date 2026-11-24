# Publication feasibility

Status date: 2026-11-25. Owner: research lead. Companion files:
[PROTOCOL_V2.md](PROTOCOL_V2.md) (what "standardized" means) and
[ROADMAP.md](ROADMAP.md) (phases, gates, checkboxes).

## Bottom line

| Question | Answer |
|---|---|
| Can the current results be published? | **No.** The comparisons have confounds that reviewers will find (see "Why the current results are not publishable"). |
| Can this project produce a real conference paper? | **Yes: two papers with different contributions are feasible on the existing single RTX 3090.** |
| Paper 1: international (IJCNN / ICIP / ICONIP class) | **Conditional.** It depends on a one-week go/no-go test of the Hutchinson-CV estimator (Gate G0). |
| Paper 2: Bangladeshi IEEE conference (IEEE Xplore) | **High probability** once Protocol v2 is applied and 3 seeds are run. |
| Top-tier venue (CVPR, NeurIPS, ICLR) | **Not feasible.** These venues expect ImageNet-scale experiments, a DiT backbone and near-state-of-the-art FID. |
| Compute needed | Paper 1 ≈ 250 GPU-hours (about 10.4 days on one 3090). Paper 2 adds about 96 GPU-hours (about 4 days); FM and MF runs are shared between the two papers. |
| Engineering needed | About 20–26 working days for one person, before any writing. |
| Target | **IJCNN 2027 for Paper 1.** Work starts Monday 30 Nov 2026; internal deadline 15 Jan 2027, with 31 Jan as the outer limit. This needs a near-full-time pace: about 26 working days of the 33 available. |

## Why the current results are not publishable

1. **Hyperparameters differ per method.** Learning rate, batch size, scheduler and gradient clipping all vary. On latents, for example, FM and Reflow use LR 2e-4 while MF and MF-Distill use 5e-5.
2. **No weight EMA** on any evaluated model. EMA is standard and changes FID substantially.
3. **Mean Flow is not faithful to the paper.** It uses uniform `t` instead of logit-normal, no adaptive loss weighting, and per-batch routing to the r=t path.
4. **The time embedding is effectively unscaled.** `SinusoidalTimeEmbedding` receives `t ∈ [0,1]` and its fastest frequency is 1, so every channel is a smooth, near-linear function of `t`. Most diffusion code scales `t` by about 1000. This likely weakens time conditioning for every method, and (r, t) conditioning for MF in particular. *Untested; this needs a pilot run.*
5. **Teacher cost is excluded.** Reflow, MF-Distill and Consistency all depend on an FM teacher. Reflow also needs 50-step pair generation. MF trains from scratch.
6. **One seed, 5k-sample FID, no confidence intervals.** The few-step latent scores (Reflow 25.7, Consistency 26.7, Hutchinson-CV 29.6, MF 34.5) cannot be told apart statistically.
7. **The Hutchinson-CV vs MF comparison is confounded.**
   - The canonical latent MF run used finite differences (FD) and the legacy `REmbed` conditioner, per its training log, while `config/mf_celeba_latent.json` says exact JVP.
   - The two runs were scored at different epochs (80 vs 90) and use different r≠t ratios.
8. **The thesis report contradicts itself.** Chapter 1, the first half of Chapter 6, and Chapter 9 still describe the old 3-method, 1k-sample CIFAR-10 results.

None of these is a dead end. Each one is a standardization item in Protocol v2.

## The two papers

### Paper 1 (international): an unbiased, reverse-mode-only target estimator for Mean Flow

- **Claim.** Hutchinson-CV (an FD baseline plus a Rademacher-probe VJP correction) gives an unbiased estimate of the Mean Flow target using reverse-mode AD only. Its variance is controlled by the FD residual. It works where forward-mode JVP is unavailable or memory-heavy, such as attention kernels and sharded training, and matches exact-JVP quality at equal or lower cost.
- **Why it might be new.** A web search on 2026-11-25 found no paper using this estimator. Nearby JVP-free work does exist: SplitMeanFlow (arXiv 2507.16884), Euler Mean Flows (arXiv 2602.02571), and an FD-replacement result (arXiv 2511.13019). **Repeat this literature check right before writing.**
- **Main risk.** arXiv 2511.13019 reports that plain FD already matches exact JVP. If that holds here, the estimator has nothing to fix and Paper 1 is dead. **Gate G0 measures this directly before any long training.**
- **Evidence required:**
  - an estimator error study (bias and variance against exact JVP over a grid of (t, r) values, in fp32 and fp16);
  - training with 6 estimator settings × 3 seeds on latent CelebA (main) and CIFAR-10 (secondary);
  - a SplitMeanFlow baseline;
  - ablations (number of probes, FD step, r≠t ratio);
  - step-time and memory profiling on an attention backbone, to justify the motivation.

### Paper 2 (Bangladeshi IEEE): compute-matched few-step generation on one GPU

- **Claim.** A matched-protocol comparison of FM, FM-LN, MF, MF-Distill, Consistency and Reflow on pixel and latent data, with **total** training cost including the teacher. It answers when from-scratch one-step training beats teacher-based methods at a fixed GPU-hour budget.
- **Extra findings to include:** the pixel-vs-latent cost result, and the latent-FID evaluation pitfall as a lessons-learned section.
- **Rule:** mention Hutchinson-CV in at most one sentence, so Paper 1 stays new. No dual submission.

## Compute estimate

Measured anchors are the 3090 training hours for 100 epochs from `THESIS_SUMMARY.md`:

| Method | CIFAR-10 (6.35M params, b128) | Latent CelebA (24M params, b32) |
|---|---:|---:|
| FM / FM-LN | 0.62 h | 1.82 h |
| MF (FD) | 0.80 h | 2.82 h |
| MF (exact JVP) | 1.36 h | ~4.8 h (estimated from the CIFAR JVP/FD ratio) |
| MF-Distill | 1.56 h | 5.58 h |
| Consistency | 1.07 h | 4.32 h |
| Reflow | 0.64 h (+ pairs) | 2.69 h (+ pairs) |
| Hutchinson-CV, 4 probes | — | ~7.6 h (6.94 h measured for 91 epochs) |

**Evaluation cost.** Logged CIFAR-10 sampling takes about 3.7 s per NFE for 5k images. FID-10k over the NFE grid {1,2,4,8,16,32} is therefore about 8 minutes of sampling plus about 3 minutes of Inception features. That is roughly **0.25 h per evaluation point**, or 0.5 h per run with two points (latent runs are a little slower because of decoding).

The per-run hours below are **upper bounds**. A paper-faithful MF computes the estimator only on the 25% of samples where r≠t, which should cut MF costs noticeably.

### Paper 1

| Block | Runs | Hours |
|---|---:|---:|
| G0 estimator study (existing checkpoints) | — | 2 |
| G1 pilot to freeze the protocol (latent FM and MF-JVP, seed 0, 30 epochs) | 2 | 4 |
| Latent main: FM, JVP, FD, double-VJP, Hutch-CV-1, Hutch-CV-4 × 3 seeds | 18 | 83 |
| CIFAR-10 main: the same 6 × 3 seeds | 18 | 23 |
| SplitMeanFlow baseline (latent + CIFAR × 3 seeds) | 6 | 22 |
| Ablations (latent, seed 0) | 6 | 30 |
| Attention-backbone cost profiling (no full training) | — | 3 |
| Evaluation (2 points per run) | 48 | 24 |
| **Subtotal** | | **191** |
| **With 30% contingency** | | **≈250 h ≈ 10.4 GPU-days** |

The CIFAR-10 block (about 30 h with evaluation) fits on the RTX 4070 laptop (8 GB). Running it there shortens the 3090 queue to about 9 days.

### Paper 2 (incremental: FM and MF-JVP are reused from Paper 1)

| Block | Runs | Hours |
|---|---:|---:|
| Latent: FM-LN, MF-Distill, Consistency, Reflow (+ pairs) × 3 seeds | 12 | 49 |
| CIFAR-10: the same × 3 seeds | 12 | 13 |
| Evaluation | 24 | 12 |
| **With 30% contingency** | | **≈96 h ≈ 4 GPU-days** |

Leave CelebA 64×64 **pixel** out of both papers. FM alone took 16 h per run there, so 3 seeds × 6 methods would need about 300 h or more.

## Engineering estimate (one person)

| Work item | Days |
|---|---:|
| Protocol v2 infrastructure: EMA, time-embedding scale, config fields, seed-aware run directories, CI-aware aggregation | 3–4 |
| MF estimator algorithm (exact JVP, double-VJP, FD, Hutchinson, Hutchinson-CV) and tests | 2–3 |
| Estimator study script (G0) | 1–2 |
| SplitMeanFlow baseline | 2–3 |
| Small attention/DiT backbone for profiling | 2 |
| Aggregation, figures, tables | 2 |
| Writing (6 pages) and supervisor review | 7–10 |
| **Total** | **≈20–26 working days** |

## Calendar options

Deadlines below come from web listings and **must be re-checked on the official sites**.

**Chosen plan (decided 2026-11-25): start Monday 30 Nov 2026 and target IJCNN 2027.**

| Paper | Primary target | Fallback |
|---|---|---|
| Paper 1 (if G0 passes) | IJCNN 2027: internal deadline 15 Jan 2027 (some sources say 31 Jan). Conference 14–18 June 2027, Cape Town, 6 pages. | ICIP 2027 (Singapore, 29 Nov–3 Dec 2027; deadline not yet announced) or ICONIP 2027 |
| Paper 2 | The first Bangladeshi IEEE conference whose deadline falls after mid-February 2027 (deadlines not yet checked) | Late-2027 Bangladeshi cycle (ICCIT, STI, ICEEICT 2028) |
| If G0 fails | Paper 2 plus an estimator-choice section goes to IJCNN 2027 instead (odds ~20–30%) | Bangladeshi IEEE |

The ICEEICT 2027 and ECCE 2027 deadlines have already passed.

**Why it fits.** From 30 Nov to 15 Jan there are 33 working days for about 26 days of engineering and writing. The GPU work (≈10.4 days of 3090 time) overlaps with coding and writing. The 3090 queue runs from 14 Dec to about 3 Jan (21 days), which leaves about 10 days of slack. Allow for holidays (16 Dec and 25 Dec) and avoid scope creep. The dated plan is in [ROADMAP.md](ROADMAP.md).

## Gates

| Gate | When | Pass condition | If it fails |
|---|---|---|---|
| **G0: estimator go/no-go** | Week 1 of building | FD's target error relative to exact JVP is above 1% (fp32, best step size) or above 5% (fp16) over a meaningful part of the (t, r) grid, **and** Hutch-CV has lower bias-plus-variance than FD there. Or: exact JVP is unavailable or more than 2× more memory-hungry on the attention backbone, and Hutch-CV is cheaper than double-VJP. | Drop Paper 1 as a method paper. Its material becomes an "estimator choice" section in Paper 2. |
| **G1: protocol pilot** | After the Protocol v2 code exists | With the time-embedding fix and EMA, latent FM and MF-JVP at 30 epochs beat or match their old 100-epoch FID. | Debug before spending about 10 GPU-days on the main runs. |
| **G2: seed-0 signal** | After the latent seed-0 block | The estimator ranking is visible at seed 0, and the Hutch-CV vs JVP gap is at least comparable to the seed-to-seed spread. | Report "matches JVP at lower memory" instead of "beats JVP", or fall back to Paper 2 only. |

## Acceptance odds

These are rough judgments, not data.

| Paper / venue | If gates pass | Notes |
|---|---|---|
| Paper 1 at IJCNN / ICIP / ICONIP | ~35–50% | Hinges on a convincing motivation (attention memory, kernel support) and on beating FD, not just matching it. |
| Paper 2 at a Bangladeshi IEEE conference | ~70–85% | Rigor, 3 seeds and teacher-inclusive cost accounting are enough. |
| Paper 2 at an international mid-tier venue | ~20–30% | A benchmark with no new method is usually rejected as "limited novelty". |

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| FD already matches exact JVP, so the estimator has nothing to fix | Medium–high | G0 is designed to catch this in one week. |
| No attention in SimpleUNet, so the "JVP unavailable" motivation is weak | Certain today | Profile a small DiT. A full DiT training run is optional. |
| Concurrent publication (the field moves monthly) | Medium | Re-check arXiv at G2 and again before writing. Frame the claim narrowly. |
| Absolute FID far from published numbers | Certain | Frame as a compute-matched, small-budget study. Make relative claims with CIs. Apply the time-embedding fix and EMA. |
| Latent quality capped by the codec (VQ-f4 rFID 10.06) | Certain | Disclose it and report the codec reconstruction score next to every latent FID. |
| One developer, shared GPU | Medium | Put CIFAR runs on the 4070. Use a queue with resume. Rely on checkpoints. |

## Decisions the team must make

1. ~~Pace~~: decided. Start 30 Nov 2026, target IJCNN 2027.
2. Dedicated 3090 time: reserved from 14 Dec 2026 to about 3 Jan 2027 for Paper 1, then about 4 more days for Paper 2.
3. Author list and supervisor sign-off for both papers.
4. Whether to add the attention backbone. Recommended: profiling only.
