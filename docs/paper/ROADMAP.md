# Publication roadmap

Start: **Monday 30 Nov 2026.** Target: **IJCNN 2027**, with an internal
submission date of **15 Jan 2027** (31 Jan is the outer limit; confirm on
ijcnn.org). Rationale: [FEASIBILITY.md](FEASIBILITY.md). Specification:
[PROTOCOL_V2.md](PROTOCOL_V2.md). Tick boxes as work lands, and record dates
and run IDs next to them.

## Week-by-week schedule

| Week | Dates | Engineering / writing | 3090 GPU | 4070 laptop | Gate |
|---|---|---|---|---|---|
| 1 | 30 Nov – 6 Dec | Phase 1: `mf_targets.py`, `estimator_study.py`, tests | G0 study (≈2 h) | G0 on the CIFAR MF checkpoint | **G0 on Fri 4 Dec** |
| 2 | 7 – 13 Dec | Phase 2: EMA, time-embedding scale, LR warmup, `mf_est`, launcher, v2 configs, 50k CIFAR FID reference | idle (or the G1 pilot over the weekend) | — | — |
| 3 | 14 – 20 Dec | SplitMeanFlow baseline; DiT-S profiling code | G1 pilot Mon–Tue (≈6 h), then P1-L seed 0 (≈28 h), then seeds 1–2 | P1-C block (≈30 h) | **G1 on Tue 15 Dec; G2 on Thu 17 Dec** |
| 4 | 21 – 27 Dec | Write the introduction, related work and method | P1-L seeds 1–2 finish; SplitMeanFlow (≈22 h) | CIFAR SplitMeanFlow, then profiling | arXiv re-check |
| 5 | 28 Dec – 3 Jan | v2 aggregation, figures, tables; write the experiments section | ablations (≈30 h) and evaluations; contingency | evaluations | Results frozen on Sun 3 Jan |
| 6 | 4 – 10 Jan | Full draft to the supervisor by Fri 8 Jan | Paper 2 runs begin (≈4 GPU-days) | Paper 2 CIFAR runs | — |
| 7 | 11 – 15 Jan | Revisions; **submit Paper 1 on Fri 15 Jan** | Paper 2 runs continue | — | — |
| 8+ | after 15 Jan | Paper 2 analysis and writing toward its Bangladeshi IEEE target | — | — | — |

**Fallback.** If G0 fails on 4 Dec, Paper 1 becomes Paper 2 with an
estimator-choice section. It keeps the IJCNN date, and its Paper 2 runs move
into weeks 3–5.

**Slip rule.** If seed-0 results (G2) are not in by 20 Dec, cut the
ablations to 3 runs and drop DiT profiling. Do not drop seeds.

**Phase 0** (thesis hygiene) is not on the IJCNN critical path. Do it in
spare hours, or after 15 Jan.

## Phase 0: thesis hygiene (parallel, no GPU)

- [ ] Rewrite Chapter 1, the first half of Chapter 6, and Chapter 9 to match the 6-method 5k-sample v1 results in `THESIS_SUMMARY.md`
- [ ] Correct the Hutchinson-vs-MF interpretation in Chapter 6 (canonical latent MF used FD and legacy `REmbed`)
- [ ] Align `config/mf_celeba_latent.json` with the run that produced the canonical result, or label it as differing
- [ ] Resolve the `mf_cifar10` provenance row (`unknown@unknown`)

## Phase 1: G0 estimator go/no-go (≈3–4 days of work, ≈2 GPU-hours)

- [ ] `algorithms/mf_targets.py`: jvp, double_vjp, fd, hutch, hutch_cv; unit tests (double_vjp equals jvp; hutch and hutch_cv are unbiased in expectation)
- [ ] `scripts/estimator_study.py`: (t, r) grid, fp32/fp16, bias, variance, cosine similarity, target relative error, step time, peak memory
- [ ] Run on an existing latent MF or Hutchinson-CV checkpoint (3090) and on the CIFAR-10 MF checkpoint (4070)
- [ ] Write the result into `results/protocol_v2/estimator_study/` and record the verdict here
- [ ] **Gate G0 decision:** ☐ GO (Paper 1 + Paper 2) ☐ NO-GO (Paper 2 only)

## Phase 2: Protocol v2 infrastructure (≈4–5 days)

- [ ] `time_embed_scale`, EMA with warmup, LR warmup, config fields, tests
- [ ] `mf_est` algorithm (paper-faithful MF + estimator switch), tests
- [ ] CIFAR-10 50k-train FID reference cache
- [ ] Seed-aware launcher `scripts/protocol_v2.py` (priority queue, resume, dry run) + wrappers
- [ ] Generate `config/protocol_v2/**` and commit

## Phase 3: G1 pilot (≈1 day of work, ≈6 GPU-hours)

- [ ] Latent FM and MF-JVP, seed 0, 30 epochs, LR 1e-4 and 2e-4
- [ ] Compare with v1: does the time-embedding fix plus EMA help?
- [ ] **Gate G1:** freeze LR, epochs and the NFE grid; tag the commit `protocol-v2-frozen`

## Phase 4: main runs (≈10 GPU-days for Paper 1; ≈4 for Paper 2)

- [ ] P1-L block, seed 0 → **Gate G2** (re-check arXiv for concurrent work)
- [ ] P1-L block, seeds 1–2
- [ ] P1-C block on the RTX 4070 (in parallel)
- [ ] SplitMeanFlow baseline, ablations, DiT profiling
- [ ] P2 blocks (FM-LN, MF-Distill, Consistency, Reflow)

## Phase 5: analysis

- [ ] v2 aggregate: CI tables, FID-vs-NFE, FID-vs-total-GPU-hours (teacher included), estimator error figure
- [ ] Sample grids per method/NFE with fixed seeds

## Phase 6: writing and submission

- [ ] Paper 1 draft (6 pages, IEEE two-column) → supervisor review → submit
- [ ] Paper 2 draft → supervisor review → submit
- [ ] Confirm each venue's deadline, page limit, anonymity and dual-submission policy on the official site

## Decision log

| Date | Decision | By |
|---|---|---|
| 2026-11-25 | Pivot from the v1 thesis comparison to the two-paper plan; Protocol v2 drafted | team |
| 2026-11-25 | Pace: start 30 Nov 2026, target IJCNN 2027 (internal deadline 15 Jan) | team |
