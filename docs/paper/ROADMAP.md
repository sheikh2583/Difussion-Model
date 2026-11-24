# Publication roadmap

Track: **steady** by default (see [FEASIBILITY.md](FEASIBILITY.md)); switch to
fast (IJCNN 2027) only if Gate G0 passes by about 3 Dec 2026.
Specification: [PROTOCOL_V2.md](PROTOCOL_V2.md). Tick boxes as work lands;
record dates and run IDs next to them.

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
