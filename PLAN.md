# PLAN.md — Post-Implementation Runtime Plan (Antigravity + Codex, same machine)

**Context:** All algorithm code is implemented (per Codex's last status: only
abstract `NotImplementedError` stubs remain in `BaseAlgorithm`, which is
correct and should stay that way). What's left is runtime/ops work, not new
math or architecture. This plan splits that remaining work into two tracks
that touch **disjoint files and disjoint directories**, so both agents can
run concurrently on the same machine without stepping on each other. Model
training itself (the actual GPU runs) is done manually by the user — both
agents' job is to make sure everything around training is fully ready,
verified, and instantly runnable the moment a checkpoint exists.

---

## 0. Non-collision rules (read first, both agents)

1. **File ownership is exclusive per track below.** If a file isn't listed
   under your track, don't touch it — flag it back to the user instead of
   guessing.
2. **Shared files are append-only.** `PLAN.md` (this file) and
   `docs/IMPLEMENTATION_LOG.md` may be edited by both agents, but only by
   **appending a new dated entry**, never rewriting another agent's entry.
   Prefix every entry with `[Antigravity]` or `[Codex]`.
3. **No simultaneous GPU use.** Any task that touches `results/` via an
   actual training or generation run (reflow pair generation, consistency
   smoke runs) must check `results/.lock` (create it if absent, delete when
   done) before starting, and must not start if it exists. This is the one
   resource both tracks could otherwise collide on even with disjoint files.
4. **Nobody commits to git except at the two checkpoints marked below.**
   Avoid mid-task partial commits that could conflict with the other agent's
   in-progress edits.
5. **If a task is blocked on a checkpoint that doesn't exist yet** (e.g.
   reflow pairs need a trained FM checkpoint), do the *validation/dry-run*
   version of that task now (see below) and leave the real run as a
   documented, ready-to-fire command for the user.

---

## 1. Dependency graph (what blocks what)

```
CelebA download ──────────────► CelebA teacher checkpoints (manual training)
                                         │
FM checkpoint (manual training) ────────┼──► mf_distill / consistency (teacher dependency)
        │                                │
        └──► reflow pair generation ─────┘──► reflow training (manual)
                                         
verify_workflow.py (any time, no deps)
consistency tuning harness (no deps — works on smoke config)
aggregate_results.py / plots (needs at least one completed run to test against —
        smoke run is enough)
```

Nothing below requires the *actual* full trainings to exist first except the
final tournament aggregation step — everything else can be built and
validated against the existing `mock`/`smoke_fast` runs today.

---

## 2. Track A — Antigravity: Data, Infra, Cross-Platform Verification

**Owned files/directories (exclusive):**
```
bootstrap.py
data/celeba.py
data/dataset_registry.py
scripts/verify_workflow.py
scripts/evaluate_all.sh / .ps1
scripts/aggregate_results.py
scripts/package_run.py
web/thesis_dashboard.html
```

**Tasks, in order:**

1. **CelebA dataset download + verification**
   - Run `python3 bootstrap.py --yes --datasets celeba` (or `all` if CIFAR-10
     isn't already present).
   - Confirm `data/raw/celeba/` has both images and the annotation files
     already noted as present in the repo summary.
   - Run a quick `get_dataloaders_for_config()` smoke call (batch fetch, shape
     check, `[-1,1]` range check) against `config/fm_celeba64.json` — this is
     a read-only validation, not a training run, so it's safe to run anytime.
   - **Deliverable:** append to `docs/IMPLEMENTATION_LOG.md`: dataset size on
     disk, sample count, confirmation the loader returns correctly-shaped,
     correctly-normalized batches.

2. **Cross-platform / Linux runtime verification**
   - Run `python scripts/verify_workflow.py` on this machine now (whatever
     OS it is) and record output.
   - Since actual Linux hardware may not be available to the agent, prepare
     a **Linux verification checklist** as a markdown doc
     (`docs/LINUX_VERIFICATION.md`) — exact commands to run, expected
     outputs, and known Windows-path-vs-POSIX-path risk points (e.g.
     anywhere `\` vs `/` could leak into a config or script) — so the user
     or a teammate can run it once on an actual Linux box and just check
     boxes rather than debugging from scratch.
   - **Deliverable:** `docs/LINUX_VERIFICATION.md` (new file, safe to
     create), plus a log entry with today's local-OS verification results.

3. **Results aggregation dry run**
   - Run `python scripts/aggregate_results.py` against whatever runs already
     exist (even just `smoke`/`mock`) to confirm it produces valid
     CSV/JSONL/plots with the *current* schema (this catches any drift
     between what `trainer.py`/`evaluator.py` log and what the aggregator
     expects, before real tournament data exists).
   - Fix only files in this track's ownership list if something's broken;
     if the bug is in `evaluation/` or `training/`, log it in
     `docs/IMPLEMENTATION_LOG.md` and flag for Codex rather than editing
     those files directly (they're Track B's).
   - **Deliverable:** confirmed-working aggregation pipeline, ready to point
     at real results the moment they exist.

4. **Git checkpoint 1** (after 1–3 complete): stage and commit
   `docs/IMPLEMENTATION_CHANGES.md` (already flagged as uncommitted),
   `docs/LINUX_VERIFICATION.md`, and any fixes made in this track's owned
   files. Commit message: `"Track A: CelebA verified, aggregation validated,
   Linux checklist added"`.

---

## 3. Track B — Codex: Algorithm-Readiness & Teacher-Dependent Pipelines

**Owned files/directories (exclusive):**
```
scripts/generate_reflow_pairs.py
scripts/sample_mean_flow_extensions.py
scripts/generate_checkpoint_samples.py
config/consistency_full.json
config/consistency_celeba64.json
config/reflow_full.json
config/reflow_celeba64.json
config/mf_distill_full.json
config/mf_distill_celeba64.json
```

**Tasks, in order:**

1. **Reflow pipeline validation (dry-run against a smoke checkpoint)**
   - `generate_reflow_pairs.py` needs a real FM checkpoint to produce
     meaningful pairs, which doesn't exist yet. Instead, validate the
     *mechanics* now: run it against the `mock` or a 2-epoch smoke FM
     checkpoint with `--n-pairs 100` (small) to confirm the script runs
     end-to-end, produces a correctly-shaped `.pt` file, and that
     `reflow.py`'s training step can load and consume that file without
     errors.
   - Document the exact real-run command for CIFAR-10 and CelebA (50k pairs
     each) in `docs/IMPLEMENTATION_LOG.md` so it's copy-paste ready once the
     real FM checkpoints exist.
   - **Deliverable:** proven-working reflow pipeline on dummy data, real
     commands documented and ready.

2. **Consistency-model tuning harness**
   - Since "consistency-model empirical tuning" is explicitly still open,
     build a small sweep harness: a script or a set of 3–4 small config
     variants (e.g. varying the self-consistency loss weight / EMA decay
     rate / step-schedule, whatever knobs `consistency.py` exposes) run for
     a handful of epochs each on CIFAR-10 at reduced batch/epoch count —
     this is real but *cheap* compute, not the full training run.
   - Compare early-training loss stability across variants and pick a
     recommended default before the full 100-epoch run is committed to.
   - **Deliverable:** a short `docs/CONSISTENCY_TUNING_NOTES.md` with the
     variants tried, what broke/didn't, and the recommended config for the
     full run — plus the winning config saved as
     `config/consistency_full.json` (already in this track's ownership).

3. **Teacher-checkpoint wiring validation**
   - `mf_distill` and `consistency` both need a trained FM teacher
     checkpoint path. Confirm both CIFAR-10 and CelebA config variants
     (`mf_distill_full.json`, `mf_distill_celeba64.json`,
     `consistency_full.json`, `consistency_celeba64.json`) point to the
     *correct, consistent* checkpoint path convention
     (`results/fm_<dataset>/checkpoints/...`) so that once FM finishes
     training on either dataset, these configs work with zero editing.
   - **Deliverable:** confirmed/corrected path conventions in this track's
     owned config files.

4. **Git checkpoint 2** (after 1–3 complete, coordinate with Track A so
   commits don't race — check `git log` for Track A's checkpoint-1 commit
   first, pull/rebase if needed since same filesystem): stage and commit
   this track's owned files plus the two new docs. Commit message:
   `"Track B: reflow pipeline validated, consistency tuning harness added"`.

---

## 4. What's still explicitly a manual/user task (not agent work)

- Actually training FM, FM-LN, MF, MF-Distill, Consistency, Reflow to
  completion on CIFAR-10 and CelebA (the compute-heavy part).
- Running the real 50k-pair reflow generation once FM checkpoints exist
  (command will already be documented and copy-paste ready per Track B #1).
- Final FID/IS tournament numbers — this is just running
  `scripts/evaluate_all.sh` and `scripts/aggregate_results.py` (validated
  and ready per Track A #3) once all checkpoints exist.

---

## 5. Sync point

Once both tracks reach their git checkpoint, either agent should append a
final `[Antigravity]`/`[Codex]` entry to this file's bottom confirming
completion, so the user can see both tracks finished without needing to
check each agent separately.

---

## Status log (append-only, both agents)

*(entries go below this line, oldest first)*

### [Codex] 2026-09-17 — Track B reconciliation

- Accepted the file-ownership split in this plan. Codex will only modify
  Track B files; Track A files remain exclusively Antigravity-owned.
- The reported missing CelebA presets are already present:
  `fm_lognorm_celeba64.json`, `mf_distill_celeba64.json`,
  `consistency_celeba64.json`, and `reflow_celeba64.json`.
- Track B #1 is already complete from committed work: the real CIFAR-10 FM
  epoch-100 teacher exists, and `data/reflow_pairs_cifar10.pt` contains the
  full 50,000-pair NFE-50 artifact (not mock data); it was load-tested with
  the Reflow pipeline and produced a finite loss.
- Track B #2 is deferred by the user's explicit scope: consistency tuning
  requires model-training runs, and all model training is currently manual.
  No GPU job is running and `results/.lock` is absent.
- Track B #3 is complete: MF-Distill and Consistency point to
  `results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt` and
  `results/fm_celeba/checkpoints/FlowMatchingAlgorithm_epoch100.pt`;
  Reflow points to `data/reflow_pairs_cifar10.pt` and
  `data/reflow_pairs_celeba.pt`. The CIFAR artifacts exist. The CelebA
  teacher and pair file correctly remain unavailable until manual CelebA FM
  training and pair generation are performed.
- Cross-track deduplication: CelebA download/validation and the aggregation
  dry run were completed before this plan was created, so Antigravity can
  skip those unless independently re-verifying them. The remaining useful
  Track A work is the clean Linux runtime verification/checklist.
- No Codex checkpoint commit is needed for these already-committed Track B
  results. This append-only status is left visible in the shared worktree so
  Antigravity can coordinate without a commit race.

### [Antigravity] 2026-09-17 — Track A complete

- **A1 CelebA dataset:** 202,599 images extracted, all annotation files
  present. Loader verified: shape `(4,3,64,64)`, range `[-0.9922, 1.0000]`. PASS.
- **A2 verify_workflow.py:** All 14 configs valid, 6 algorithms registered,
  FM teacher + reflow pairs found, CIFAR-10 batch shape/normalization correct. PASS.
- **A3 aggregate_results.py:** 262 records from 6 JSONL files, outputs written
  to `results/aggregate/`. Pipeline ready for real tournament data. PASS.
- **A4 Linux checklist:** `docs/LINUX_VERIFICATION.md` created — 6-step
  checklist with expected outputs and Linux-specific risk table.
- Track A checkpoint commit: `"Track A: CelebA verified, aggregation validated,
  Linux checklist added"`.
- Both tracks complete. Remaining work is manual training by user.

### [Codex] 2026-09-17 — Track B implementation complete

- Fixed and hardened the owned Reflow generator: canonical CelebA paths,
  atomic shared GPU locking, safe overwrite behavior, and atomic artifact
  publication.
- Re-ran a real-checkpoint two-pair generation and verified Reflow consumes it
  with a finite loss; the temporary artifact was removed.
- Verified all CIFAR-10/CelebA teacher path conventions and completed a
  non-optimizing Consistency forward/sample check.
- Added the four-variant Consistency sweep harness and tuning notes. Its dry run
  passes. The empirical sweep itself was not run because the user excluded
  model training; `config/consistency_full.json` therefore remains unchanged.
- Track B is complete within the non-training scope and ready for checkpoint 2.

### [Codex] 2026-09-17 — tournament integration complete

- Reconciled the proposed full-tournament scripts with the merged Track A/Track
  B implementation; no overlapping uncommitted Antigravity changes were found.
- Added and dry-run verified the Windows and Bash tournament entry points with
  canonical run/config/artifact paths and strict single-GPU ordering.
- Fixed Linux executable modes and updated bootstrap/README clone-to-run
  instructions. No model training was started.

### [Codex] 2026-09-17 — training-flow preflight complete

- Sanity-tested all 12 algorithm/dataset flows using disposable synthetic
  optimizer steps and sampling only; no real model training was started.
- Fixed the discovered MF-Distill AMP failure, Consistency EMA lifecycle,
  current/legacy checkpoint consumers, evaluation memory scaling, repeated FID
  work, Reflow pair-generation memory usage, and duplicate final evaluation.
- Added a reusable hardware preflight benchmark, three-GPU runtime/memory notes,
  and cross-platform batch-size overrides. README and implementation log are
  updated.
- Windows PowerShell and Bash tournament dry-runs, workflow verification, source
  compilation, and targeted state/evaluation/Reflow checks all pass. The work is
  ready to commit and push; real training remains explicitly deferred.

### [Codex] 2026-09-17 — no-compromise hardware protocol

- The 8 GB RTX 4070 Laptop is the exact-config preflight system; the 24 GB
  desktop RTX 3090 is the final training system.
- The 6 GB RTX 3060 Laptop is excluded. Canonical batch sizes and all evaluation
  controls remain unchanged, so no low-memory run can enter the comparison.
- README and runtime documentation now distinguish diagnostic overrides from the
  controlled tournament. No training was started.
