# Active Shared Plan

Updated: 2026-09-18 (Asia/Dhaka)

Coordinator for `PLAN.md` / `AGENTS.md`: **Antigravity/Claude** after the
run-lifecycle commit. Codex must request a hand-off before editing them again.

## Verified baseline

- The CIFAR-10 tournament completed successfully from 2026-09-17 14:35 to
  2026-09-18 02:57 (about 12 h 22 min).
- Canonical log:
  `results/tournament_run_20260917_143512_pid10672.log`.
- All six algorithms have epoch-100 checkpoints: FM, FM-LogNorm, MF,
  Consistency, MF-Distill, and Reflow.
- Evaluation and aggregation completed. The latest aggregate contains all six
  canonical runs; older review notes that say several models were untrained are
  now stale and must not be copied without re-verification.
- MF completed but is not healthy: its loss rose sharply late in training
  (approximately 0.454 at epoch 95 to 0.795 at epoch 100). Completion is not
  evidence of convergence.
- No new research-model training is authorized in the current phase.

## Parallel ownership

### Track C — Codex

Owned files/directories:

- `train.py`
- `experiments/runner.py`
- `utils/run_lifecycle.py`
- `tests/test_run_lifecycle.py`
- `scripts/interactive_train.py`
- training wrapper/orchestrator scripts only during the current lifecycle task
- lifecycle sections of `README.md`

Current task: **C0 — explicit continue/fresh lifecycle** (`COMPLETE`)

- Add the same `continue` / `fresh` contract to direct CLI, beginner menu,
  Windows/Linux wrappers, suite runner, and tournament runner.
- Preserve old runs in timestamped history; never mix fresh metrics into an old
  run directory.
- Let a completed run be extended by setting a larger total epoch target.
- Add standard-library lifecycle tests and dry-run both platform command plans.

Next Codex tasks (do not start training):

- **C1:** add checkpoint/config provenance validation for future resumes.
- **C2:** prepare, but do not run, MF diagnostic instrumentation and an exact
  JVP versus finite-difference benchmark after Track A completes data forensics.

### Track A — Antigravity/Claude

Owned files/directories:

- `scripts/aggregate_results.py`
- `evaluation/`
- result-schema code in `utils/results.py` and `utils/logging.py`
- forensic analysis utilities and derived files under `results/aggregate/`
- evidence corrections in documentation after metrics are verified

Current task: **A0 — post-tournament data-integrity audit** (`READY`)

- Re-test the review notes against the newly completed tournament instead of
  assuming the pre-tournament findings are still current.
- Verify run discovery and deterministic JSONL selection when a metrics folder
  contains multiple files.
- Print the discovered run set and the ingested aggregate run set, then diff
  them. The canonical six runs must all be present exactly once.
- Investigate concatenated or restarted MF records without rewriting raw logs.
  Copy evidence before any split or normalization.
- Add `epoch`, `checkpoint_path`, and `num_generated_samples` to future
  evaluation records. Do not invent provenance for historical rows.

Next Antigravity/Claude tasks:

- **A1:** regenerate aggregate outputs from the verified source selection and
  report every changed value.
- **A2:** design matched 5,000-sample FM/FM-LogNorm re-evaluation and repeated
  seeds. Do not launch it until the user approves GPU sampling time.
- **A3:** update report conclusions only after A1/A2 evidence exists. Preserve
  the submitted-report scope unless the user explicitly changes it.

## Sequencing and hand-off

1. C0 and A0 may run concurrently because their owned files are disjoint.
2. C1 may start after C0 is committed.
3. C2 waits for A0's MF log/checkpoint identity findings.
4. A1 waits for A0; A2 waits for A1 and user approval for GPU time.
5. MF v2 architecture/config work waits for C2 and A0. Any 10–15 epoch
   validation or full rerun requires explicit user approval.
6. Report tables and claims change only after corrected metrics are generated
   and reviewed.

## Status log

- `[Codex][2026-09-18]` Verified the tournament log and all six epoch-100
  checkpoints. Completed C0 with recoverable fresh-run history, continue/extend
  behavior, four lifecycle unit tests, Python compilation, existing-run guard
  validation, interactive dry runs, PowerShell parsing, and continue/fresh
  tournament dry runs. Bash was statically reviewed because this Windows host
  has no working Bash/WSL runtime.
- `[Antigravity/Claude]` Append status here only after the coordinator hand-off;
  include changed paths, validation performed, and any blocker.
