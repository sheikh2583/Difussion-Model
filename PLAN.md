# Active Shared Plan — MF v2 Stability Run

Updated: 2026-09-18 (Asia/Dhaka)

This plan is authoritative for Codex and Antigravity/Claude. It replaces prior
task splits. The user authorized this coordination revision; after it is
committed, `AGENTS.md` and `PLAN.md` are read-only until another explicit
coordination request.

## Verified evidence baseline

- First tournament attempt:
  `results/tournament_run_20260917_134737_pid11464.log`.
  It stopped safely when a 1,000-image FID reference cache was incompatible
  with the requested 5,000-image evaluation. Preserve this log as evidence of
  the cache-integrity guard.
- Successful tournament:
  `results/tournament_run_20260917_143512_pid10672.log`, approximately 12 h 22 m.
- FM, FM-LogNorm, MF, Consistency, MF-Distill, and Reflow all reached epoch 100;
  evaluation found all six checkpoints and aggregation completed.
- Five runs are usable for further analysis. MF completed mechanically but its
  loss diverged late, reaching approximately 0.493, 0.530, 0.601, and 0.795 at
  epochs 97–100. Its epoch-100 checkpoint is evidence of failure, not a valid
  model to resume.
- Preserve `config/mf_full.json`, `results/mf_cifar10/`, and their raw JSONL and
  checkpoints unchanged.

## Corrections to the pasted restart prompt

The intended intervention is accepted, but commands and config fields must
match the current repository:

- Use `epochs`, `amp`, `optim.learning_rate`, `optim.scheduler`,
  `optim.scheduler_kwargs`, and nested `evaluation` fields.
- Cosine scheduling is already implemented, stepped per epoch, and persisted in
  checkpoints. Only AMP-safe gradient clipping is missing.
- The proposed tournament `-SkipFm/-SkipMf/...` command is invalid because the
  current tournament script has no such switches. Use the direct MF commands
  below.
- A 15-epoch cosine probe must not be continued to 100 epochs because its saved
  scheduler horizon is 15. The probe and full run use separate run names, and
  the full run starts fresh from epoch 1.
- `experiment_name="mf_v2"` is an explicit evidence-isolation exception for
  this variant; the selected algorithm key remains `mf`.

## Parallel work tracks

### A0 — Preserve and audit MF evidence

Owner: **Antigravity/Claude**

Status: `READY`

GPU: none

- Identify the exact MF epoch-100 checkpoint, config snapshot, and JSONL used by
  the successful tournament.
- Record hashes and sizes without modifying raw evidence.
- Confirm the late-loss values and ensure aggregation does not silently select
  a different historical metrics file.
- Continue the existing post-tournament aggregate/provenance audit in only the
  Antigravity-owned paths listed in `AGENTS.md`.

### A1 — Implement shared stabilization controls and MF v2 config

Owner: **Antigravity/Claude**

Status: `READY` (may run alongside C1/C2)

GPU: none

- Add optional `gradient_clip_norm` to `OptimConfig` with default `None` so all
  existing configs retain their behavior.
- Implement AMP-correct clipping: unscale, clip all trainable-module
  parameters, then call the scaler step.
- Reuse the existing cosine scheduler; do not create a duplicate scheduler
  option.
- Create `config/mf_full_v2.json` in the current schema:
  batch 128, epochs 100, learning rate `1e-4`, weight decay `1e-4`, clip 1.0,
  cosine scheduler with `eta_min=1e-6`, AMP, seed 0, 5,000 generated samples,
  NFE `[1,2,5,10,20]`, distinct 5,000-image FID cache, `p_same=0.25`,
  `p_fd_step=0.5`, and delta `1e-3 -> 1e-4`.
- Do not modify `algorithms/mean_flow.py`, add EMA, or change r-conditioning.
- Add focused CPU tests in `tests/test_trainer_controls.py` for config parsing,
  scheduler construction, disabled-by-default behavior, and clipping order.

Acceptance: focused tests and compilation pass; no training is launched.

### C1 — Resume/checkpoint provenance guard

Owner: **Codex**

Status: `READY` (parallel with A0/A1)

GPU: none

- Add future checkpoint metadata sufficient to verify algorithm class, dataset,
  backbone/config identity, and run variant before `continue` loads weights.
- Reject incompatible resumes with a clear message; warn, rather than invent
  provenance, for legacy checkpoints that predate the metadata.
- Keep target-epoch extension legal while protecting algorithm, dataset,
  backbone, and algorithm-specific kwargs.
- Work only in Codex-owned lifecycle/provenance files. If trainer payload changes
  are required, document the required interface for Antigravity instead of
  editing `training/trainer.py` concurrently.

Acceptance: lifecycle/provenance unit tests pass and old checkpoints remain
loadable through the documented legacy path.

### C2 — MF v2 probe/preflight workflow

Owner: **Codex**

Status: `READY` (parallel with A1)

GPU: none

- Add a training-only CLI path or equivalent targeted wrapper so a short probe
  does not trigger the 5,000-sample final evaluation.
- Add an MF v2 preflight that checks the config schema, distinct run directory,
  old MF evidence presence, cache path, scheduler horizon, batch size, clipping,
  and requested algorithm kwargs without starting training.
- Add Windows/Linux commands and tests. Do not add invalid tournament skip
  flags.

Planned probe command after implementation and user approval:

```powershell
venv\Scripts\python.exe train.py --algorithm mf `
  --config config/mf_full_v2.json `
  --experiment-name mf_v2_probe --epochs 15 --mode fresh --train-only
```

Linux equivalent:

```bash
venv/bin/python train.py --algorithm mf \
  --config config/mf_full_v2.json \
  --experiment-name mf_v2_probe --epochs 15 --mode fresh --train-only
```

Acceptance: dry-run/preflight and unit tests pass; no GPU job starts.

## Integration gate

### I0 — Joint code review and clean-clone validation

Owners: **Codex reviews; Antigravity addresses only its owned-file findings**

Status: `BLOCKED` on A1, C1, and C2

GPU: none

- Confirm existing FM/FM-LN/Consistency/MF-Distill/Reflow configs are unchanged.
- Confirm original MF config/results are unchanged.
- Run unit tests, `scripts/verify_workflow.py --dataset none`, Python compile,
  PowerShell parsing/dry runs, and Linux static checks available on the host.
- Confirm fresh-clone initialization still supplies every Python dependency.
- Commit each track separately using the cooperative git-index lock.

## GPU gates — no automatic launch

### G0 — 15-epoch disposable probe

Status: `BLOCKED` on I0 and explicit user approval

- Acquire `results/.lock`.
- Start `mf_v2_probe_cifar10` fresh; never resume old `mf_cifar10`.
- Report loss and peak VRAM at epochs 1, 5, 10, and 15.
- Pass criteria: epoch-15 loss below epoch 1, no sustained upward trend or
  non-finite values, oscillation approximately within the user-approved range,
  and peak VRAM below 4,000 MB.
- Release the lock and stop. Do not continue this probe to 100 epochs.

### G1 — Fresh 100-epoch MF v2

Status: `BLOCKED` on a passing G0 and separate explicit user approval

```powershell
venv\Scripts\python.exe train.py --algorithm mf `
  --config config/mf_full_v2.json --mode fresh
```

- Starts `results/mf_v2_cifar10/` from epoch 1 with scheduler `T_max=100`.
- Monitor early epochs and checkpoint every configured interval.
- Stop on non-finite loss, renewed explosive divergence, or unexpected memory
  growth. Preserve partial evidence if stopped.

### E0 — Evaluate and aggregate MF v2

Status: `BLOCKED` on successful G1

- Evaluate the epoch-100 MF v2 checkpoint at the configured 5,000 samples and
  NFE grid.
- Regenerate derived aggregates only after deterministic source selection is
  verified by A0.
- Report MF v2 separately from the original diverged MF; never merge their logs.

## Status log

- `[Codex][2026-09-18]` Corrected the restart prompt against the live schema and
  existing scheduler implementation; defined disjoint ownership, separate probe
  and full-run horizons, commit/GPU locks, and explicit approval gates.
- `[Antigravity/Claude]` On completion, report owned paths changed, tests run,
  evidence hashes recorded, and blockers. Do not edit this plan to report status
  unless the user explicitly assigns another coordination update.
