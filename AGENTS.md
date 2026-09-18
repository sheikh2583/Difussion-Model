# Codex + Antigravity Collaboration Rules

Codex and Antigravity/Claude share one filesystem, branch, git index, GPU, and
results directory. Edits are visible immediately; neither agent should pull to
synchronize local work.

## Session start

Every agent must:

1. Read this file and `PLAN.md` completely.
2. Run `git status --short` and `git log -3 --oneline`.
3. Confirm its task is marked `READY` or `IN PROGRESS` in `PLAN.md`.
4. Touch only its owned paths below. Report a cross-track defect instead of
   fixing it in the other agent's files.

## Current objective

Prepare a fresh Mean Flow v2 stability experiment without altering the original
diverged MF evidence. No GPU run is authorized merely by these instructions.

## Fixed file ownership

### Antigravity/Claude — trainer, configuration, and data integrity

Antigravity exclusively owns during this phase:

- `config/config.py`
- `training/trainer.py`
- new `config/mf_full_v2.json`
- new `tests/test_trainer_controls.py`
- `scripts/aggregate_results.py`
- `evaluation/`
- `utils/results.py` and `utils/logging.py`
- forensic utilities and derived outputs under `results/aggregate/`

Antigravity must not edit Codex-owned files, `algorithms/mean_flow.py`, the old
`config/mf_full.json`, or the submitted report during this phase.

### Codex — lifecycle, provenance, and MF v2 preflight

Codex exclusively owns during this phase:

- `train.py`
- `experiments/runner.py`
- `utils/run_lifecycle.py`
- new checkpoint/config provenance helpers under `utils/`
- `tests/test_run_lifecycle.py` and new provenance/preflight tests
- new MF v2 validation/preflight scripts under `scripts/`
- MF v2 command documentation in `README.md`

Codex must not edit the trainer/config files owned by Antigravity or any
algorithm mathematics.

### Frozen evidence and shared files

- `algorithms/mean_flow.py` is read-only until the v2 probe fails and the user
  explicitly approves an exact-JVP investigation.
- `config/mf_full.json` and `results/mf_cifar10/` describe the diverged run and
  must not be modified, moved, merged, or overwritten.
- `THESIS_REVIEW_PACKAGE.md`, `verified_findings_and_scaffolded_plan.md`, and
  `docs/RUN 1 logs` are untracked evidence owned by the user. Do not stage,
  edit, rename, or delete them.
- `AGENTS.md` and `PLAN.md` are read-only after this coordination commit unless
  the user explicitly appoints a coordinator for another update.

## Agreed implementation contract

The pasted MF prompt uses field names from a different schema. Implement the
current repository schema only:

- `epochs`, not `num_epochs`
- `amp`, not `use_amp`
- `optim.learning_rate`, not `optim.lr`
- existing `optim.scheduler`, not a new `lr_scheduler`
- evaluation settings nested under `evaluation`

Cosine scheduling already exists in `training/trainer.py`, steps once per epoch,
and is checkpointed. Do not add a second scheduler mechanism. MF v2 uses:

```json
"optim": {
  "optimizer": "adamw",
  "learning_rate": 0.0001,
  "weight_decay": 0.0001,
  "gradient_clip_norm": 1.0,
  "scheduler": "cosine",
  "scheduler_kwargs": {"eta_min": 0.000001}
}
```

Add only `gradient_clip_norm: Optional[float] = None` to `OptimConfig`. With
AMP, clipping must occur after `GradScaler.unscale_(optimizer)` and before
`GradScaler.step(optimizer)`. Clip the union of parameters from
`self.trainable_modules`; `BaseAlgorithm` is not guaranteed to implement
`.parameters()`.

`mf_full_v2.json` uses the current nested config shape, batch 128, 100 epochs,
AMP, seed 0, a 5,000-sample evaluation cache with a distinct filename, and the
requested MF kwargs (`p_same=0.25`, `p_fd_step=0.5`, delta `1e-3` to `1e-4`).
The controlled `experiment_name="mf_v2"` variant is permitted so its evidence
lands in `results/mf_v2_cifar10/`; algorithm selection remains `--algorithm mf`.

Do not add EMA or change r-conditioning in this phase. This run intentionally
tests clipping, cosine decay, batch size, learning rate, weight decay, and delta
without changing MF mathematics.

## Probe versus full run

The 15-epoch probe and 100-epoch run are separate experiments. Never continue
the probe to epoch 100: its cosine scheduler checkpoint has `T_max=15`.

- Probe: `mf_v2_probe_cifar10`, 15 epochs, training only.
- Full: `mf_v2_cifar10`, fresh from epoch 1 with `T_max=100`.

Codex may add a tested `--train-only` path so the probe does not spend time on
5,000-sample FID. The full run must use the normal evaluation workflow.

## GPU and commit locks

- Any CUDA training, sampling, evaluation, pair generation, or benchmark must
  acquire `results/.lock` atomically. Record host, PID, UTC start time, and the
  command/run name. CPU tests and dry runs do not take the GPU lock.
- Never remove a lock only because it looks old. Verify the same-host PID is
  dead, or ask the user to confirm it is stale.
- Lock absence prevents contention; it does not authorize a run. The 15-epoch
  probe and the full run each require explicit user approval.
- The git index is also shared. Before staging, acquire the cooperative
  `results/.agent_git.lock`. Only its owner may stage or commit. Stage explicit
  paths, inspect `git diff --cached --name-status` and
  `git diff --cached --check`, commit, then remove the lock. Never use
  `git add .`.

## Verification and hand-off

- All code must remain usable after clone plus `INIT_ALL.cmd` or
  `./init_all.sh`.
- Antigravity runs focused trainer/config tests and records changed paths and
  test output without launching training.
- Codex runs lifecycle/provenance/preflight tests and reviews integration
  without launching training.
- One agent never commits the other agent's files. When both commits exist,
  Codex performs the read-only integration audit defined in `PLAN.md`.
