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
5. **Read `CROSS_TRACK.md`** and act on every open item addressed to you
   before starting new work. Post an `ACK` reply within the same session.

## Current objective

Fix and statically validate the errors exposed by the completed tournament
without altering the original diverged MF evidence. Agents prepare safe code and
configuration only; the user alone decides whether and when to run models.

## Absolute no-training rule

- Codex and Antigravity/Claude must not execute model training of any length.
- This includes research runs, 15-epoch probes, one-epoch tests, smoke training,
  tournament commands, resumed runs, and fresh runs.
- Agents must not launch GPU sampling/evaluation, pair generation, or performance
  benchmarks as a substitute for training.
- Agents may run read-only inspections, unit tests with synthetic tensors,
  Python compilation, config parsing, workflow verification that does not train,
  shell/PowerShell parsing, and command dry-runs.
- Agents may prepare and document commands for the user, but must not execute
  those commands. When the user supplies logs, agents diagnose and fix errors.

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

Codex may add a unit-tested `--train-only` path so a user-run probe does not
spend time on 5,000-sample FID. Agents must not execute either the probe or the
full run.

## Inter-agent communication protocol

When one agent cannot complete work on its **own** file because it depends on a
change in the **other** agent's owned file, it must not silently leave the work
undone or guess at the other agent's interface. Instead it posts a structured
message to `CROSS_TRACK.md` and continues with whatever it can do independently.

### The channel: `CROSS_TRACK.md`

- Lives at project root alongside `AGENTS.md` and `PLAN.md`.
- **Append-only:** agents may only ADD new entries; never edit or delete
  another agent's text. Use strikethrough or an `ACK`/`RESOLVED` reply entry
  to update status — never overwrite.
- Both agents read it at every session start (rule 5 above).
- The user may also read and add entries.
- Entries are committed inside the normal agent git-lock commit.

### Entry format

Every entry starts with a level-3 heading:

```
### [TYPE] [ID] — [DATE] — FROM: [Agent] → TO: [Agent]

**Status:** OPEN | ACKNOWLEDGED | RESOLVED
**Blocking file (FROM owns):** `path/to/blocked_file.py`
**Depends on (TO owns):** `path/to/dependency_file.py`

[One paragraph describing exactly what is needed and why the cross-track
dependency exists. Be specific: name the function, field, or interface
required. Include the line numbers if helpful.]

**Acceptance criteria:** How the FROM agent will know the dependency is met.
```

### Entry types

| Type | Use when |
|------|----------|
| `BLOCKS` | Your file cannot be completed until the other agent changes theirs |
| `NEEDS_INTERFACE` | You need a new function/field exposed in the other agent's file |
| `DEFECT` | You found a bug in the other agent's owned file (do not fix it yourself) |
| `HANDOFF` | Your work is complete and unlocks the other agent's next step |
| `ACK` | Acknowledging receipt of an item addressed to you |
| `RESOLVED` | Confirming the dependency/defect is fixed; closes the entry |

### Rules

1. **Never fix the other agent's file.** Post a `BLOCKS` or `DEFECT` entry
   instead and continue with what you can do independently.
2. **Never silently skip** work that has a cross-track dependency. Post the
   entry so the other agent knows, even if you think they will discover it
   themselves.
3. **One entry per dependency.** If two blocked files share the same root
   cause, one entry covering both is fine.
4. **ACK within the same session.** When you read an open item addressed to
   you, post an `ACK` reply entry in the same git commit — even if you cannot
   fix it immediately. `ACK` means "seen and understood," not "done."
5. **RESOLVED closes the loop.** After fixing the dependency, post a `RESOLVED`
   entry referencing the original ID and commit hash, then notify the FROM agent
   by updating the original entry's status line.
6. **IDs are sequential per agent prefix:** `AGY-001`, `AGY-002`, … for
   Antigravity; `CDX-001`, `CDX-002`, … for Codex. Choose the next available
   number when posting.
7. **Severity:** Use `[BLOCKS]` for work that cannot proceed at all;
   `[NEEDS_INTERFACE]` for work that can proceed with a stub but needs the
   real interface before the run; `[DEFECT]` for correctness issues that won't
   break agent work but will break user execution.

### What CROSS_TRACK.md is NOT for

- Do not use it for general status updates — use the `PLAN.md` status log.
- Do not use it to request changes to `AGENTS.md` or `PLAN.md` — those
  require explicit user coordination.
- Do not use it to ask the other agent to run models — that is the user's
  exclusive domain.

## GPU and commit locks

- Agents do not launch CUDA training, sampling, evaluation, pair generation, or
  benchmarks. User-run commands should acquire `results/.lock` atomically and
  record host, PID, UTC start time, and the command/run name. CPU-only unit tests
  and dry runs do not take the GPU lock.
- Never remove a lock only because it looks old. Verify the same-host PID is
  dead, or ask the user to confirm it is stale.
- Lock absence prevents contention; it never authorizes an agent-run job. The
  15-epoch probe and full run are user-operated tasks only.
- The git index is also shared. Before staging, acquire the cooperative
  `results/.agent_git.lock`. Only its owner may stage or commit. Stage explicit
  paths, inspect `git diff --cached --name-status` and
  `git diff --cached --check`, commit, then remove the lock. Never use
  `git add .`.

## Verification and hand-off

- All code must remain usable after clone plus `INIT_ALL.cmd` or
  `./init_all.sh`.
- Antigravity runs focused synthetic trainer/config unit tests and records
  changed paths and test output without launching training.
- Codex runs lifecycle/provenance/preflight unit tests and reviews integration
  without launching training, sampling, or GPU evaluation.
- One agent never commits the other agent's files. When both commits exist,
  Codex performs the read-only integration audit defined in `PLAN.md`.
