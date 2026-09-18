# Concurrent Development Rules

Codex and Antigravity/Claude share this working tree. File changes are visible
immediately; git pull/push is not a synchronization mechanism between local
agents.

## Start-of-session checklist

1. Read `PLAN.md` and this file.
2. Run `git status --short` before editing.
3. Work only in the file ownership listed in `PLAN.md`.
4. Treat existing modified or untracked files as another contributor's work
   unless the plan explicitly assigns them to you.

## Collision prevention

- Never edit a file owned by the other active track.
- `AGENTS.md` and `PLAN.md` have one coordinator at a time. The current
  coordinator is recorded in `PLAN.md`; request a hand-off before editing them.
- Stage explicit paths only. Do not use `git add .` or commit another track's
  files.
- Before committing, inspect `git diff --cached --name-status` and
  `git diff --cached --check`.
- Do not rewrite, clean, reset, or delete another agent's changes or local
  results.
- The untracked review documents `THESIS_REVIEW_PACKAGE.md` and
  `verified_findings_and_scaffolded_plan.md` are evidence inputs. Do not edit,
  stage, or delete them unless `PLAN.md` assigns that work.

## Runtime and GPU rules

- The completed CIFAR-10 tournament is the current evidence baseline. Do not
  start new model training unless the user explicitly approves it.
- Sampling-only evaluation and benchmarks must still use `results/.lock` so
  two agents do not contend for the GPU.
- A missing lock file does not grant permission for a long-running job; it only
  prevents accidental concurrency.
- Preserve raw JSONL, checkpoints, and generated datasets before forensic
  changes. Derived aggregate files may be regenerated after their source set is
  recorded.

## Run lifecycle contract

- Every training front end exposes `continue` and `fresh` modes.
- `continue` resumes the numerically latest compatible checkpoint.
- `fresh` never deletes the previous canonical run. It moves it into
  `results/history/<run>_<timestamp>/` before starting at epoch 1.
- Direct training against a non-empty run directory without an explicit mode
  must fail safely instead of mixing logs or overwriting checkpoints.
- To extend a completed run, set a total epoch target greater than its latest
  checkpoint epoch.

## Verification standard

Changes must work after a clean clone followed by the platform initializer:
`INIT_ALL.cmd` on Windows or `./init_all.sh` on Linux. Use the project virtual
environment, keep paths relative to the repository, and provide equivalent
Windows/Linux behavior for user-facing scripts.
