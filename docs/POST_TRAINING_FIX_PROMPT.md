# Post-training implementation prompt

> **Historical acceptance specification:** these safeguards and the verified
> log migration were implemented on 22–23 September 2026. Do not rerun the
> migration against active logs. The current operational instructions are in
> `README.md` and `RUNBOOK.md`; this file remains as design/audit evidence.

Use this prompt only after `scripts/linux/train_celeba_latent.sh` and every
child `train.py`/Reflow-generation process have exited. Do not interrupt or
modify the active experiment to apply these changes.

## Objective

Strengthen run isolation, provenance, and log naming for future unattended
multi-model GPU suites without changing any measured result, checkpoint,
metric, dataset, cache, or training-log content.

## Required changes

1. Add one shared, ownership-checked GPU lock implementation and use it from
   `train_celeba_latent.sh`, `train_all.sh`, `train_all_datasets.sh`, Reflow
   generation, codec validation, and the tournament launchers. A launcher must
   reject a genuinely active owner, report stale-lock recovery instructions,
   and remove only a lock whose token it owns. Nested suite calls must not
   deadlock; either the top-level suite owns the lock and passes an explicit
   child token, or each serialized GPU step acquires/releases the same lock.

2. Freeze a training-relevant source identity at suite startup. Hash the live
   algorithm, model, trainer, dataset, config, evaluator, sampler, and launcher
   inputs used by the planned jobs. Before starting each later model, recompute
   the identity and stop with an actionable error if those inputs changed.
   Documentation-only edits must not invalidate the identity.

3. Record that source identity, the selected config digest, lifecycle mode,
   resolved checkpoint series, machine label, and parent-suite timestamp in
   every per-model log and run-environment record. Do not represent a dirty
   worktree solely by its last Git commit.

4. In `continue` mode, skip an already-complete epoch-N model only after
   verifying checkpoint provenance. Do not rerun final evaluation implicitly;
   expose a separate explicit evaluation option.

5. Add CPU-only tests for active-lock rejection, stale-lock handling, nested
   suite behavior, source-change rejection between jobs, documentation-only
   change tolerance, and completed-run skipping. Add shell dry-runs proving the
   seven latent jobs, including latent-only MF-Hutchinson, remain serialized in
   dependency order.

6. After confirming that every trainer, data-loader child, evaluator, and suite
   process has exited, migrate historical central transcripts to an explicit
   representation layout:

   - `training_logs/<device>/pixel/<dataset>_pixel_<algorithm>_...log`
   - `training_logs/<device>/latent/celeba_latent_<algorithm>_...log`
   - keep codec-validation and scratch-codec transcripts under a distinct
     `codec/` category;
   - include logs currently stored directly under `training_logs/` as well as
     those already under a device directory;
   - move each `.log.meta.json` sidecar with its transcript and update its
     recorded `path` if present;
   - create a machine-readable old-path → new-path manifest containing the
     pre/post SHA-256 digest, representation, dataset, algorithm, and move UTC;
   - reject collisions instead of overwriting, preserve log bytes exactly, and
     verify that every pre/post digest is identical;
   - rebuild the training-log catalog and update documentation references after
     the migration.

   Do not rename the currently open FM-LN transcript or any later transcript
   produced by the already-running parent suite. Existing sidecars provide the
   temporary pixel/latent distinction until this migration is safe.

## Constraints

- Preserve all existing checkpoints, results, caches, datasets, experiment
  directory names, and log contents. Log paths may change only through the
  verified migration in requirement 6.
- Do not execute training, CUDA sampling, evaluation, caching, downloads, or
  Reflow generation while implementing the safeguards.
- Avoid broad formatting changes in algorithm/training files; the repository
  currently has substantial legacy lint debt that should be handled separately
  from behavioral changes.
- Run compilation, shell syntax checks, the static latent smoke test, and the
  complete CPU test suite before handoff.

## Acceptance criteria

- A second GPU workflow cannot start while one owns the project lock.
- A source/config change between serialized jobs prevents the next job from
  starting and names the changed inputs.
- Completed compatible jobs are skipped deterministically in `continue` mode.
- Every log path and catalog row states `pixel`, `latent`, or `codec`; the rename
  manifest proves that migration did not change transcript contents.
- All existing and new CPU/static checks pass without touching experiment
  artifacts.
