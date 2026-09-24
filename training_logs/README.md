# Training-log research guide

This directory is the immutable transcript store. Logs are not moved or
renamed after a run because their original paths are recorded in provenance
headers, JSON sidecars, and source manifests.

For the structured, research-facing view, use:

- `results/aggregate/TRAINING_LOG_INDEX.md` for logs grouped by evidence value;
- `results/aggregate/training_log_catalog.csv` for filtering in a spreadsheet;
- `results/aggregate/training_log_catalog.json` for machine-readable provenance.

The generated index separates four categories:

1. **Completed research results** — full epoch-level training trajectories.
2. **Meaningful negative and partial evidence** — rejected codecs, divergent
   trajectories, interrupted training with useful measurements, and completed
   training followed by a packaging failure.
3. **Supporting artifacts and validation** — successful codec checks, pair
   generation, and other non-epoch prerequisites.
4. **Operational diagnostics** — startup, launcher, environment, and duplicate
   planning failures that produced no scientific measurements.

## Canonical CelebA latent evidence

The canonical suite transcripts are under
`nvidia-geforce-rtx-3090-24gb/latent/<algorithm>/`. Older logs directly under
the `latent/` directory are historical attempts, not the canonical suite.

Important negative evidence is intentionally retained:

- `latent/mf/`: the completed ordinary Mean Flow run whose late loss rise
  motivates the epoch-80 evaluation and Hutchinson-CV diagnostic;
- `latent/reflow/`: completed epoch-100 training followed by an `ENOSPC`
  checkpoint-archive failure;
- `codec/celeba_codec_scratch_kl_vae_train_20260921_182111_pid188820.log`:
  the rejected scratch-codec experiment;
- `latent/mf_hutchinson/celeba_latent_mf_hutchinson_ndag-m-lab_20260923T102538Z.log`:
  the original numerically divergent Hutchinson run.

The `_CUuuid` JSON serialization transcript in `latent/mf_hutchinson/` is an
operational startup failure, not experimental evidence. It remains available
for auditability but is placed in the diagnostic section of the generated
index.

## Current Hutchinson-CV run

The isolated CV experiment uses run-local evidence rather than a central suite
transcript:

`results/mf_hutchinson_cv_celeba_latent/`

Its trainer log, structured metrics, events, frozen config, and environment
provenance are included in the final thesis context. Raw `.pt` checkpoints and
checkpoint ZIPs are deliberately excluded from that compact shareable archive.
