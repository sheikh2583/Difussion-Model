# Linux quickstart

From the repository root:

```bash
chmod +x scripts/linux/*.sh
./scripts/linux/init.sh
./scripts/linux/train_cifar.sh
```

To run CIFAR-10 and then CelebA unattended with one command:

```bash
./scripts/linux/train_all_datasets.sh
```

The launchers run each model sequentially in dependency order. CIFAR-10 uses
the batch-128 exact-JVP MeanFlow configuration. A separate complete terminal
log is saved for every dataset/model pair under a device-specific directory,
with pixel and latent transcripts separated under `pixel/` and `latent/`, for
example `training_logs/nvidia-geforce-rtx-3090-24gb/pixel/`. The launchers do not
stage, commit, or push files; version-control decisions remain with the
operator. Large results, checkpoints, datasets, and generated samples remain
excluded from Git.

All GPU launchers share the ownership-checked `results/.lock`, and nested suite
calls inherit its random token without reacquiring it. Each suite also freezes
a content hash of training-relevant source/config inputs and verifies it before
starting the next serialized job. Documentation-only edits are excluded.

The machine label is generated automatically from the OS, hostname, GPU, and
VRAM (for example, `linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb`). No setup
is required. To use a shorter custom label instead, optionally run:

```bash
export DIFFUSION_MACHINE_LABEL=linux-rtx3090
```

Run `./scripts/linux/train_cifar.sh` again after an interruption. Completed
models are detected, and an interrupted model resumes from its latest completed
10-epoch checkpoint.

For the model-selection menu:

```bash
./scripts/linux/train.sh
```

Useful checks that do not start training:

```bash
./scripts/linux/train.sh --list
./scripts/linux/train_cifar.sh --dry-run
./scripts/linux/train_all_datasets.sh --dry-run
```

The CIFAR-10 suite keeps each preset's controlled batch size: 128 for FM,
FM-LN, exact-JVP MF, Consistency, and Reflow; 64 for MF-Distill.

Outputs are stored under `results/<algorithm>_cifar10/` and are ignored by Git.

## Summaries, animations, and dataset bundles

After training finishes, rebuild aggregate tables and dataset-specific outputs:

```bash
./scripts/linux/make_summary.sh
./scripts/linux/generate_cifar10_outputs.sh
./scripts/linux/generate_celeba_outputs.sh
venv/bin/python scripts/package_dataset_bundles.py
./scripts/linux/make_thesis_context.sh
./scripts/linux/refresh_thesis_context.sh --interval 300
```

These tools use existing metrics, configs, and atomically published checkpoint
archives. They never start, stop, signal, or modify a training process. The
summary helper refuses to run during training unless `--allow-running` is
explicitly supplied; `--dry-run` prints its command without writing outputs.
The animation launchers follow the same `--dry-run` and `--allow-running`
convention. The CelebA launcher creates separate pixel and latent GIFs, then
uses the shared GPU lock while decoding checkpoint samples into RGB grids. If
training is active, `--allow-running` generates metric GIFs only and skips
checkpoint sampling.

`make_thesis_context.sh` produces a compact package for external thesis
discussion. Preview its exact inventory with `--dry-run`; it excludes raw
datasets, checkpoint tensors, checkpoint ZIPs, and FID caches.
It rebuilds and validates the canonical summary by default, requires the
essential implementation files to be Git-tracked, and verifies every archived
member by SHA-256 before replacing `thesis_context.zip`.
Small ignored provenance records (the collaboration plans/decisions, codec
model card, and latent-cache manifest) are included explicitly; ignored binary
weights, tensor caches, datasets, checkpoints, and FID caches remain excluded.
`refresh_thesis_context.sh` runs this complete operation repeatedly under a
single-instance lock. It retries safely after failures and preserves the last
verified ZIP.
The generated `thesis_context.zip` and reporting-generated
`THESIS_SUMMARY.md` are written to the repository root and are Git-trackable.

Run the read-only repository health check at any time:

```bash
python scripts/verify_project_layout.py
```

Before a structural migration, require training to be stopped:

```bash
python scripts/verify_project_layout.py --fail-if-training
```

## Pretrained CelebA latent codec (primary path)

The active latent experiment uses the frozen `CompVis/ldm-celebahq-256`
VQ-f4 codec. It maps 64×64 RGB images to three-channel 16×16 quantized latents;
every algorithm then trains its own randomly initialized latent U-Net.

Train or resume all six latent algorithms with per-model logs:

```bash
./scripts/linux/train_celeba_latent.sh \
  --machine-label NDAG-M-Lab-RTX3090 \
  --mode continue
```

Use `--dry-run` to preview training without writing logs or starting models.

Preview the exact download and validation commands, then run them:

```bash
./scripts/linux/prepare_pretrained_codec.sh --dry-run \
  --accept-quality-failure \
  --acceptance-reason "User selected the face-specific VQ-f4 latent experiment despite the measured reconstruction ceiling"
./scripts/linux/prepare_pretrained_codec.sh \
  --accept-quality-failure \
  --acceptance-reason "User selected the face-specific VQ-f4 latent experiment despite the measured reconstruction ceiling"
```

On success, continue with the accepted checkpoint:

```bash
venv/bin/python codec/cache_latents.py \
  --codec-path results/codecs/celeba_vq_f4/accepted_codec.pt \
  --celeba-root data/raw --output-dir data/latent_cache \
  --split both --device cuda
```

The preparation launcher resolves `main` to an immutable Hugging Face commit,
records it in `source_manifest.json`, saves a timestamped validation log, and
exits nonzero if a structural or latent-statistics check fails. The explicit
quality override preserves the failed rFID/PSNR values in both the validation
report and accepted checkpoint while allowing the selected latent experiment
to proceed.

## Rejected self-trained CelebA codec experiment

The completed scratch factor-4 codec run is retained as a thesis result, but it
failed the reconstruction-FID gate and must not be resumed or substituted for
the primary codec. The old launcher remains available only for reproducibility
of that historical experiment; running it would create a new experiment.

```bash
./scripts/linux/train_scratch_codec.sh --dry-run
./scripts/linux/train_scratch_codec.sh
```

The historical launcher uses the RTX 3090 preset (batch 128, 60
epochs, AdamW at `1e-4`, weight decay `1e-4`, AMP, gradient clipping `1.0`, and
KL warmup `1e-5` to `1e-4` over 20 epochs). It resumes the numerically latest
five-epoch checkpoint automatically and refuses to overwrite checkpoints in
fresh mode. Use `--help` to see safe path and batch-size overrides.

Every scratch-codec launch writes standard `[run]` metadata and a complete
timestamped transcript under `results/scratch_vae/logs/`. Running
`./scripts/linux/make_summary.sh` discovers it together with every `.log` under
`training_logs/` and `results/`, then writes a generic log catalog and Markdown
index under `results/aggregate/`. This discovery is not tied to a fixed list of
algorithms, so future trainer logs appear automatically.
