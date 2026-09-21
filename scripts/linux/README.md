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
for example `training_logs/nvidia-geforce-rtx-3090-24gb/`. After all
requested jobs have been attempted, the logs are committed together and pushed
to the current branch. Committing at the end keeps one source-code identity
across every comparison. Large results, checkpoints, datasets, and generated
samples remain excluded from Git.

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

After training finishes, rebuild aggregate tables and dataset-specific GIFs:

```bash
./scripts/linux/make_summary.sh
./scripts/linux/generate_cifar10_outputs.sh
./scripts/linux/generate_celeba_outputs.sh
venv/bin/python scripts/package_dataset_bundles.py
./scripts/linux/make_thesis_context.sh
```

These tools use existing metrics, configs, and atomically published checkpoint
archives. They never start, stop, signal, or modify a training process. The
summary helper refuses to run during training unless `--allow-running` is
explicitly supplied; `--dry-run` prints its command without writing outputs.
The animation launchers follow the same `--dry-run` and `--allow-running`
convention.

`make_thesis_context.sh` produces a compact package for external thesis
discussion. Preview its exact inventory with `--dry-run`; it excludes raw
datasets, checkpoint tensors, checkpoint ZIPs, and FID caches.
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

## Self-trained CelebA codec fallback

Only if the pretrained factor-4 codec is rejected, preview and then launch the
scratch KL-VAE trainer with:

```bash
./scripts/linux/train_scratch_codec.sh --dry-run
./scripts/linux/train_scratch_codec.sh
```

The launcher uses the recommended RTX 3090 starting preset (batch 128, 60
epochs, AdamW at `1e-4`, weight decay `1e-4`, AMP, gradient clipping `1.0`, and
KL warmup `1e-5` to `1e-4` over 20 epochs). It resumes the numerically latest
five-epoch checkpoint automatically and refuses to overwrite checkpoints in
fresh mode. Use `--help` to see safe path and batch-size overrides.
