# Linux quickstart

From the repository root:

```bash
chmod +x platform/linux/*.sh
./platform/linux/init.sh
./platform/linux/train_cifar.sh
```

The CIFAR launcher runs the full dependency-aware suite sequentially and uses
the batch-128 exact-JVP MeanFlow configuration. It saves the complete terminal
stream under `training_logs/`. When the workflow exits, it commits only that
text log and attempts to push it to the current branch. Large results,
checkpoints, datasets, and generated samples remain excluded from Git.

The machine label is generated automatically from the OS, hostname, GPU, and
VRAM (for example, `linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb`). No setup
is required. To use a shorter custom label instead, optionally run:

```bash
export DIFFUSION_MACHINE_LABEL=linux-rtx3090
```

Run `./platform/linux/train_cifar.sh` again after an interruption. Completed
models are detected, and an interrupted model resumes from its latest completed
10-epoch checkpoint.

For the model-selection menu:

```bash
./platform/linux/train.sh
```

Useful checks that do not start training:

```bash
./platform/linux/train.sh --list
./platform/linux/train_cifar.sh --dry-run
```

The CIFAR-10 suite keeps each preset's controlled batch size: 128 for FM,
FM-LN, exact-JVP MF, Consistency, and Reflow; 64 for MF-Distill.

Outputs are stored under `results/<algorithm>_cifar10/` and are ignored by Git.
