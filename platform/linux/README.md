# Linux quickstart

From the repository root:

```bash
chmod +x platform/linux/*.sh
./platform/linux/init.sh
./platform/linux/train_cifar.sh
```

To run CIFAR-10 and then CelebA unattended with one command:

```bash
./platform/linux/train_all_datasets.sh
```

The launchers run each model sequentially in dependency order. CIFAR-10 uses
the batch-128 exact-JVP MeanFlow configuration. A separate complete terminal
log is saved for every dataset/model pair under `training_logs/`. After all
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
./platform/linux/train_all_datasets.sh --dry-run
```

The CIFAR-10 suite keeps each preset's controlled batch size: 128 for FM,
FM-LN, exact-JVP MF, Consistency, and Reflow; 64 for MF-Distill.

Outputs are stored under `results/<algorithm>_cifar10/` and are ignored by Git.
