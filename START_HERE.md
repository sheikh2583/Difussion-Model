# Start here: clone, initialize, train

The model code, configurations, checkpoint format, and datasets are shared
between operating systems. Only the launch commands differ.

Choose one platform guide:

- [Linux](platform/linux/README.md)
- [Windows](platform/windows/README.md)

## Repository layout

```text
platform/
  linux/                 Linux entry points and commands
  windows/               Windows entry points and commands
config/                  Shared experiment configurations
algorithms/              Shared model algorithms
training/                Shared training and checkpoint engine
results/                 Local outputs (ignored by Git)
data/raw/                Downloaded datasets (ignored by Git)
```

The original root-level Windows launchers are preserved for compatibility.
The files under `platform/` are stable, discoverable wrappers around the
maintained entry points; they do not duplicate the model implementation.

## Recovery rule

The platform training launchers use continue mode. On a clean clone, continue
mode starts at epoch 1. After an interruption, running the same command again
loads the latest completed checkpoint. Checkpoints and self-contained ZIP
archives are written every 10 epochs.

## Comparing machines

Set a stable label before training on each machine. Every metric row and
checkpoint archive then records that label together with the hostname, OS,
Python/PyTorch/CUDA versions, GPU model, Git commit, and dirty/clean state.

Linux:

```bash
export DIFFUSION_MACHINE_LABEL=linux-rtx3090
./platform/linux/train_cifar.sh
```

Windows Command Prompt:

```bat
set DIFFUSION_MACHINE_LABEL=windows-rtx3090
platform\windows\train_cifar.cmd
```

Use the same Git commit and unchanged configs for controlled comparisons.

Do not copy a `venv/` between operating systems. Clone the repository and run
the initialization command on each machine instead.
