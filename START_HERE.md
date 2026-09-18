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

Do not copy a `venv/` between operating systems. Clone the repository and run
the initialization command on each machine instead.

