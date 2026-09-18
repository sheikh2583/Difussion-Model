# Windows quickstart

From Command Prompt in the repository root:

```bat
platform\windows\init.cmd
platform\windows\train_cifar.cmd
```

For cross-machine comparisons, label this machine first:

```bat
set DIFFUSION_MACHINE_LABEL=windows-rtx3090
```

Run `platform\windows\train_cifar.cmd` again after an interruption. Continue
mode starts at epoch 1 on a clean clone and resumes the latest completed
checkpoint on an existing run.

For the model-selection menu:

```bat
platform\windows\train.cmd
```

These wrappers preserve and call the original `INIT_ALL.cmd`, `TRAIN.cmd`, and
PowerShell tournament scripts. The legacy files remain in their original paths
so existing shortcuts and documentation continue to work.

The CIFAR-10 configurations save every 10 epochs. Outputs are stored under
`results\<algorithm>_cifar10\` and are ignored by Git.
