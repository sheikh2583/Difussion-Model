# Windows quickstart

From Command Prompt in the repository root:

```bat
scripts\windows\init.cmd
scripts\windows\train_cifar.cmd
```

The machine label is generated automatically from the OS, hostname, GPU, and
VRAM. No setup is required. To use a shorter custom label instead, optionally
run:

```bat
set DIFFUSION_MACHINE_LABEL=windows-rtx3090
```

Run `scripts\windows\train_cifar.cmd` again after an interruption. Continue
mode starts at epoch 1 on a clean clone and resumes the latest completed
checkpoint on an existing run.

For the model-selection menu:

```bat
scripts\windows\train.cmd
```

All Windows Command Prompt and PowerShell entrypoints live in this directory.
They call the platform-independent Python tools in the parent `scripts`
directory, so training and evaluation logic remains shared with Linux.

The CIFAR-10 configurations save every 10 epochs. Outputs are stored under
`results\<algorithm>_cifar10\` and are ignored by Git.

## Reporting and verification

Preview output commands without writing anything:

```powershell
.\scripts\windows\make_summary.ps1 -DryRun
.\scripts\windows\generate_cifar10_outputs.ps1 -DryRun
.\scripts\windows\generate_celeba_outputs.ps1 -DryRun
.\scripts\windows\make_thesis_context.ps1 -DryRun
python scripts\verify_project_layout.py
```

The reporting wrappers refuse to modify generated outputs while training is
active unless `-AllowRunning` is explicitly supplied for a read-only snapshot.
They never stop or signal a training process. The CelebA wrapper keeps pixel
and latent animations separate and generates decoded checkpoint grids only
when training is inactive; `-AllowRunning` skips checkpoint sampling.
