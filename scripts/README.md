# Platform entry points

This directory separates operating-system launchers without forking the shared
training implementation.

| Task | Linux | Windows |
|---|---|---|
| Initialize CIFAR-10 | `./scripts/linux/init.sh` | `scripts\windows\init.cmd` |
| Interactive training | `./scripts/linux/train.sh` | `scripts\windows\train.cmd` |
| Train/resume CIFAR-10 suite | `./scripts/linux/train_cifar.sh` | `scripts\windows\train_cifar.cmd` |
| Build aggregate summary | `./scripts/linux/make_summary.sh` | `.\scripts\windows\make_summary.ps1` |
| Build dataset bundles | `venv/bin/python scripts/package_dataset_bundles.py` | `venv\Scripts\python.exe scripts\package_dataset_bundles.py` |
| Build compact thesis context | `./scripts/linux/make_thesis_context.sh` | `.\scripts\windows\make_thesis_context.ps1` |
| Generate CIFAR-10 GIFs | `./scripts/linux/generate_cifar10_outputs.sh` | `.\scripts\windows\generate_cifar10_outputs.ps1` |
| Generate CelebA GIFs | `./scripts/linux/generate_celeba_outputs.sh` | `.\scripts\windows\generate_celeba_outputs.ps1` |
| Verify project layout | `python scripts/verify_project_layout.py` | `python scripts\verify_project_layout.py` |

See the platform-specific README before the first run:

- [Linux guide](linux/README.md)
- [Windows guide](windows/README.md)

All launchers resolve the repository root from their own location, so they work
even when the clone is stored in a path containing spaces.

The compact thesis context ZIP is intended for external reviewers and agents.
It includes source, configs, report material, metrics, provenance, plots,
samples, and training logs, but excludes raw datasets and checkpoint binaries.
