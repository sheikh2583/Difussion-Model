# Platform entry points

This directory separates operating-system launchers without forking the shared
training implementation.

| Task | Linux | Windows |
|---|---|---|
| Initialize CIFAR-10 | `./scripts/linux/init.sh` | `scripts\windows\init.cmd` |
| Interactive training | `./scripts/linux/train.sh` | `scripts\windows\train.cmd` |
| Train/resume CIFAR-10 suite | `./scripts/linux/train_cifar.sh` | `scripts\windows\train_cifar.cmd` |
| Build aggregate summary | `./scripts/linux/make_summary.sh` | `venv\Scripts\python.exe scripts\aggregate_results.py` |
| Build dataset bundles | `venv/bin/python scripts/package_dataset_bundles.py` | `venv\Scripts\python.exe scripts\package_dataset_bundles.py` |
| Generate result GIFs | `./scripts/linux/generate_cifar10_outputs.sh` | `venv\Scripts\python.exe scripts\generate_result_gifs.py --dataset cifar10` |

See the platform-specific README before the first run:

- [Linux guide](linux/README.md)
- [Windows guide](windows/README.md)

All launchers resolve the repository root from their own location, so they work
even when the clone is stored in a path containing spaces.
