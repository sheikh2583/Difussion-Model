# Platform entry points

This directory separates operating-system launchers without forking the shared
training implementation.

| Task | Linux | Windows |
|---|---|---|
| Initialize CIFAR-10 | `./platform/linux/init.sh` | `platform\windows\init.cmd` |
| Interactive training | `./platform/linux/train.sh` | `platform\windows\train.cmd` |
| Train/resume CIFAR-10 suite | `./platform/linux/train_cifar.sh` | `platform\windows\train_cifar.cmd` |
| Build aggregate summary | `./platform/linux/make_summary.sh` | `venv\Scripts\python.exe scripts\aggregate_results.py` |
| Build minimal review ZIP | `./platform/linux/make_minimal_zip.sh` | `venv\Scripts\python.exe scripts\package_review.py` |

See the platform-specific README before the first run:

- [Linux guide](linux/README.md)
- [Windows guide](windows/README.md)

All launchers resolve the repository root from their own location, so they work
even when the clone is stored in a path containing spaces.
