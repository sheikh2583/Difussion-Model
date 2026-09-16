# AGENTS.md — Shared context for all AI coding agents (Antigravity, Codex, etc.)

## What this project is

A research scaffold comparing 6 generative image algorithms (Flow Matching, Mean Flow,
Consistency Models, Rectified Flow Reflow) on CIFAR-10 and CelebA.
All algorithms share one backbone, one dataloader, one trainer, one sampler, one evaluator.

## Repository layout

```
train.py / evaluate.py       CLI entry points (always stay in root)
bootstrap.py                 Universal setup: detects GPU, installs PyTorch, downloads data
INIT_ALL.cmd / init_all.sh   Beginner double-click launchers
TRAIN.cmd / train_interactive.sh  Beginner training menu

algorithms/   6 algorithm implementations + BaseAlgorithm interface + ALGORITHM_REGISTRY
config/       JSON experiment presets  (experiment_name = algorithm key only, NO dataset suffix)
data/         Dataset loaders + dataset_registry.py (routes CIFAR-10 / CelebA by config)
experiments/  ExperimentRunner — wires all components, derives run_dir automatically
models/       SimpleUNet backbone
training/     Trainer (AMP, checkpointing, zip archiving)
sampling/     Sampler
evaluation/   FID / IS evaluator
utils/        Logging, plotting, timing, seeding, checkpoints.py
scripts/      All helper scripts (train_all, evaluate_all, interactive_train, etc.)
web/          inference_server.py (auto-discovers runs) + HTML pages
docs/         IMPLEMENTATION_LOG.md, THEORY_NOTES.md, IMPLEMENTATION_CHANGES.md
```

## Critical design rules — do not break these

1. **`experiment_name` in configs = algorithm key only** (`"fm"`, `"mf"`, etc.).
   The runner appends `_<dataset.name>` automatically:
   `run_dir = results/<experiment_name>_<dataset.name>/`

2. **No dataset names hardcoded in source code.** `cifar10` / `celeba` only appear
   in config files (`dataset.name` field) and the derived run directory path.

3. **`web/inference_server.py` auto-discovers runs** — no hardcoded model catalog.
   It scans `results/` for dirs with `config.json` + `checkpoints/*.pt`.

4. **No hardware names in configs or scripts.** Use `fm_lognorm_budget` (intent),
   not `fm_lognorm_rtx3060` (machine).

5. **`algorithm_kwargs` must not shadow shared controls** (batch_size, epochs,
   optimizer, lr, backbone, dataset, seed, evaluation). Enforced at runtime by
   `experiments/runner.py::assert_no_protected_key_override`.

6. **`evaluate.py` uses `utils.checkpoints.load_algorithm_state`** — handles both
   checkpoint schemas (legacy `module_state_dicts` and current `model_state`).

7. **`bootstrap.py` uses `python -m pip`** (not a `VENV_PIP` variable). It supports
   `--gpu cuda118|cuda121|cuda128|rocm|cpu` and `--datasets cifar10|celeba|all|none`.

8. **Canonical entry points** (`train.py`, `evaluate.py`) stay in the project root.
   `inference_server.py` lives in `web/`.

## Registered algorithms

| Key | Class | Config file |
|---|---|---|
| `fm` | FlowMatchingAlgorithm | `config/fm_full.json` |
| `fm_lognorm` | FlowMatchingLognormAlgorithm | `config/fm_lognorm_full.json` |
| `mf` | MeanFlowAlgorithm | `config/mf_full.json` |
| `mf_distill` | MeanFlowDistillAlgorithm | `config/mf_distill_full.json` |
| `consistency` | ConsistencyAlgorithm | `config/consistency_full.json` |
| `reflow` | ReflowAlgorithm | `config/reflow_full.json` |
| `mock` | MockAlgorithm | `config/smoke_fast.json` |

## Known missing items (create these before using)

- `config/fm_lognorm_celeba64.json`
- `config/mf_distill_celeba64.json`
- `config/consistency_celeba64.json`
- `config/reflow_celeba64.json`

These are referenced by `scripts/interactive_train.py` for the CelebA menu options.

## Workflow for dual-agent editing

**Before starting any session:**
```bash
git pull
```

**After completing any task:**
```bash
git add -A && git commit -m "<description>" && git push
```

Both agents must pull before editing and push after. Never leave uncommitted changes
when switching agents — the other agent will not see them.

## Naming conventions

- Config files: `<algorithm>_<variant>.json` — e.g. `fm_full.json`, `fm_lognorm_budget.json`
- Experiment run dirs: `<experiment_name>_<dataset.name>` — derived by runner, not set manually
- Script files: `snake_case.py` / `snake_case.sh` / `snake_case.ps1`
- No GPU model names anywhere (`rtx3060`, `4070`, etc.)
