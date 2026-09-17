# Diffusion Models: Comparative Study

This repository is a runnable research scaffold for comparing six image-generation
algorithms with a shared backbone, data pipeline, sampler, and evaluation stack.
It supports CIFAR-10 (32x32) and CelebA (64x64), persists resumable checkpoints,
and includes a small browser UI for inspecting saved models.

## Easiest setup and training

No command-line knowledge is required on Windows:

1. Double-click **`INIT_ALL.cmd`**. It finds or installs Python, creates the
   project environment, selects a supported GPU/CPU PyTorch build, installs all
   dependencies, and downloads both CIFAR-10 and CelebA.
2. Double-click **`TRAIN.cmd`**. Choose a model from the numbered menu and
   confirm. Training starts immediately.
3. Find checkpoints under `results/<run>/checkpoints/`. Every checkpoint is
   automatically zipped, and the completed run is exported to
   `results/exports/<run>.zip`.

On Linux, open a terminal in the cloned repository and run:

```bash
./init_all.sh
./train_interactive.sh
```

The executable bits are stored in Git, so these commands work immediately after
a normal Linux clone. If a ZIP download stripped permissions, restore them with
`chmod +x init_all.sh train_interactive.sh scripts/*.sh`.

The training menu includes the smoke test and all six research algorithms for
both datasets. If a selected distillation model needs an FM teacher, the menu
trains it first. If Reflow pairs are missing, it trains the teacher if needed
and generates the 50,000-pair artifact before starting Reflow. These automatic
prerequisites can take hours and require substantial disk space.

CelebA is hosted upstream on Google Drive. If its automated download is blocked
by a quota, follow the manual fallback below and rerun the initializer; already
completed setup steps are safely reused.

## One-command setup

The cross-platform entry point is `bootstrap.py`. It uses only the Python standard
library, so it works on Windows, Linux, and macOS. It creates `venv/`, detects an
NVIDIA/ROCm/CPU PyTorch target, installs dependencies, creates runtime folders, and
downloads the requested dataset.

```bash
git clone https://github.com/sheikh2583/Difussion-Model.git DiffusionProject
cd DiffusionProject
./init_all.sh                              # Linux: install everything + both datasets
```

On Windows, clone the same repository, open the `DiffusionProject` folder, and
double-click `INIT_ALL.cmd`. Both launchers can install Python when it is missing,
create `venv/`, install the appropriate dependencies, and download both datasets.

The default download is CIFAR-10 (~170 MB), which is enough for the smoke test and
all `*_full.json` CIFAR-10 presets. GPU PyTorch wheels can themselves be several GB.
It is safe to rerun setup: existing environments and datasets are reused.

Windows PowerShell:

```powershell
.\scripts\setup.ps1 -Yes
```

Linux/macOS:

```bash
./scripts/setup.sh --yes
```

`INIT_ALL.cmd` and `init_all.sh` are the beginner launchers and request both
datasets. The lower-level setup commands above are useful when only one dataset
is wanted.

### Dataset download options

| Command | Result |
|---|---|
| `python bootstrap.py --yes` | Installs dependencies and gets CIFAR-10 (~170 MB). |
| `python bootstrap.py --yes --datasets celeba` | Installs dependencies and gets CelebA (~1.4 GB). |
| `python bootstrap.py --yes --datasets all` | Gets both datasets. |
| `python bootstrap.py --yes --datasets none` | Installs dependencies only; use when data is already present or offline. |

Datasets live in `data/raw/` and are intentionally ignored by Git. CelebA is
downloaded through torchvision; if its upstream Google Drive source is temporarily
unavailable, place the extracted images and official annotation files under
`data/raw/celeba/`, then use `--datasets none`.

The setup wrappers reuse a healthy project `venv` first and install Python 3.12
through `winget` on Windows or a supported system package manager on Linux when
Python 3.9+ is absent. Directly invoking `bootstrap.py` still requires Python.
Pass `--gpu cpu`, `--gpu cuda118`, `--gpu cuda121`, `--gpu cuda128`, or
`--gpu rocm` to override hardware detection. `--skip-torch` is for a
pre-populated project virtual environment only.

## First run

Activate the environment when invoking commands directly:

```bash
source venv/bin/activate              # Linux/macOS
.\venv\Scripts\Activate.ps1          # Windows PowerShell

# CPU-safe, two-epoch end-to-end check
python train.py --algorithm mock --config config/smoke_fast.json
```

## Algorithms

| Key | Method | Prerequisite |
|---|---|---|
| `fm` | Flow Matching (uniform time) | None |
| `fm_lognorm` | Flow Matching (logit-normal time) | None |
| `mf` | Mean Flow | None |
| `mf_distill` | Mean Flow Distillation | Trained FM teacher |
| `consistency` | Consistency Models | Trained FM teacher |
| `reflow` | Rectified Flow Reflow | FM checkpoint and generated reflow pairs |

Teacher-based methods should report the cost of their FM teacher separately from
their own training run.

## Training

Run one experiment with an explicit preset:

```bash
python train.py --algorithm fm --config config/fm_full.json
python train.py --algorithm mf --config config/mf_full.json
python train.py --algorithm consistency --config config/consistency_full.json
```

### Full two-dataset tournament

The full-tournament scripts run one GPU job at a time in dependency order:
FM, FM-LogNorm, Mean Flow, Consistency, Mean Flow Distillation, Reflow-pair
generation, and Reflow. After training, they evaluate the selected dataset(s)
and rebuild the aggregate tables and plots.

Always inspect the plan first; dry-run mode never starts training:

```bash
# Linux
./scripts/run_full_tournament.sh --dry-run
./scripts/run_full_tournament.sh
./scripts/run_full_tournament.sh --dataset cifar10
```

```powershell
# Windows PowerShell
.\scripts\run_full_tournament.ps1 -DryRun
.\scripts\run_full_tournament.ps1
.\scripts\run_full_tournament.ps1 -Dataset cifar10
```

The scripts use the project interpreter under `venv/`, skip an algorithm when
its epoch-100 checkpoint already exists, skip existing Reflow pair artifacts,
and write a timestamped transcript to `results/tournament_run_*.log`. Re-running
after interruption therefore keeps all completed algorithms, although an
algorithm interrupted between saved checkpoints is restarted by this
orchestrator. A shared `results/.lock` prevents another project GPU job from
overlapping a training/evaluation step; Reflow generation uses the same lock.

Use `--dataset celeba` / `-Dataset celeba` for CelebA only. The dataset key is
`celeba` even though the corresponding preset filenames end in `_celeba64.json`.
For a 6 GB GPU, run CIFAR-10 normally, then run CelebA with
`--batch-size 32` / `-BatchSize 32` to cap the two batch-64 presets safely.

Before a multi-day run, benchmark every disposable training flow on the target
GPU without training on the datasets:

```bash
python scripts/benchmark_training_flows.py
```

Measured memory and three-GPU duration estimates are documented in
[`docs/TRAINING_TIME_ESTIMATES.md`](docs/TRAINING_TIME_ESTIMATES.md).

Use the orchestration scripts to run the suite in dependency order:

```bash
bash scripts/train_all.sh                         # Linux/macOS
.\scripts\train_all.ps1                           # Windows PowerShell

bash scripts/train_all.sh --skip-reflow
bash scripts/train_all.sh --only mf
bash scripts/train_all.sh --dataset celeba

# Validate commands and prerequisites without starting training:
bash scripts/train_all.sh --dry-run
.\scripts\train_all.ps1 -DryRun
```

`mf_distill` and `consistency` expect an FM teacher checkpoint. Reflow additionally
needs pairs generated from that checkpoint:

```bash
python scripts/generate_reflow_pairs.py \
  --checkpoint results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt \
  --config config/fm_full.json \
  --n-pairs 50000 \
  --output data/reflow_pairs_cifar10.pt
```

Config presets are in `config/`: `smoke_fast.json` is the tiny CPU test;
`*_full.json` are the main CIFAR-10 runs; and every research algorithm has a
`*_celeba64.json` preset for CelebA at 64x64. Command-line `--epochs` and
`--experiment-name` override the configured values.

For an interactive model menu, use `TRAIN.cmd` on Windows or
`./train_interactive.sh` on Linux. Advanced non-interactive use is also
available:

```bash
python scripts/interactive_train.py --list
python scripts/interactive_train.py --choice cifar10:mf --yes
python scripts/interactive_train.py --choice celeba:fm --dry-run
```

## Outputs, evaluation, and inference

Each run is written to `results/<experiment>_<dataset>/` and includes its resolved
config, logs, samples, metrics, and checkpoints. Every checkpoint is packaged as
a self-contained archive containing weights, config, and metadata. Interactive
training additionally packages the run's configuration, logs, metrics, samples,
and checkpoint archives into `results/exports/<run>.zip`.

```bash
# Evaluate a saved checkpoint
python evaluate.py --algorithm fm \
  --checkpoint results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt \
  --config results/fm_cifar10/config.json --make-plots

# Create sample grids from discovered checkpoints
python scripts/generate_checkpoint_samples.py

# Aggregate every run into a comparison CSV, JSONL, and plots
python scripts/aggregate_results.py

# Try the optional adaptive-NFE or coarse-to-fine Mean Flow samplers
python scripts/sample_mean_flow_extensions.py adaptive --help
python scripts/sample_mean_flow_extensions.py multiscale --help

# Start the local inference UI, then open http://127.0.0.1:8000
python web/inference_server.py
```

Convenience launchers are provided in `scripts/run_train.*`,
`scripts/run_evaluate.*`, and `scripts/run_inference.*`.

Before a long run, verify imports, configs, one dataset batch, and artifact
prerequisites with the same command on Windows or Linux:

```bash
python scripts/verify_workflow.py --dataset cifar10
python scripts/verify_workflow.py --strict-prerequisites
```

Evaluate every available epoch-100 checkpoint with:

```bash
./scripts/evaluate_all.sh --dry-run              # Linux/macOS plan only
.\scripts\evaluate_all.ps1 -DryRun               # Windows plan only
./scripts/evaluate_all.sh --dataset celeba --dry-run
```

## Repository map

```text
bootstrap.py          Cross-platform environment and dataset setup
INIT_ALL.cmd / init_all.sh  Install everything and download both datasets
TRAIN.cmd / train_interactive.sh  Choose, train, and package a model
train.py / evaluate.py Training and evaluation CLIs
algorithms/           Flow Matching, Mean Flow, Consistency, and Reflow methods
models/               Shared SimpleUNet backbone
data/                 CIFAR-10/CelebA loaders and dataset registry
training/             Trainer, AMP, logging, checkpoints, archives
sampling/             Sampling implementations
evaluation/           FID, IS, and plotting support
config/               Experiment presets
scripts/              Setup, batch training, evaluation, and asset utilities
web/                  Local inference server and browser pages
docs/                 Theory notes, implementation log, and thesis material
results/              Generated experiment outputs (gitignored)
```

## Experimental consistency

All algorithms use the same backbone configuration, dataset registry,
preprocessing, optimizer controls, and evaluator. `algorithm_kwargs` is guarded
at runtime so it cannot override shared controls such as epochs, batch size,
backbone, dataset, optimizer, or evaluation settings. Timing and GPU memory are
recorded during training and sampling; FID reference statistics are reused across
comparable runs.

For implementation status, verification commands, and known limitations, see
`docs/IMPLEMENTATION_LOG.md`. Mathematical notes and citations live in
`docs/THEORY_NOTES.md`.
