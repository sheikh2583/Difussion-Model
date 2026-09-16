# Diffusion Models: Comparative Study

This repository is a runnable research scaffold for comparing six image-generation
algorithms with a shared backbone, data pipeline, sampler, and evaluation stack.
It supports CIFAR-10 (32x32) and CelebA (64x64), persists resumable checkpoints,
and includes a small browser UI for inspecting saved models.

## One-command setup

The cross-platform entry point is `bootstrap.py`. It uses only the Python standard
library, so it works on Windows, Linux, and macOS. It creates `venv/`, detects an
NVIDIA/ROCm/CPU PyTorch target, installs dependencies, creates runtime folders, and
downloads the requested dataset.

```bash
git clone <repo-url>
cd DiffusionProject
python3 bootstrap.py --yes                 # Linux/macOS
```

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

The bootstrap script requires Python 3.9+ (install it first if it is not already
available). Pass `--gpu cpu`, `--gpu cuda118`, `--gpu cuda121`, `--gpu cuda128`,
or `--gpu rocm` to override hardware detection. `--skip-torch`
is for a pre-populated project virtual environment only.

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

Use the orchestration scripts to run the suite in dependency order:

```bash
bash scripts/train_all.sh                         # Linux/macOS
.\scripts\train_all.ps1                           # Windows PowerShell

bash scripts/train_all.sh --skip-reflow
bash scripts/train_all.sh --only mf

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
`*_full.json` are the main CIFAR-10 runs; `fm_celeba64.json` and
`mf_celeba64.json` select CelebA at 64x64. Command-line `--epochs` and
`--experiment-name` override the configured values.

## Outputs, evaluation, and inference

Each run is written to `results/<experiment>_<dataset>/` and includes its resolved
config, logs, samples, metrics, and checkpoints. Every checkpoint is also packaged
as a self-contained archive containing weights, config, and metadata.

```bash
# Evaluate a saved checkpoint
python evaluate.py --algorithm fm \
  --checkpoint results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt \
  --config results/fm_cifar10/config.json --make-plots

# Create sample grids from discovered checkpoints
python scripts/generate_checkpoint_samples.py

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
```

## Repository map

```text
bootstrap.py          Cross-platform environment and dataset setup
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
