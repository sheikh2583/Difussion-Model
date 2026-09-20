# DiffusionProject

Research implementation and experiment pipeline for unconditional generative
models on CIFAR-10 (32×32) and CelebA (64×64).

Implemented methods:

- Flow Matching (FM)
- Flow Matching with logit-normal time sampling (FM-LN)
- Mean Flow (MF), including exact-JVP and finite-difference experiment presets
- Mean Flow Distillation (MF-Distill)
- Consistency Models
- Rectified Flow Reflow

The project provides resumable training, numbered checkpoint runs, evaluation
at configurable numbers of function evaluations (NFE), FID and Inception Score,
result aggregation, GIF generation, portable dataset bundles, and a local
results/inference web interface.

## Repository layout

```text
algorithms/             Algorithm implementations
config/                 Reproducible JSON experiment presets
data/                   CIFAR-10 and CelebA loaders/registry
evaluation/             FID, Inception Score, and evaluation orchestration
experiments/            Shared experiment runner
models/                 Shared time-conditioned U-Net backbone
sampling/               Generic sampling and timing
training/               Optimizer, AMP, checkpoint, and training loop
utils/                  Results, provenance, lifecycle, and plotting helpers

scripts/
  *.py                  Platform-independent utilities
  linux/                Linux/macOS shell entrypoints
  windows/              Windows PowerShell and Command Prompt entrypoints

web/                    Results browser and checkpoint inference UI
tests/                  CPU-oriented regression and workflow tests
docs/report/             Submission report source
docs/assets/             Figures referenced by the report

results/                Generated runs and exports (Git-ignored)
training_logs/          Versioned terminal training transcripts by device
```

See [scripts/README.md](scripts/README.md) for the platform command map.

## Quick start

Run commands from the repository root. Paths containing spaces are supported.

### Linux

```bash
chmod +x scripts/linux/*.sh
./scripts/linux/init.sh
./scripts/linux/train.sh
```

The initializer creates `venv/`, installs dependencies, and downloads CIFAR-10
by default. To initialize both datasets:

```bash
./scripts/linux/init.sh --datasets all
```

Useful setup variants:

```bash
./scripts/linux/init.sh --gpu cuda128 --datasets cifar10
./scripts/linux/init.sh --gpu rocm --datasets all
./scripts/linux/init.sh --gpu cpu --datasets none
```

### Windows

From Command Prompt:

```bat
scripts\windows\init.cmd
scripts\windows\train.cmd
```

From PowerShell, the lower-level scripts are also available directly:

```powershell
.\scripts\windows\setup.ps1 -Yes -Datasets all
.\scripts\windows\run_train.ps1 -Algorithm fm -Config config/fm_full.json
```

## Training

### Interactive selection

List available dataset/algorithm combinations without starting training:

```bash
./scripts/linux/train.sh --list
```

Preview a selection:

```bash
./scripts/linux/train.sh --choice cifar10:fm --mode fresh --dry-run
```

Start or resume interactively:

```bash
./scripts/linux/train.sh
```

### Train one configured model

Linux:

```bash
./scripts/linux/run_train.sh \
  --algorithm fm \
  --config config/fm_full.json \
  --mode continue \
  --checkpoint-every 10
```

Windows PowerShell:

```powershell
.\scripts\windows\run_train.ps1 `
  -Algorithm fm `
  -Config config/fm_full.json `
  -Mode continue
```

Lifecycle modes:

- `continue` resumes the latest compatible numbered checkpoint.
- `fresh` preserves prior run metadata under `results/history/` and starts a
  new numbered checkpoint series.
- `--train-only` skips evaluation while retaining normal checkpointing.

Checkpoint compatibility is validated using saved configuration and algorithm
provenance. Existing checkpoints are never silently overwritten.

### Dataset suites

Dependency-aware CIFAR-10 suite:

```bash
./scripts/linux/train_cifar.sh --dry-run
./scripts/linux/train_cifar.sh
```

Run either dataset or both, with a separate terminal log for every model:

```bash
./scripts/linux/train_all_datasets.sh --dataset cifar10 --dry-run
./scripts/linux/train_all_datasets.sh --dataset celeba
./scripts/linux/train_all_datasets.sh --dataset all
```

The equivalent Windows entrypoint is:

```bat
scripts\windows\train_cifar.cmd
```

The suite resolves FM teacher checkpoints before MF-Distill and Consistency,
and resolves or generates Reflow pairs before Reflow training.

### Full tournament

The tournament runner trains, evaluates, and aggregates in dependency order:

```bash
./scripts/linux/run_full_tournament.sh --dataset cifar10 --dry-run
./scripts/linux/run_full_tournament.sh --dataset cifar10
```

```powershell
.\scripts\windows\run_full_tournament.ps1 -Dataset cifar10 -DryRun
.\scripts\windows\run_full_tournament.ps1 -Dataset cifar10
```

Do not launch overlapping GPU training jobs against the same result directory.

## Experiment presets

Canonical presets:

| Dataset | Algorithm | Configuration |
|---|---|---|
| CIFAR-10 | FM | `config/fm_full.json` |
| CIFAR-10 | FM-LN | `config/fm_lognorm_full.json` |
| CIFAR-10 | Mean Flow | `config/mf_v3_exact_jvp_b128.json` |
| CIFAR-10 | MF-Distill | `config/mf_distill_full.json` |
| CIFAR-10 | Consistency | `config/consistency_full.json` |
| CIFAR-10 | Reflow | `config/reflow_full.json` |
| CelebA | FM | `config/fm_celeba64.json` |
| CelebA | FM-LN | `config/fm_lognorm_celeba64.json` |
| CelebA | Mean Flow | `config/mf_celeba64.json` |
| CelebA | MF-Distill | `config/mf_distill_celeba64.json` |
| CelebA | Consistency | `config/consistency_celeba64.json` |
| CelebA | Reflow | `config/reflow_celeba64.json` |

Additional MF presets are retained for controlled diagnostics, exact-JVP versus
finite-difference comparison, and multiscale inference. `config/smoke_fast.json`
is intended for quick workflow checks rather than research results.

## Outputs and naming

The canonical run directory is derived from configuration metadata:

```text
results/<experiment_name>_<dataset>/
```

Example:

```text
results/fm_lognorm_celeba/
  config.json
  run_environment.jsonl
  logs/
  metrics/
    fm_lognorm_celeba.jsonl
  samples/
  checkpoints/
    run_1/
      FlowMatchingLognormAlgorithm_epoch10.pt
      ...
      archive/
```

Checkpoint ZIPs are self-contained and include the checkpoint payload,
configuration, environment metadata, and epoch metadata. Raw generated results,
datasets, checkpoint tensors, GIFs, and ZIP files are intentionally ignored by
Git.

Terminal transcripts are stored under a device-specific directory such as:

```text
training_logs/nvidia-geforce-rtx-3090-24gb/
```

An active log remains at its original open path until its writer finishes; it
should only be reorganized afterward.

## Evaluation

Evaluate one checkpoint:

```bash
./scripts/linux/run_evaluate.sh \
  --algorithm fm \
  --checkpoint results/fm_cifar10/checkpoints/run_1/FlowMatchingAlgorithm_epoch100.pt \
  --config results/fm_cifar10/config.json \
  --make-plots
```

Evaluate all available checkpoints for a dataset:

```bash
./scripts/linux/evaluate_all.sh --dataset cifar10 --epoch 100 --dry-run
./scripts/linux/evaluate_all.sh --dataset cifar10 --epoch 100
```

Windows equivalents live under `scripts/windows/`.

Evaluation is separate from training. It loads existing checkpoints and appends
traceable metric rows containing checkpoint, sample-count, machine, and code
identity metadata.

## Aggregation, plots, GIFs, and bundles

After training has finished, build aggregate CSV/JSONL tables and static plots:

```bash
./scripts/linux/make_summary.sh
```

The summary launcher refuses to write while this project is training unless a
read-only snapshot is explicitly requested:

```bash
./scripts/linux/make_summary.sh --allow-running
```

Generate dataset-specific animations from existing metrics and checkpoint
filenames without loading model tensors:

```bash
./scripts/linux/generate_cifar10_outputs.sh
./scripts/linux/generate_celeba_outputs.sh
```

Generated animations include:

- NFE versus FID
- NFE versus FID versus epoch (3D)
- epoch versus training loss, with checkpoint markers

Create portable checkpoint/results bundles containing configs, provenance,
metrics, aggregates, and published checkpoint archives:

```bash
venv/bin/python scripts/package_dataset_bundles.py --dataset cifar10
venv/bin/python scripts/package_dataset_bundles.py --dataset celeba
```

These reporting and packaging tools do not start, stop, pause, or signal a
training process.

## Results browser and inference UI

Start the local server:

```bash
./scripts/linux/run_inference.sh
```

Open <http://127.0.0.1:8000>.

The web application provides:

- a read-only catalog of discovered CIFAR-10 and CelebA runs;
- checkpoint counts, latest loss, and available evaluation summaries;
- config-derived algorithm and dataset labels;
- dynamic checkpoint and NFE controls;
- sample generation from compatible checkpoints.

The server discovers runs from `results/` and does not rely on hard-coded run
directory names.

## Verification and tests

Run the full CPU test suite:

```bash
python -m pytest -q
```

Validate a planned workflow without training:

```bash
venv/bin/python scripts/verify_workflow.py --dataset cifar10
./scripts/linux/train_all.sh --dataset cifar10 --dry-run
```

The tests cover configuration parsing, lifecycle behavior, checkpoint
provenance, numbered checkpoint layouts, Mean Flow exact JVP behavior,
Consistency sampling, platform entrypoints, aggregation, and web discovery.

## Submission report

The report source is preserved at [docs/report/main.tex](docs/report/main.tex).
Its chapter sources, bibliography, university class/style files, and referenced
figures are all versioned. Report figures live in `docs/assets/` and can be
regenerated where source metrics are available with:

```bash
python scripts/generate_report_figures.py
```

No slide deck is currently present in this repository.

## Additional documentation

| Document | Purpose |
|---|---|
| [START_HERE.md](START_HERE.md) | Short operational entrypoint |
| [scripts/README.md](scripts/README.md) | Linux/Windows command map |
| [docs/THEORY_NOTES.md](docs/THEORY_NOTES.md) | Algorithm derivations and references |
| [docs/IMPLEMENTATION_CHANGES.md](docs/IMPLEMENTATION_CHANGES.md) | Engineering and verification history |
| [docs/TRAINING_TIME_ESTIMATES.md](docs/TRAINING_TIME_ESTIMATES.md) | GPU time and memory estimates |
| [docs/CONSISTENCY_TUNING_NOTES.md](docs/CONSISTENCY_TUNING_NOTES.md) | Consistency training guidance |
| [docs/LINUX_VERIFICATION.md](docs/LINUX_VERIFICATION.md) | Linux environment verification checklist |

## Safety notes

- `results/`, raw datasets, checkpoint tensors, exports, and generated media are
  intentionally Git-ignored.
- Training logs are research evidence; do not delete or relocate an actively
  written log.
- Use `--dry-run` before long training or evaluation workflows.
- Use `continue` for recovery and `fresh` only when a new run is intended.
- Never evaluate, aggregate, or package by terminating an active trainer.
