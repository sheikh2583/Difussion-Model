# Implementation Changes

Last updated: 2026-09-23

This document summarizes the portability, setup, training, evaluation, sampling,
packaging, and documentation work completed for DiffusionProject.

Sections dated by the original Windows bring-up are retained as historical
validation evidence. Current operator behavior is defined by `README.md`,
`RUNBOOK.md`, and the platform guides.

## Relevant commits

| Commit | Description |
|---|---|
| `8b0e29e` | Portable Windows/Linux bootstrap and setup workflow |
| `63fe857` | Workflow validation and cross-platform hardening |
| `6c053f2` | Missing algorithm and evaluation implementations |
| `eaf870a` | Clickable initialization and interactive training workflow |

## Beginner workflow

The repository now has a two-step workflow intended for users with no command-line
experience.

### Windows

1. Double-click `scripts\windows\init.cmd`.
2. Wait for installation and downloads to finish.
3. Double-click `scripts\windows\train.cmd`.
4. Select a model from the numbered menu.
5. Confirm the selection to start training.

### Linux

```bash
chmod +x scripts/linux/init.sh scripts/linux/train.sh
./scripts/linux/init.sh
./scripts/linux/train.sh
```

The initializer and training scripts always resolve the repository directory, so
they do not depend on the user's current working directory.

## Initialization and dependency changes

The initialization workflow now:

- reuses a healthy project virtual environment when one already exists;
- recognizes `venv/Scripts/python.exe` on Windows and `venv/bin/python` on Linux;
- uses a compatible system Python when available;
- installs Python 3.12 through `winget` on Windows when Python is missing;
- installs Python through `apt`, `dnf`, `yum`, `pacman`, `zypper`, or `apk` on
  supported Linux distributions when Python is missing;
- creates or repairs the project virtual environment;
- detects NVIDIA CUDA, AMD ROCm, or CPU-only hardware;
- installs the appropriate PyTorch build;
- installs all project dependencies;
- installs the previously omitted `torch-fidelity` dependency required by FID
  and Inception Score;
- installs `gdown`, which torchvision requires to download CelebA from its
  upstream Google Drive source;
- downloads or verifies the pretrained Inception evaluation weights;
- creates the runtime data and results directories;
- downloads/verifies CIFAR-10 and CelebA when the beginner initializer is used;
- safely reuses completed installation steps when rerun.

The lower-level setup commands remain available when only one dataset is wanted:

```powershell
.\scripts\windows\setup.ps1 -Yes -Datasets cifar10
```

```bash
./scripts/linux/setup.sh --yes --datasets cifar10
```

CelebA is distributed through an upstream Google Drive source. If its automatic
download is quota-limited, the dataset must be downloaded manually as described
in the main README, after which initialization can be rerun safely.

## Interactive training workflow

`scripts/interactive_train.py` provides 13 choices:

- quick smoke test;
- Flow Matching on CIFAR-10 and CelebA;
- Flow Matching with logit-normal time sampling on both datasets;
- Mean Flow on both datasets;
- Mean Flow Distillation on both datasets;
- Consistency Models on both datasets;
- Reflow on both datasets.

It also supports non-interactive automation:

```bash
venv/bin/python scripts/interactive_train.py --list
venv/bin/python scripts/interactive_train.py --choice cifar10:mf --yes
venv/bin/python scripts/interactive_train.py --choice celeba:fm --dry-run
```

### Automatic prerequisites

For Mean Flow Distillation and Consistency Models, the launcher checks the FM
teacher checkpoint. If it is absent, the correct FM experiment is trained first.

For Reflow, the launcher:

1. checks for the dataset-specific FM teacher;
2. trains the FM teacher if it is missing;
3. generates 50,000 Reflow pairs at NFE 50 if the pairs artifact is missing;
4. starts Reflow training.

These prerequisite jobs are real full experiments and can require hours of GPU
time and substantial disk space.

## Checkpoints and run packaging

The trainer saves resumable `.pt` checkpoints at the configured interval and at
the final epoch. Each checkpoint is also packaged as:

```text
results/<run>/checkpoints/archive/<Algorithm>_epoch<N>.zip
```

Each checkpoint ZIP contains:

- `checkpoint.pt`;
- the resolved `config.json`;
- `meta.json` with the algorithm, dataset, epoch, experiment, and timestamp.

Resumption restores model and algorithm-owned state, optimizer, scheduler, AMP
scaler, counters, dataloader shuffle state, and Python/NumPy/PyTorch RNG state.
The tournament, batch scripts, and interactive menu use automatic latest-
checkpoint resumption.

Interactive training additionally invokes `scripts/package_run.py` after the run
completes. It produces:

```text
results/exports/<run>.zip
```

The run bundle contains its configuration, logs, metrics, samples, metadata, and
checkpoint archives. Raw `.pt` files are not duplicated inside the final bundle
because they already exist inside the per-checkpoint ZIP files.

## Dataset-aware output paths

Experiment output paths are now derived consistently as:

```text
results/<experiment_name>_<dataset_name>/
```

The shared configs use algorithm-only experiment names such as `fm`, `mf`, and
`consistency`; the runner appends `cifar10` or `celeba`. This prevents dataset
collisions and aligns teacher checkpoint paths, workflow scripts, evaluation, and
inference discovery.

The inference server now discovers valid run directories and algorithm classes
from saved configurations and checkpoint filenames instead of relying on a fixed
hard-coded list of CIFAR-10 directories.

## Completed algorithm implementations

### Adaptive Mean Flow sampling

`AdaptiveMeanFlowSampler` now implements dynamic per-sample NFE allocation:

- validates minimum/maximum NFE and confidence settings;
- evaluates only active samples;
- compares successive clean-image predictions;
- retires converged samples early;
- forces completion at the maximum NFE;
- records per-sample NFE and average NFE;
- returns unbounded normalized latents when no codec is supplied;
- decodes normalized latents through the frozen codec before clamping RGB.

### Multiscale Mean Flow sampling

`MultiScaleMeanFlowPipeline` is a single-model two-phase latent refinement
wrapper. It generates a 16×16 draft, re-noises it onto the training path, and
refines it from `t_renoise` to zero with the same trained MF model. It never
bilinearly upsamples latents or assumes a nonexistent 32×32 model. Without a
codec it returns unbounded normalized latents; with a codec it decodes before
clamping RGB. `scripts/sample_mean_flow_extensions.py` exposes both inference
wrappers and always loads the frozen codec before writing display images.

### Configurable logit-normal Flow Matching

Flow Matching with logit-normal time sampling now honors:

- `logit_mean`;
- `logit_std`.

Non-positive standard deviations are rejected. Default behavior remains a
standard normal followed by a sigmoid.

## Added CelebA presets

CelebA 64x64 presets now exist for all six research algorithms:

- `config/fm_celeba64.json`;
- `config/fm_lognorm_celeba64.json`;
- `config/mf_celeba64.json`;
- `config/mf_distill_celeba64.json`;
- `config/consistency_celeba64.json`;
- `config/reflow_celeba64.json`.

The distillation and consistency presets reference the CelebA FM teacher. The
Reflow preset references dataset-specific generated pairs.

## Checkpoint compatibility

Checkpoint loading now supports:

- current multi-module checkpoint archives;
- older backbone-only checkpoint layouts;
- legacy Mean Flow `REmbed` state dictionaries whose keys used a bare sequential
  layout instead of the current `net.*` prefix.

This allows existing trained checkpoints to be reused by evaluation, inference,
Reflow generation, and Mean Flow extension sampling.

## Evaluation and metrics

Evaluation now records a sampling row for every NFE value, including:

- total sampling time;
- time per image;
- images per second;
- peak GPU memory;
- FID;
- Inception Score mean and standard deviation.

The `--make-plots` option now produces the standard metric plots rather than
passing an unsupported argument. Plot generation also avoids empty-legend
warnings when a metric type has no records.

Metric filenames consistently use the full run directory name, keeping training,
sampling, and evaluation rows together.

## Result aggregation

`scripts/aggregate_results.py` now:

- discovers experiment JSONL files;
- validates and combines metric records;
- keeps the latest copy of duplicate logical measurements;
- keeps different datasets/runs separate;
- writes combined JSONL output;
- writes a thesis-ready CSV summary;
- generates loss, timing, memory, FID, Inception Score, and quality-versus-time
  plots when the required records exist.

Run it with:

```bash
venv/bin/python scripts/aggregate_results.py
```

## Batch training and evaluation

Windows PowerShell and Linux shell workflows now support CIFAR-10 and CelebA for
all six algorithms. They validate teacher checkpoints and Reflow pairs before
starting dependent jobs and support dry-run planning.

Examples:

```powershell
.\scripts\windows\train_all.ps1 -Dataset cifar10 -DryRun
.\scripts\windows\evaluate_all.ps1 -Dataset cifar10 -DryRun
```

```bash
./scripts/linux/train_all.sh --dataset celeba --dry-run
./scripts/linux/evaluate_all.sh --dataset celeba --dry-run
```

## Workflow verification

`scripts/verify_workflow.py` now checks:

- all algorithm registrations;
- CIFAR-10 and CelebA dataset registration;
- all CIFAR-10 and CelebA configs;
- the coarse Mean Flow config;
- FM teacher checkpoint prerequisites;
- Reflow pair prerequisites;
- dataset batch shape and normalization.

Expected missing large artifacts are reported as blockers without hiding code or
configuration failures.

## Verification performed

The following checks completed successfully on the Windows development machine:

- Python 3.12 project environment discovery;
- CUDA PyTorch discovery on an NVIDIA GPU;
- dependency installation and `pip check`;
- pretrained evaluation asset initialization;
- JSON parsing and `ExperimentConfig` loading for every config;
- Python compilation across the project;
- PowerShell syntax parsing;
- CIFAR-10 batch loading and normalization;
- setup rerun/idempotency behavior;
- clickable Windows training launcher dry run;
- interactive numbered menu input;
- CIFAR-10 and CelebA prerequisite planning;
- successful 1.44 GB CelebA image download plus all official annotations;
- real CelebA batch validation at `(64, 3, 64, 64)` in `[-1, 1]`;
- full generation of 50,000 CIFAR-10 Reflow pairs at NFE 50;
- structural, numeric, and Reflow-loss validation of the 1.144 GB pairs file;
- training and evaluation workflow dry runs;
- adaptive Mean Flow inference with an existing checkpoint;
- multiscale Mean Flow pipeline shape/range validation;
- legacy checkpoint loading;
- result aggregation;
- inference checkpoint auto-discovery;
- creation and inspection of a portable smoke-run ZIP.

The beginner Linux launchers and setup wrapper were converted to POSIX `sh` and
parsed successfully by an actual Linux shell through WSL. Their mounted-path
resolution and missing-environment diagnostic were also exercised. This host
still lacks a general-purpose Linux distribution, so full Linux Python,
dependency, and dataset installation remains necessary on Linux or in CI.

## Large artifacts not stored in Git

The following remain intentionally outside Git:

- CIFAR-10 and CelebA source data;
- pretrained metric weights in the user's cache;
- generated Reflow pairs;
- trained `.pt` checkpoints;
- per-checkpoint archives;
- complete run export ZIP files;
- evaluation caches and generated samples.

At the last verification, CIFAR-10, CelebA, the CIFAR-10 FM teacher checkpoint,
and the full 50,000-pair CIFAR-10 Reflow artifact were available and validated on
the development machine. CelebA teacher checkpoints and CelebA Reflow pairs
remain unavailable because they inherently require CelebA model training.

## Main entry points

| Purpose | Windows | Linux |
|---|---|---|
| Install everything | `scripts\windows\init.cmd` | `./scripts/linux/init.sh` |
| Interactive training | `scripts\windows\train.cmd` | `./scripts/linux/train.sh` |
| Verify workflow | `venv\Scripts\python.exe scripts\verify_workflow.py` | `venv/bin/python scripts/verify_workflow.py` |
| Aggregate results | `venv\Scripts\python.exe scripts\aggregate_results.py` | `venv/bin/python scripts/aggregate_results.py` |
| Inference UI | `venv\Scripts\python.exe web\inference_server.py` | `venv/bin/python web/inference_server.py` |

## 23 September 2026 repository audit

- Python 3.10 is now the consistently documented minimum, and maintained
  launchers invoke the project interpreter without requiring activation.
- Linux and Windows latent suites contain seven serialized jobs. The seventh,
  MF-Hutchinson, is restricted to CelebA latent space and has no pixel preset.
- The interactive picker dispatches latent selections to Bash on POSIX and the
  PowerShell latent launcher on Windows.
- `verify_workflow.py` discovers every experiment JSON recursively, validates
  all three dataset registrations, and reports latent codec/teacher/Reflow
  prerequisites as explicit blockers.
- Per-job latent transcripts and schema-2 sidecars carry Git commit/dirty-diff,
  frozen source/config digests, checkpoint series, and GPU
  index/UUID/name/memory under device-specific log partitions.
- `pytest` and Pillow are direct requirements because the documented
  verification suite and report-figure utility import them directly.
- Static validation completed on Linux: full tests, shell parsing, project-layout
  verification, MF-v3 preflight, seven-job latent dry-run, and dependency
  consistency. No model training, sampling, evaluation, or artifact migration
  was started by this audit.

## 23 September 2026 post-Reflow evidence

### Reflow latent pairs — source identity mismatch (accepted)

The latent Reflow pairs were generated from the completed FM teacher checkpoint
at `results/fm_celeba_latent/checkpoints/run_4/FlowMatchingAlgorithm_epoch100.pt`.
The teacher records source identity
`1a0feaa6767e5710a5b30d5c3d04961ff22f0fa38cac27689e0e7bfe9a2ef52a`,
whereas pair generation and Reflow training record
`600972b2044964c9d003a582166ab09d0f21c91c7794b2b0799a0d8f048be50e`.
The Reflow session ran from commit `490719776ef2` with dirty-diff prefix
`9bfb479bc89f`.

The mismatch was accepted explicitly with the launcher's
`--allow-teacher-source-mismatch` option; it was not silently or automatically
ignored. Structured checkpoint provenance still matched the FM algorithm,
three-channel 16×16 CelebA latent representation, SimpleUNet architecture, and
run variant. The pair manifest records both source identities, the exact FM
checkpoint digest, and `source_identity_mismatch_accepted: true`. Source changes
since FM training did not change `algorithms/flow_matching.py` or the shared
backbone recorded by the two source manifests.

Reflow training reached epoch 100 with loss `0.006143` and wrote
`ReflowAlgorithm_epoch100.pt`. The session nevertheless recorded
`exit_status=1`: creation of the epoch-100 ZIP archive failed with `ENOSPC`
(`No space left on device`). The checkpoint itself remained valid and was
evaluated explicitly after storage was recovered. Epoch-100 results are:

| NFE | FID | IS mean | IS std |
|---:|---:|---:|---:|
| 1 | 300.8571 | 2.4908 | 0.0685 |
| 2 | 300.7416 | 2.4962 | 0.0829 |
| 5 | 300.2236 | 2.5218 | 0.0778 |
| 10 | 299.9267 | 2.5167 | 0.0788 |
| 20 | 299.7504 | 2.5080 | 0.0799 |

### MF latent — divergence after epoch 90, epoch-80 checkpoint used

MF latent training completed 100 epochs in `run_2`. Loss remained approximately
`1.69–1.74` through epoch 94, then rose from `1.755604` at epoch 95 to
`2.163722` at epoch 100. The increase is strictly monotonic from epoch 94 through
epoch 100, not throughout the entire epoch 91–100 interval.

The epoch-80 checkpoint (`loss=1.725121`) is designated for the canonical MF
latent evaluation, while the epoch-100 checkpoint is retained as historical
evidence. The checkpoint contains the historical `REmbed` conditioner
(`1→64→3`) rather than the later `RCond` architecture. Evaluation therefore
uses a shape-detected compatibility path that reproduces the original
r-dependent image bias; it does not reinterpret the weights as `RCond`.

Canonical epoch-80 results are:

| NFE | FID | IS mean | IS std |
|---:|---:|---:|---:|
| 1 | 294.7134 | 2.3358 | 0.0551 |
| 2 | 297.2778 | 2.4037 | 0.0476 |
| 5 | 300.4877 | 2.3924 | 0.0662 |
| 10 | 299.8622 | 2.3838 | 0.0546 |
| 20 | 299.4497 | 2.3886 | 0.0520 |

No epoch-100 metric is relabelled as an epoch-80 result. A likely cause of the
late loss rise is an interaction between the JVP target estimator and the
latent distribution; the separate Hutchinson-CV experiment is the designed
diagnostic for this regime.
