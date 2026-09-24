# From Pixel Flow Matching to Frozen-Latent Face Generation

## A chronological research record

This repository documents an experimental study of unconditional
continuous-time generative modelling: first establishing pixel-space evidence
on CIFAR-10, then scaling the comparison to CelebA, and finally migrating the
same objectives to a frozen face-specific latent representation. The code is
the executable record of that study; failed codec candidates, corrective
implementations, intermediate probes, logs, checkpoints, and accepted
limitations are retained because they explain the final design.

The central question is not simply whether each model runs. It is whether
Flow Matching variants and fast-sampling objectives can be compared fairly
under a shared backbone, and whether latent-space training reduces computational
cost without allowing the frozen codec to dominate the measured image quality.

The implemented methods are:

- Flow Matching (FM)
- Flow Matching with logit-normal time sampling (FM-LN)
- Mean Flow (MF), with exact-JVP and finite-difference variants
- Mean Flow with randomized Hutchinson VJP estimation (latent-only diagnostic)
- Mean Flow Distillation (MF-Distill)
- Consistency Models
- Rectified Flow Reflow

The study contains three sequential experiment families:

| Family | Model state | Generative input | Purpose |
|---|---|---:|---|
| CIFAR-10 pixel | RGB | `3×32×32` | Controlled algorithm comparison |
| CelebA pixel | RGB | `3×64×64` | Higher-resolution face baseline |
| CelebA latent | normalized VQ-f4 code | `3×16×16` | Lower-activation latent comparison |

> **Current latent-study limitation:** the selected pretrained VQ-f4 codec did
> not meet this project's strict reconstruction FID and PSNR gates. It was
> admitted by an explicit, recorded operator override after two alternatives
> also failed. Latent results must therefore be interpreted under a measurable
> reconstruction ceiling; the override does not turn the failed gate into a
> pass.

## Thesis focus and contributions

The work contributes one controlled experimental chain rather than a collection
of unrelated model implementations:

1. A common time-conditioned U-Net and evaluation protocol for FM, FM-LN,
   Mean Flow, MF-Distill, Consistency, and Reflow.
2. A measured comparison of quality against sampling NFE, not only final image
   quality at one solver budget.
3. An engineering study of exact-JVP versus finite-difference Mean Flow,
   a randomized Hutchinson-VJP latent diagnostic, teacher-dependent objectives,
   EMA correctness, mixed precision, and Reflow pair generation.
4. A three-candidate codec investigation in which two candidates were rejected
   and the selected candidate was accepted only under an auditable quality
   override.
5. A representation-aware evaluation boundary: latent states remain unbounded,
   are decoded in batches, and are compared with raw CelebA validation pixels.
6. Reproducibility evidence linking runs to configurations, backbone identity,
   lifecycle, machine, checkpoints, and pixel/latent log classification.

## Chronological experimental record

This sequence is the recommended way to read the repository. Dates refer to
the retained run evidence, not to publication milestones.

| Date / phase | Question or obstacle | Work performed | Evidence and decision |
|---|---|---|---|
| Before 18 Sep 2026 — experimental design | How can six objectives be compared without architecture drift? | Established the shared `SimpleUNet`, common configuration schema, dataset registry, trainer, sampler, and FID/IS evaluator. | Algorithm-specific code controls the objective; shared controls remain in the experiment config. |
| 18 Sep 2026 — first CIFAR runs | Do baseline FM and Mean Flow workflows train and resume correctly? | Ran CIFAR-10 pixel experiments, checkpoint probes, and finite-difference Mean Flow diagnostics. | Exposed the need for numbered runs, lifecycle control, and exact source/config provenance. |
| 19 Sep 2026 — controlled CIFAR comparison | Which methods improve few-step generation, and which implementation defects distort the comparison? | Completed FM, FM-LN, exact-JVP MF, MF-Distill, Consistency, and Reflow runs; corrected AMP teacher/student dtype handling and Consistency EMA ordering/persistence. | FM-LN improved FID@50 from `43.427` to `35.400`; Reflow produced `53.573` at NFE 1 versus FM's `380.620`, while FM remained better at NFE 20. |
| 19–21 Sep 2026 — CelebA pixel scaling | Do the FM conclusions survive at 64×64 face resolution? | Trained CelebA pixel FM and FM-LN for 100 epochs with the same 8,947,459-parameter backbone. | FM-LN improved FID@50 from `16.779` to `15.304`; both runs took about 16.1 hours and peaked near 5.23 GiB. |
| 21 Sep 2026 — scratch latent attempt | Can a locally trained factor-4 KL-VAE provide an adequate face representation? | Trained the scratch codec for 60 epochs and evaluated 5,000 validation reconstructions. | PSNR passed, but final rFID `14.2381` failed the `<5` gate. The run was rejected rather than silently reused. |
| 21 Sep 2026 — generic pretrained latent attempt | Can a frozen Stable Diffusion VAE remove scratch-codec training cost? | Evaluated `stabilityai/sd-vae-ft-mse` at factor 8. | rFID `16.1529` and PSNR `22.3193 dB` failed; its `4×8×8` state also forced a `1×1` U-Net bottleneck. Rejected. |
| 21 Sep 2026 — face-specific codec selection | Does domain-specific pretraining improve the latent boundary? | Downloaded, pinned, validated, and froze statistics for `CompVis/ldm-celebahq-256` VQ-f4. | rFID improved to `10.0613` and PSNR to `27.3798 dB`, still below the strict gates. Structural and spread checks passed, so the user accepted an explicit recorded override. |
| 21–22 Sep 2026 — latent dataset construction | Can codec identity and normalization be made reproducible? | Cached deterministic quantized train/validation latents using frozen channel statistics and a content-addressed manifest. | Published 162,752 train and 19,867 validation tensors of shape `3×16×16`; ambiguity and hash mismatches are hard failures. |
| 21–22 Sep 2026 — first latent generative runs | Does the smaller spatial state reduce resource use while preserving algorithm trends? | Completed latent FM and FM-LN through epoch 100 using the frozen VQ-f4 cache. Added bounded decoding, separate decoder timing, unbounded latent sampling, and isolated latent result directories. | Both populated latent checkpoint series are complete and provenance-compatible. MF, MF-Hutchinson, MF-Distill, Consistency, and Reflow remain outside the completed comparison until final compatible evaluations exist; no partial latent metric is treated as final. |
| 22 Sep 2026 — post-training isolation hardening | Can the remaining unattended suite run without overlapping GPU work or losing source identity? | Added one ownership-checked, nestable GPU lock; froze training-relevant source/config inputs at suite startup; added source verification between jobs; made completed-run skipping provenance-aware; and separated explicit evaluation from continuation. | CPU/static validation passed. The remaining latent jobs can continue serially without rerunning completed FM/FM-LN or implicitly reevaluating them. |
| 22 Sep 2026 — evidence normalization and audit | Can historical logs and checkpoints be identified consistently without modifying measured output? | Migrated central transcripts into `pixel/`, `latent/`, and `codec/`; normalized all 48 sidecars to schema 2; recorded RTX 3090 24 GB identity and transcript digests; rebuilt the catalog; and audited numbered checkpoint series and ZIP archives. | All migrated transcript digests remained unchanged. Fourteen populated diffusion checkpoint series passed filename/archive/payload/provenance checks; historical exceptions are documented rather than silently rewritten. |

### Quantitative gains already supported by completed runs

| Comparison | Observed change | Interpretation |
|---|---:|---|
| CIFAR FM-LN versus FM at NFE 50 | FID `43.427 → 35.400` (`18.5%` lower) | Logit-normal time sampling improved the strongest completed CIFAR baseline. |
| CelebA pixel FM-LN versus FM at NFE 50 | FID `16.779 → 15.304` (`8.8%` lower) | The time-sampling gain transferred to 64×64 faces. |
| CIFAR Reflow versus FM at NFE 1 | FID `380.620 → 53.573` (`85.9%` lower) | Rectification strongly improved the single-evaluation regime. |
| CIFAR FM versus Reflow at NFE 20 | FID `45.467` versus `50.020` | The few-step gain did not make Reflow universally better; FM retained higher quality at the larger budget. |
| CelebA pixel to latent state size | `3×64×64 → 3×16×16` (`16×` fewer spatial values) | Expected activation/throughput gain; it does not reduce parameter count or erase codec error. |

These percentages describe the retained 5,000-sample evaluations. They should
not be generalized beyond the recorded configurations, and the latent study is
deliberately excluded from the completed-results table until all six methods
have compatible final evaluations. MF-Hutchinson is an additional diagnostic
and is not silently folded into that canonical six-method comparison.

## Research questions

The repository is designed to answer four practical questions under a shared
backbone and evaluation protocol:

1. How does ordinary Flow Matching compare with logit-normal time sampling?
2. How much sampling speed can Mean Flow, distillation, consistency training,
   or Reflow gain, and what quality is lost at low NFE?
3. Does moving CelebA generation from 64×64 RGB pixels to 16×16 learned codes
   reduce training cost without making the codec the dominant quality limit?
4. Can every result be traced to a configuration, source identity, dataset,
   checkpoint, lifecycle mode, and machine rather than only to a model file?

FID, Inception Score, training time, peak memory, and quality versus number of
function evaluations (NFE) are the main comparison axes. A fair comparison uses
the same dataset split, sample count, seed policy, reference statistics, and
NFE values.

## Detailed decision and failure record

For the concise stage-by-stage reconstruction contract—completed evidence,
partial experiments, implementation-only extensions, and exact audit
commands—see [`docs/THESIS_PROGRESSION.md`](docs/THESIS_PROGRESSION.md).

This codebase intentionally keeps unsuccessful paths visible. They explain why
the current design exists and prevent a future user from repeating the same
experiments without context.

### Stage 1 — Pixel-space baselines

The first complete experiments trained the shared time-conditioned
`SimpleUNet` directly on CIFAR-10 and 64×64 CelebA pixels. FM established the
teacher and quality baseline. FM-LN changed the time-sampling distribution,
while MF, MF-Distill, Consistency, and Reflow explored fewer-step generation.

The same experiment runner, result layout, checkpoint format, and evaluator are
used across algorithms. This keeps an objective change from silently becoming
an infrastructure change.

### Stage 2 — Faster-path methods and diagnostics

Mean Flow was tested with both finite-difference and exact Jacobian-vector
products. Exact JVP is a useful controlled diagnostic, but it costs more memory
and compute. MF-Distill and Consistency introduce a trained FM teacher; Reflow
introduces a generated pair dataset. These are real dependencies, not merely a
recommended execution order.

Several engineering corrections were needed during this stage:

- MF-Distill mixed-precision teacher/student dtype handling was corrected.
- Consistency EMA updates were moved after optimizer updates and persisted in
  checkpoints, with legacy-checkpoint fallback behavior.
- Teacher models are loaded lazily instead of consuming memory for unrelated
  jobs.
- Reflow pair storage is preallocated and generated in chunks; its default
  batch size was reduced to avoid unnecessary memory pressure.
- Evaluation batches generated images, moves completed images to CPU, reuses
  real FID statistics across NFE values, and avoids duplicate final evaluation.

### Stage 3 — First latent attempt: scratch KL-VAE

A factor-4 scratch KL-VAE produced `4×16×16` codes and was trained for 60
epochs. It reached acceptable PSNR but failed the reconstruction-FID gate:

| Measurement | Result | Project gate | Decision |
|---|---:|---:|---|
| Final reconstruction FID | `14.2381` | `< 5` | Failed |
| Best earlier reconstruction FID | about `11.3973` | `< 5` | Failed |
| Final PSNR | `33.7268 dB` | `> 30 dB` | Passed |
| Latent spread | Passed | non-collapsed | Passed |

The scratch run is retained as historical evidence. It is not resumed and is
not an automatic fallback for the accepted latent experiment.

### Stage 4 — Second latent attempt: frozen Stable Diffusion VAE

The frozen `stabilityai/sd-vae-ft-mse` factor-8 VAE produced `4×8×8` codes.
Its reconstruction metrics were worse for this 64×64 face setting:

| Measurement | Result | Project gate | Decision |
|---|---:|---:|---|
| Reconstruction FID | `16.1529` | `< 5` | Failed |
| PSNR | `22.3193 dB` | `> 30 dB` | Failed |
| Minimum channel standard deviation | `0.6434` | `> 0.1` | Passed |

It was rejected. Its `4×8×8` representation would also have driven the
four-stage U-Net to a `1×1` bottleneck, making it a poor architectural match for
the existing comparison.

### Stage 5 — Selected migration: frozen CelebA-HQ VQ-f4

The selected codec is the face-specific
`CompVis/ldm-celebahq-256` VQ-f4 model. It maps 64×64 RGB images to deterministic,
codebook-quantized `3×16×16` states. The accepted artifact records its immutable
source revision, weights digest, normalization statistics, validation report,
and the explicit quality-gate override.

| Measurement | Recorded value |
|---|---:|
| Reconstruction FID | `10.0613` |
| PSNR | `27.3798 dB` |
| Minimum native latent std | `0.8564` |
| Training latent mean | `[0.163503, -0.053753, 0.120234]` |
| Training latent std | `[0.856374, 1.171198, 1.333852]` |
| Cached training examples | `162,752` |
| Cached validation examples | `19,867` |

The project gates are reconstruction FID `< 5`, PSNR `> 30 dB`, and minimum
channel standard deviation `> 0.1`. The VQ-f4 model passed the structural and
latent-spread checks but failed the two reconstruction-quality gates. The
accepted checkpoint consequently retains `quality_gate_passed: false` and the
operator's reason for continuing.

### Stage 6 — What “frozen pretrained migration” means

The migration did **not** replace the generative U-Net with a pretrained U-Net.
Only the image boundary became pretrained and frozen:

```text
training image [3×64×64]
        │
        ▼
frozen CelebA-HQ VQ-f4 encoder + quantizer
        │ native latent [3×16×16]
        ▼
frozen per-channel normalization
        │ normalized latent [3×16×16]
        ▼
trainable algorithm-specific SimpleUNet

generation noise [3×16×16]
        │
        ▼
trained SimpleUNet / numerical sampler
        │ normalized generated latent (unbounded)
        ▼
denormalization + frozen VQ-f4 decoder
        │
        ▼
generated RGB image [3×64×64]
```

Every FM, FM-LN, MF, MF-Hutchinson, MF-Distill, Consistency, and Reflow run
still trains its own generative U-Net from the configured initialization. The
codec is frozen so all seven latent jobs see the same representation and cannot
improve or degrade it during training. MF-Hutchinson remains a diagnostic
outside the canonical six-method result table.

The completed latent suite did not retain the initial parameter-matched
proposal. Its frozen configs use `base_channels=128` and multipliers
`[1,2,2]`, producing 24,026,627 backbone parameters and a
`16→8→4` spatial path. CelebA pixel uses `base_channels=64`, multipliers
`[1,2,2,2]`, 8,947,459 parameters, and a `64→32→16→8` path. The latent state
has 16× fewer spatial values at the boundary, but the trainable backbone has
about 2.69× more parameters. Pixel-versus-latent comparisons must disclose
both changes rather than attributing every difference to representation size.
Latent samples are unbounded normalized states and must never be clamped to
`[-1,1]` before codec decoding.

### Stage 7 — Isolation and evidence normalization before continuation

After latent FM and FM-LN reached epoch 100 and all training processes exited,
the operational layer was hardened before launching the remaining methods. An
ownership-checked GPU lock rejects competing workflows, supports nested suite
calls through an inherited ownership token, and requires explicit stale
recovery. Each suite freezes the live training-relevant source and selected
configurations, then verifies them before every later GPU job.

Historical transcript bytes were not edited. Their sidecars were normalized to
schema 2 with representation, dataset, algorithm, canonical machine/GPU
identity, byte count, and SHA-256. The confirmed lab identity is
`linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb`, corresponding to an NVIDIA
GeForce RTX 3090 with 24 GB of memory. The catalog contains 48 fully identified
rows. Central transcripts occupy explicit `pixel/`, `latent/`, or `codec/`
directories, and migration manifests retain matching pre/post transcript
digests.

The checkpoint audit found 14 populated diffusion series with consistent
algorithm classes, epoch metadata, provenance version, and matching ZIP
archives. Two historical lifecycle details remain visible: CelebA pixel FM-LN
`run_1` stopped at epoch 90 before the completed `run_2`, and MF-Distill has an
empty reserved `run_1` followed by the completed `run_2`. Scratch-codec
checkpoints retain their older flat codec-specific layout so historical resume
paths are not rewritten.

## Algorithm and dependency map

| Method | Training signal | Sampling intent | Dependency |
|---|---|---|---|
| FM | Conditional velocity along a probability path | Strong multi-step baseline | None |
| FM-LN | FM with logit-normal time sampling | Reweight difficult time regions | None |
| MF | Average velocity over an interval | Few-step generation | None |
| MF-Hutchinson | Mean Flow with randomized reverse-mode VJP probes | Latent diagnostic | None; latent space only |
| MF-Distill | Student guided by completed FM teacher | Compress teacher behavior | FM epoch 100 |
| Consistency | Self-consistent states with EMA target and FM teacher | One/few-step generation | FM epoch 100 |
| Reflow | Rectification on teacher-generated endpoint pairs | Straighter trajectories | FM epoch 100 and Reflow pairs |

On one GPU, run jobs serially. The valid latent order is:

```text
FM ───────────────┬──► MF-Distill
                  ├──► Consistency
                  └──► generate Reflow pairs ───► Reflow

FM-LN ───────────────► independent
Mean Flow ───────────► independent
MF-Hutchinson ───────► independent, latent only
```

## Reproducing and auditing the study

The remainder of this document is the operational appendix: it explains how to
reproduce a particular stage, inspect its evidence, and avoid mixing pixel and
latent experiments. Reading the chronology and limitations above first is
essential for interpreting the commands below.

### Repository map

```text
algorithms/              Training objectives and algorithm-specific sampling
codec/                   Codec API, pretrained/scratch codecs, validation/cache tools
config/                  Versioned JSON experiment presets
data/                    Pixel datasets, latent dataset, and registry
evaluation/              FID, Inception Score, and evaluation orchestration
experiments/             Shared experiment runner
models/                  Current U-Net plus frozen legacy CIFAR backbone
sampling/                Generic samplers and backbone/decoder timing
training/                Optimizer, AMP, checkpoint, and training loop
utils/                   Lifecycle, provenance, result, and plotting helpers
scripts/
  *.py                   Platform-independent utilities
  linux/                 Bash entrypoints
  windows/               PowerShell and Command Prompt entrypoints
web/                     Local result browser and checkpoint inference UI
tests/                   CPU-oriented regression and workflow tests
docs/report/              Thesis/report source
docs/assets/              Report figures
research_legacy/          Git-exact historical source states tied to old checkpoints
results/                  Generated runs and codec reports; mostly Git-ignored
training_logs/            Timestamped terminal transcripts grouped by device
```

`research_legacy/` preserves source evidence associated with historical
checkpoints. It should not be treated as the current implementation. A dirty
historical run can establish a committed base state, but it cannot reconstruct
uncommitted runtime edits that were never recorded.

## Prerequisites

- Python 3.10 or newer
- Git for obtaining the repository; Git is not part of the training lifecycle
- NVIDIA CUDA, AMD ROCm, or CPU-only PyTorch
- Enough disk space for CelebA, cached latents, checkpoints, samples, and logs
- A CUDA GPU is strongly recommended for full CelebA and codec workflows

Python 3.10 is the minimum because the implementation uses modern union type
syntax such as `Path | None`. The setup scripts install PyTorch first and then
install `requirements.txt`. `accelerate` is a declared dependency because
Diffusers uses it for low-CPU-memory pretrained-model loading; if that warning
appears, the environment is incomplete or the wrong Python interpreter is
being used.

Run all commands from the repository root. Paths containing spaces are
supported by the maintained launchers.

## Platform-specific initialization

### Linux: guided initialization

```bash
chmod +x scripts/linux/*.sh
./scripts/linux/init.sh
```

The default initializes `venv/`, installs dependencies, detects the available
compute backend, and prepares CIFAR-10. To prepare both datasets:

```bash
./scripts/linux/init.sh --datasets all
```

Explicit variants are useful on headless machines:

```bash
./scripts/linux/init.sh --gpu cuda128 --datasets all
./scripts/linux/init.sh --gpu cuda121 --datasets celeba
./scripts/linux/init.sh --gpu cuda118 --datasets cifar10
./scripts/linux/init.sh --gpu rocm --datasets all
./scripts/linux/init.sh --gpu cpu --datasets none
```

The lower-level noninteractive entrypoint is:

```bash
./scripts/linux/setup.sh --yes --gpu cuda128 --datasets all
```

### Windows: Command Prompt

```bat
INIT_ALL.cmd
```

`INIT_ALL.cmd` is the complete one-shot initializer. It installs the full
training environment, downloads CIFAR-10 and CelebA, checks CUDA, and prepares
the accepted CelebA codec and latent cache. Codec quality failures still require
an explicit, documented override:

```bat
INIT_ALL.cmd -AcceptQualityFailure -AcceptanceReason "Approved for the documented VQ-f4 latent experiment"
```

Use `-DryRun` to preview the stages, or `-SkipLatentAssets` on a machine that
only needs the pixel-space workflows. For the smaller CIFAR-10-only setup, use:

```bat
scripts\windows\init.cmd
scripts\windows\train.cmd
```

Both Command Prompt entrypoints delegate to maintained PowerShell setup scripts
while avoiding manual execution-policy and quoting mistakes.

### Windows: PowerShell

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\setup.ps1 -Yes -Gpu cuda128 -Datasets all
```

Other accepted backend choices mirror the setup script's help. On a machine
without a suitable Python, the Windows setup can use `winget` to install a
supported interpreter.

### Verify an existing environment

Linux:

```bash
venv/bin/python -c "import torch, torchvision, diffusers, accelerate; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('accelerate', accelerate.__version__)"
venv/bin/python scripts/verify_project_layout.py
```

Windows PowerShell:

```powershell
.\venv\Scripts\python.exe -c "import torch, torchvision, diffusers, accelerate; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('accelerate', accelerate.__version__)"
.\venv\Scripts\python.exe scripts\verify_project_layout.py
```

If dependencies are stale, update the active environment from the declared
requirements rather than installing packages into a different global Python:

```bash
venv/bin/pip install -r requirements.txt
```

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Datasets and generated artifacts

CIFAR-10 can normally be downloaded automatically. CelebA downloads may be
blocked by Google Drive quota or authentication; if so, place the official
files in the layout expected under `data/raw/` and rerun setup. Dataset files,
latent caches, raw checkpoints, generated images, and large exports are not
source-controlled.

The latent pipeline adds two large artifact families:

- `data/pretrained/ldm-celebahq-256/`: downloaded frozen codec source
- `data/latent_cache/latents_<content-hash>/`: verified normalized tensors and
  manifest

Cache identity includes codec content, split, preprocessing, posterior mode,
and normalization schema. It does not rely on an absolute filename alone, and
multiple codec caches may coexist safely. Do not rename cache internals or
hand-edit their manifest.

## Pixel-space training

### Interactive launcher

List choices or preview a command without training:

```bash
./scripts/linux/train.sh --list
./scripts/linux/train.sh --choice cifar10:fm --mode fresh --dry-run
```

Start the menu on Linux:

```bash
./scripts/linux/train.sh
```

Or use the Windows menu:

```bat
scripts\windows\train.cmd
```

The interactive menus target the established pixel experiments. Use the
dedicated procedure below for CelebA latent work.

### Train one configuration

Linux:

```bash
./scripts/linux/run_train.sh \
  --algorithm fm \
  --config config/fm_full.json \
  --mode fresh \
  --checkpoint-every 10
```

Windows PowerShell:

```powershell
.\scripts\windows\run_train.ps1 `
  -Algorithm fm `
  -Config config/fm_full.json `
  -Mode fresh `
  -CheckpointEvery 10
```

Use `--mode continue`/`-Mode continue` after an interruption. Use `fresh` only
when a separate numbered run is intended. A fresh lifecycle preserves earlier
metadata rather than silently replacing it.

### Pixel suites

```bash
./scripts/linux/train_cifar.sh --dry-run
./scripts/linux/train_cifar.sh

./scripts/linux/train_all_datasets.sh --dataset celeba --dry-run
./scripts/linux/train_all_datasets.sh --dataset all
```

Windows provides the CIFAR entrypoint:

```bat
scripts\windows\train_cifar.cmd
```

### Choose the CIFAR backbone

The completed CIFAR-10 results used the architecture now frozen in
`models/legacy_cifar_backbone.py`. At the time it was frozen, it was
state-dict- and output-equivalent to `simple_unet`; keeping it as an independent
module prevents later backbone improvements from changing what “rerun CIFAR”
means.

Use `current` for the normal presets, or `legacy` for an isolated reproduction
suite. Always preview before starting the legacy suite:

```bash
./scripts/linux/train_cifar.sh --cifar-backbone legacy --mode fresh --dry-run
./scripts/linux/train_cifar.sh --cifar-backbone legacy --mode fresh
```

Windows Command Prompt:

```bat
scripts\windows\train_cifar.cmd -CifarBackbone legacy -Mode fresh -DryRun
scripts\windows\train_cifar.cmd -CifarBackbone legacy -Mode fresh
```

Windows PowerShell:

```powershell
.\scripts\windows\train_all.ps1 -Dataset cifar10 -CifarBackbone legacy -Mode fresh -DryRun
.\scripts\windows\train_all.ps1 -Dataset cifar10 -CifarBackbone legacy -Mode fresh
```

For one model, select its explicit preset:

```bash
./scripts/linux/run_train.sh --algorithm fm \
  --config config/cifar_legacy/fm.json --mode fresh
```

The six legacy presets live in `config/cifar_legacy/`. They write to separate
`*_legacy_backbone_cifar10` directories, use their own FM teacher path, and use
`data/reflow_pairs_cifar10_legacy_backbone.pt`. Existing `config/*_full.json`
files and completed checkpoint provenance remain unchanged.

The full tournament runner trains, evaluates, and aggregates in dependency
order:

```bash
./scripts/linux/run_full_tournament.sh --dataset cifar10 --dry-run
./scripts/linux/run_full_tournament.sh --dataset cifar10
```

```powershell
.\scripts\windows\run_full_tournament.ps1 -Dataset cifar10 -DryRun
.\scripts\windows\run_full_tournament.ps1 -Dataset cifar10
```

## CelebA latent workflow

The authoritative checkpoint/cache procedure is also available in
[RUNBOOK.md](RUNBOOK.md). Stop on the first error; never substitute one of the
rejected codecs.

### Linux: prepare and validate the codec

Preview the exact commands:

```bash
./scripts/linux/prepare_pretrained_codec.sh --dry-run \
  --accept-quality-failure \
  --acceptance-reason "Proceeding with the measured VQ-f4 reconstruction ceiling"
```

Download, validate, freeze statistics, and record the known override:

```bash
./scripts/linux/prepare_pretrained_codec.sh \
  --accept-quality-failure \
  --acceptance-reason "Proceeding with the measured VQ-f4 reconstruction ceiling"
```

The command produces:

```text
results/codecs/celeba_vq_f4/accepted_codec.pt
results/codecs/celeba_vq_f4/validation_report.json
results/codecs/celeba_vq_f4/reconstruction_grid.png
```

Open the grid and inspect identity, eyes, hair, edges, and color. The override
is only for the already measured reconstruction shortfall. Structural errors,
non-finite values, or collapsed latent channels remain hard failures.

### Windows: prepare and validate the codec

There is no wrapper-specific behavior in the codec Python tools, so PowerShell
can invoke the same stages directly:

```powershell
.\venv\Scripts\python.exe -m codec.download_pretrained_vq `
  --repo-id CompVis/ldm-celebahq-256 `
  --revision main `
  --output-dir .\data\pretrained\ldm-celebahq-256

.\venv\Scripts\python.exe codec\validate_codec.py `
  --codec-source CompVis/ldm-celebahq-256 `
  --codec-source-path .\data\pretrained\ldm-celebahq-256\vqvae `
  --codec-source-revision auto `
  --celeba-root .\data\raw `
  --output-dir .\results\codecs\celeba_vq_f4 `
  --batch-size 32 `
  --n-val-images 5000 `
  --num-workers 4 `
  --device cuda `
  --accept-quality-failure `
  --acceptance-reason "Proceeding with the measured VQ-f4 reconstruction ceiling"
```

### Cache normalized latents

Linux:

```bash
venv/bin/python codec/cache_latents.py \
  --codec-path results/codecs/celeba_vq_f4/accepted_codec.pt \
  --celeba-root data/raw \
  --output-dir data/latent_cache \
  --split both \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda
```

Windows PowerShell:

```powershell
.\venv\Scripts\python.exe codec\cache_latents.py `
  --codec-path .\results\codecs\celeba_vq_f4\accepted_codec.pt `
  --celeba-root .\data\raw `
  --output-dir .\data\latent_cache `
  --split both `
  --batch-size 64 `
  --num-workers 4 `
  --device cuda
```

Record the exact printed `latents_<hash>` directory. Rerunning the same command
verifies an existing cache and should report that there is nothing to do.

### Smoke-test the latent boundary

The static contract test is CPU-safe:

```bash
venv/bin/python scripts/smoke_latent.py --mode static
```

Then run the full codec/cache boundary check with the actual cache directory:

```bash
CACHE_DIR=data/latent_cache/latents_<hash>
venv/bin/python scripts/smoke_latent.py --mode full \
  --codec-path results/codecs/celeba_vq_f4/accepted_codec.pt \
  --latent-cache-dir "$CACHE_DIR" \
  --celeba-root data/raw \
  --device cuda
```

PowerShell equivalent:

```powershell
$CacheDir = ".\data\latent_cache\latents_<hash>"
.\venv\Scripts\python.exe scripts\smoke_latent.py --mode full `
  --codec-path .\results\codecs\celeba_vq_f4\accepted_codec.pt `
  --latent-cache-dir $CacheDir `
  --celeba-root .\data\raw `
  --device cuda
```

Proceed only after `ALL FULL SMOKE CHECKS PASSED`.

### Linux: train the complete latent suite

Always preview first:

```bash
./scripts/linux/train_celeba_latent.sh --dry-run \
  --mode continue
```

Continue the current study, skipping compatible completed FM and FM-LN runs:

```bash
./scripts/linux/train_celeba_latent.sh \
  --mode continue
```

On a clean checkout with no latent checkpoint series, start a new suite with:

```bash
./scripts/linux/train_celeba_latent.sh \
  --mode fresh
```

The launcher serializes all seven GPU jobs, including latent-only
MF-Hutchinson, resolves the FM teacher dependency,
generates missing Reflow pairs, and writes a separate timestamped terminal log
for each job. It automatically records the detected GPU name, memory, and
machine label; `--machine-label` remains an explicit override. It does not run
`git add`, commit, or push.

Train only one latent method when debugging or recovering:

```bash
./scripts/linux/train_celeba_latent.sh --only fm_lognorm \
  --mode continue
```

### Windows: train the complete latent suite

The maintained Windows launcher mirrors the Linux dependency order, teacher
binding, Reflow-pair generation, logging, and lifecycle options. Preview first:

```powershell
.\scripts\windows\train_celeba_latent.ps1 -Only all -Mode continue -DryRun
```

Run or resume the complete seven-method latent suite:

```powershell
.\scripts\windows\train_celeba_latent.ps1 -Only all -Mode continue
```

Run one method when debugging or recovering. Hutchinson MF remains latent-only
but is also included in `-Only all`:

```powershell
.\scripts\windows\train_celeba_latent.ps1 -Only mf -Mode fresh
.\scripts\windows\train_celeba_latent.ps1 -Only mf_hutchinson -Mode fresh
```

## Canonical experiment presets

| Dataset | Method | Configuration |
|---|---|---|
| CIFAR-10 | FM | `config/fm_full.json` |
| CIFAR-10 | FM-LN | `config/fm_lognorm_full.json` |
| CIFAR-10 | controlled MF | `config/mf_v3_exact_jvp_b128.json` |
| CIFAR-10 | MF-Distill | `config/mf_distill_full.json` |
| CIFAR-10 | Consistency | `config/consistency_full.json` |
| CIFAR-10 | Reflow | `config/reflow_full.json` |
| CelebA pixel | FM | `config/fm_celeba64.json` |
| CelebA pixel | FM-LN | `config/fm_lognorm_celeba64.json` |
| CelebA pixel | MF | `config/mf_celeba64.json` |
| CelebA pixel | MF-Distill | `config/mf_distill_celeba64.json` |
| CelebA pixel | Consistency | `config/consistency_celeba64.json` |
| CelebA pixel | Reflow | `config/reflow_celeba64.json` |
| CelebA latent | FM | `config/fm_celeba_latent.json` |
| CelebA latent | FM-LN | `config/fm_lognorm_celeba_latent.json` |
| CelebA latent | MF | `config/mf_celeba_latent.json` |
| CelebA latent | MF-Hutchinson CV diagnostic | `config/mf_hutchinson_cv_celeba_latent.json` |
| CelebA latent | MF-Distill | `config/mf_distill_celeba_latent.json` |
| CelebA latent | Consistency | `config/consistency_celeba_latent.json` |
| CelebA latent | Reflow | `config/reflow_celeba_latent.json` |

Additional MF presets are diagnostic experiments, not interchangeable aliases
for the canonical comparison. MF-Hutchinson is deliberately restricted to the
`celeba_latent` dataset and has no pixel-space preset. `config/smoke_fast.json`
checks workflow plumbing and is not a research-quality training configuration.

For exact CIFAR architecture reproduction, use the parallel presets in
[`config/cifar_legacy/`](config/cifar_legacy/README.md) rather than editing the
canonical JSON files in place.

## Checkpoints, recovery, and logs

The canonical run directory is derived from the configuration:

```text
results/<experiment_name>_<dataset>/
  config.json
  run_environment.jsonl
  logs/
  metrics/
  samples/
  checkpoints/
    run_1/
      <AlgorithmClass>_epoch10.pt
      ...
      archive/
```

Checkpoint compatibility is checked against configuration and algorithm
provenance. Numbered run directories avoid silently overwriting an earlier
experiment. Published checkpoint ZIPs include the tensor payload,
configuration, environment metadata, and epoch metadata.

The 22 September 2026 audit verified every populated numbered diffusion series:
14 series had matching checkpoint/archive epoch sets and internally consistent
payload epoch, algorithm, dataset, and version-1 provenance. Historical
checkpoints predate the frozen suite source-identity field; new checkpoints
record it. The scratch VAE remains a separately classified codec experiment and
is not migrated by the diffusion checkpoint organizer.

Use `continue` when the same compatible run was interrupted. Use `fresh` for a
deliberately separate result. Do not point two active jobs at the same result
directory.

Linux suite launchers capture stdout and stderr under a detected device and
representation folder. New logs use:

```text
training_logs/nvidia-geforce-rtx-3090-24gb/
  pixel/
    cifar10_pixel_fm_<host>_<UTC timestamp>.log
    celeba_pixel_fm_lognorm_<host>_<UTC timestamp>.log
  latent/
    fm/
      celeba_latent_fm_<host>_<UTC timestamp>.log
    mf_hutchinson/
      celeba_latent_mf_hutchinson_<host>_<UTC timestamp>.log
  codec/
    <codec-validation-or-training transcript>.log
```

Every new latent-suite transcript records its representation, frozen source identity,
selected config digest, Git commit/dirty-diff identity, lifecycle mode, resolved
checkpoint series, parent-suite timestamp, detected GPU index/UUID/name/memory,
device log partition, and machine label. Sidecars use schema 2 and retain the
same fields plus the final transcript digest. On the current lab
system the automatic label is
`linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb`.

Historical central transcripts were moved into the explicit representation
layout without changing their bytes; matching pre/post digests are recorded in
the timestamped `training_logs/log_migration_manifest_*.json` files. All 48
known transcripts have schema-2 sidecars containing representation, dataset,
algorithm, machine/GPU identification, byte count, and transcript SHA-256.
Suite/aggregate transcripts use `algorithm=multiple` instead of leaving the
field blank.

Inspect or rebuild the catalog with:

```bash
venv/bin/python scripts/catalog_training_logs.py --dry-run
venv/bin/python scripts/catalog_training_logs.py
```

If another confirmed historical machine must be normalized, preview before
writing sidecars:

```bash
venv/bin/python scripts/annotate_training_log_spaces.py \
  --machine-label <label> --gpu-name '<GPU name>' --gpu-memory-gb <GiB>
```

Add `--apply` only after verifying that all discovered transcripts belong to
that machine. This operation changes sidecars, never transcript bytes.

An open log stays at its original path until the writer exits. Do not rename,
truncate, delete, stage, or reorganize it during training.

Logs and sidecars are eligible for Git tracking but launchers never stage or
commit them automatically. This keeps Git operations explicit while preserving
the exact Git identity inside every run record.

GPU entry points share the ownership-checked `results/.lock`. A live owner is
never displaced; stale recovery is explicit:

```bash
venv/bin/python scripts/workflow_guard.py lock-recover --lock-file results/.lock
```

In `--mode continue`, a completed model is skipped only after checkpoint
provenance validation. Evaluation is explicit via `train.py --evaluate-only`
or `evaluate.py`; it is not rerun as a side effect of a completed-run skip.

## Evaluation protocol

Evaluate one checkpoint on Linux:

```bash
./scripts/linux/run_evaluate.sh \
  --algorithm fm \
  --checkpoint results/fm_cifar10/checkpoints/run_1/FlowMatchingAlgorithm_epoch100.pt \
  --config results/fm_cifar10/config.json \
  --make-plots
```

Evaluate all compatible checkpoints for one dataset:

```bash
./scripts/linux/evaluate_all.sh --dataset cifar10 --epoch 100 --dry-run
./scripts/linux/evaluate_all.sh --dataset cifar10 --epoch 100
```

For latent evaluation, the model emits normalized latent tensors. The evaluator
denormalizes and decodes them in bounded batches before FID/IS. It compares
against raw 64×64 CelebA validation RGB images, uses a separate latent-study
reference cache, and records decoder time separately from generative-backbone
NFE timing. A three- or four-channel latent tensor must never be passed directly
to an RGB image saver.

Result rows retain checkpoint, sample count, machine label, configuration, and
source identity. Changing batch size can diagnose memory problems, but canonical
comparisons should retain the versioned batch sizes and all other controls.

## Reports, plots, bundles, and UI

Build aggregate CSV/JSONL tables and plots after training:

```bash
./scripts/linux/make_summary.sh
```

The aggregate includes controlled FM-relative progression, cross-dataset
transfer, pixel-versus-latent pairs, quality/compute Pareto flags, and explicit
coverage of missing experiment cells. The matching and interpretation rules
are documented in [the controlled comparison protocol](docs/COMPARISON_PROTOCOL.md).

While a project trainer is active, the launcher refuses to write unless a
read-only snapshot was explicitly requested:

```bash
./scripts/linux/make_summary.sh --allow-running
```

Generate dataset animations from existing metrics and checkpoint filenames.
The CelebA launcher keeps pixel and latent animations separate and, when no
trainer is active, also decodes checkpoint samples into RGB image grids:

```bash
./scripts/linux/generate_cifar10_outputs.sh
./scripts/linux/generate_celeba_outputs.sh
```

CelebA outputs are written under
`results/aggregate/animations/celeba/`,
`results/aggregate/animations/celeba_latent/`, and
`results/checkpoint_samples/<experiment>/`. During active training,
`--allow-running` produces metric-only snapshots and deliberately skips GPU
checkpoint sampling.

Package portable evidence without global datasets or a virtual environment:

```bash
venv/bin/python scripts/package_dataset_bundles.py --dataset cifar10
venv/bin/python scripts/package_dataset_bundles.py --dataset celeba

./scripts/linux/make_thesis_context.sh --dry-run
./scripts/linux/make_thesis_context.sh
./scripts/linux/refresh_thesis_context.sh --interval 300
```

While training is active, build one read-only live summary and verified ZIP
snapshot with:

```bash
./scripts/linux/make_thesis_context.sh --allow-running
```

To refresh both artifacts every five minutes throughout training, use:

```bash
./scripts/linux/refresh_thesis_context.sh --allow-running --interval 300
```

Only one refresh watcher may run for a checkout. Stop an existing watcher in
its terminal before starting one with different options. Live summaries are
intermediate snapshots; regenerate once without `--allow-running` after
training finishes for the final thesis evidence.

`make_thesis_context.sh` is the canonical Claude Web handoff command. By
default it first rebuilds the aggregate summary and training-log catalog, then
requires every essential implementation/documentation file to be Git-tracked,
checks aggregate schemas and freshness, enforces a 100 MiB uncompressed size
ceiling, builds `thesis_context.zip` atomically, verifies ZIP CRCs, and verifies
the size and SHA-256 digest of every packaged member. Use `--skip-summary` only
when intentionally packaging an already-verified canonical summary. The ZIP
contains its Git revision/worktree status, required-artifact audit, complete
file inventory, and per-file digests in `CONTEXT_MANIFEST.json`.

The compact archive deliberately excludes datasets, environments, raw `.pt`
checkpoints, checkpoint ZIPs, and FID caches. It includes current source,
configs, tests, comparison tables, metrics, plots, sample images, log catalog,
training transcripts, the interactive checkpoint-demo implementation, and the
generated thesis summary. It also explicitly includes the otherwise ignored
collaboration/interface plans, decision record, pretrained model card, and
latent-cache manifest because these small files explain implementation and
codec provenance. Binary weights, latent tensors, Hugging Face cache internals,
IDE state, and nested worktrees remain excluded.

`refresh_thesis_context.sh` repeats the complete summary-and-ZIP operation in a
single-instance loop and runs the CPU-only MF-v3 preflight before each write
cycle. The Windows `refresh_thesis_context.ps1` follows the same contract. Both
default to five-minute intervals and refuse active
training snapshots; add `--allow-running` only when a potentially partial,
read-only live snapshot is intentional. Use `--once` for automation that needs
one rigorously checked refresh and a meaningful exit status.

`THESIS_SUMMARY.md` is a generated evidence snapshot, not the source of truth
for an unfinished active run. Do not report partial latent metrics as final.

Start the local, read-only thesis UI:

```bash
./scripts/linux/run_inference.sh
```

To generate the default multi-seed CelebA presentation outputs first and then
start the UI, use:

```bash
./scripts/linux/run_inference.sh --generate-outputs
```

If a training suite is still active, either run the preparation separately
after it finishes or let the launcher wait safely for the shared GPU lock:

```bash
./scripts/linux/run_inference.sh --generate-outputs --wait-for-gpu
```

The dedicated output command is configurable and generates real checkpoint
samples for every requested `(epoch, NFE, seed)` cell:

```bash
./scripts/linux/generate_inference_outputs.sh \
  --experiments fm_lognorm_celeba \
  --nfe 1,5,10,20,50 \
  --seeds 0,1,2,3 \
  --n-samples 64
```

Then open <http://127.0.0.1:8000>. The UI discovers compatible runs under
`results/` and shows checkpoints and evaluation summaries without loading
weights or running a model. In **Model Studio**, choose the dataset,
algorithm, backbone/run, checkpoint epoch, and NFE. The studio presents the
experiment as **noise → generated images**, with the recorded FID and Inception
Score for the same epoch/NFE selection. Image playback is enabled when that
exact combination has a pre-rendered grid under
`results/checkpoint_samples/<run>/`. The grid was genuinely generated by the
selected checkpoint from the recorded random seed; only the browser transition
between the saved noise and final grid is synthetic unless intermediate solver
states were explicitly recorded. If the artifact is missing, the UI disables
image playback while keeping the recorded loss/FID/IS animation available.
The paired training control uses the same selection and reverses that visual as
**image → noise** while revealing the saved loss, FID, and IS history. This is a
matched explanatory view of the forward/training and reverse/generation
directions, not a claim that browser playback reruns optimization or contains
unrecorded solver states.

Prepare those artifacts after training, outside the live presentation:

```bash
venv/bin/python scripts/generate_checkpoint_samples.py \
  --experiments fm_celeba_latent --nfe 1,5,20 \
  --seeds 0,1,2,3 --n-samples 64
```

That offline command reconstructs the architecture and dataset space from the
saved run config, restores each checkpoint's learned state, and loads the
recorded codec for latent runs before writing RGB grids and provenance
sidecars. Do not run it while a trainer owns the GPU; the shared lock rejects
concurrent model work.

**Play recorded history** reveals saved loss rows epoch by epoch and advances
the measured FID-versus-NFE view at checkpoint epochs. During the live demo,
neither action loads checkpoints, runs inference, acquires the GPU lock, starts
training, or contacts a generation endpoint. The server rejects
`/api/generate` because live model execution is disabled.

The **Thesis evidence** panel distinguishes measured metric cells from missing
ones, reports reconstructed training cost and provenance, warns about
objective-specific loss and NFE interpretation, and compares only runs with an
identical dataset/representation, image size, backbone, seed, FID sample count,
and FID reference cache. Its evidence JSON export contains the current
selection, recorded metrics, protocol, and provenance; it does not recompute
or invent missing measurements.

## Reproducibility rules

For a defensible experiment:

1. Start from a versioned JSON configuration and record any override.
2. Use an explicit `--algorithm`, `--config`, and `--mode`.
3. Preserve dataset split, preprocessing, seed, sample count, NFE, and reference
   statistics across comparisons.
4. Preserve codec identity and content-addressed cache metadata for latent runs.
5. Let launchers detect the machine/GPU label for long jobs; use an explicit
   override only when the automatic hardware identity is unsuitable.
6. Keep raw logs, metrics, validation reports, configs, and provenance together.
7. Never compare latent and pixel rankings as though their reconstruction
   ceilings and representation costs were identical.
8. Treat teacher training and Reflow pair generation as part of total method
   cost when discussing efficiency.

The project records checkpoint and environment provenance, restores RNG and
dataloader state where supported, and archives prior numbered checkpoints. The
the repository-wide GPU lock, frozen run-wide source identity, completed-run
skip policy, and representation-aware log migration are implemented. Their
requirements and acceptance criteria remain documented in
[docs/POST_TRAINING_FIX_PROMPT.md](docs/POST_TRAINING_FIX_PROMPT.md).

## Known challenges and practical responses

| Challenge | Consequence | Current response |
|---|---|---|
| CelebA host quota/authentication | Automatic download may fail | Support manual official-file placement and rerun validation |
| Long GPU runs and thermal limits | Interrupted or throttled experiments | Numbered resumable checkpoints, per-job logs, serialized jobs |
| Codec reconstruction ceiling | Latent FID cannot be interpreted alone | Validate first, preserve rFID/PSNR, require explicit override |
| Cache/config mismatch | Training on the wrong latent representation | Content-addressed cache plus manifest/checkpoint validation |
| Distillation teacher memory | Higher peak memory and startup cost | Lazy teacher load, smaller canonical student batches |
| Exact-JVP Mean Flow cost | Greater compute and memory | Retain FD-JVP and exact-JVP as labeled controlled variants |
| Evaluation memory | OOM during 5,000-image metrics | Batched generation/decoding, CPU accumulation, reused real stats |
| Reflow dataset size | RAM/disk pressure | Chunked generation and preallocated pair storage |
| Low-CPU-memory Diffusers warning | Slow or unsupported loading path | Install declared `accelerate` dependency in the project venv |
| One physical GPU | Competing jobs corrupt timing or exhaust memory | Serialize jobs and preview suites with `--dry-run` |
| Unfinished run summaries | Partial values mistaken for conclusions | Label snapshots and regenerate only after training completes |

## Troubleshooting

### `accelerate` warning appears during codec loading

Confirm that the project interpreter can import it:

```bash
venv/bin/python -c "import accelerate; print(accelerate.__version__)"
venv/bin/pip install -r requirements.txt
```

On Windows, replace `venv/bin/python` with
`.\venv\Scripts\python.exe`. Do not fix the project venv by installing into an
unrelated system Python.

### CUDA out of memory

Stop overlapping GPU processes, inspect the active process with `nvidia-smi`,
and retry the same checkpoint using the intended configuration. A temporary
batch-size override is useful for diagnosis, but it changes the controlled
experiment and must be recorded. Never run two suite jobs concurrently on one
GPU.

### No compatible checkpoint is found

Check the lifecycle mode, algorithm class, run number, configuration, and
expected epoch. Teacher-dependent latent configs expect the completed FM
checkpoint at the configured path; a partial FM run is not a valid teacher.

### Latent cache is missing or ambiguous

Rerun `codec/cache_latents.py` with the accepted codec. Let the dataset resolve
the exact manifest match. Do not copy a tensor into a differently named cache
or weaken the identity checks.

### CelebA download fails

Follow the dataset message, obtain the official files manually, place them
under `data/raw/`, and rerun setup. Do not replace CelebA with CelebA-HQ images;
the latter name identifies the codec's pretraining source, not this experiment's
dataset split.

### A run was interrupted

Read the final log lines, confirm there is no remaining trainer, then use
`--mode continue`. Starting with `fresh` creates a new run and is not a resume
operation.

## Verification and development

Run CPU/static checks without starting a research training job:

```bash
venv/bin/python -m pytest -q
venv/bin/python scripts/verify_workflow.py --dataset none
./scripts/linux/train_all.sh --dataset cifar10 --dry-run
venv/bin/python scripts/verify_project_layout.py
venv/bin/python scripts/smoke_latent.py --mode static
```

Windows:

```powershell
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe scripts\verify_workflow.py --dataset none
.\venv\Scripts\python.exe scripts\verify_project_layout.py
.\venv\Scripts\python.exe scripts\smoke_latent.py --mode static
```

The layout verifier is read-only and reports active trainer PIDs. Use
`--fail-if-training` when a future structural migration must not overlap with a
run. Tests cover configuration parsing, lifecycle behavior, provenance,
numbered checkpoints, Mean Flow JVP behavior, Consistency sampling, platform
entrypoints, aggregation, latent contracts, and web discovery.

## Documentation map

| Document | Purpose |
|---|---|
| [RUNBOOK.md](RUNBOOK.md) | Exact CelebA latent codec, cache, smoke, and training gates |
| [BLOCKER_DECISIONS.md](BLOCKER_DECISIONS.md) | Resolved latent architecture decision and measured parameter counts |
| [scripts/README.md](scripts/README.md) | Linux/Windows command index |
| [docs/THEORY_NOTES.md](docs/THEORY_NOTES.md) | Algorithm derivations and references |
| [docs/IMPLEMENTATION_CHANGES.md](docs/IMPLEMENTATION_CHANGES.md) | Chronological engineering and validation history |
| [docs/TRAINING_TIME_ESTIMATES.md](docs/TRAINING_TIME_ESTIMATES.md) | GPU-time, memory, and optimization notes |
| [docs/CONSISTENCY_TUNING_NOTES.md](docs/CONSISTENCY_TUNING_NOTES.md) | Consistency-specific tuning guidance |
| [docs/LINUX_VERIFICATION.md](docs/LINUX_VERIFICATION.md) | Linux environment verification checklist |
| [docs/POST_TRAINING_FIX_PROMPT.md](docs/POST_TRAINING_FIX_PROMPT.md) | Implemented isolation, provenance, continuation, and log-migration acceptance specification |
| [docs/report/main.tex](docs/report/main.tex) | Thesis/report source |
| [THESIS_SUMMARY.md](THESIS_SUMMARY.md) | Generated result snapshot; may be partial during training |

## Operational safety

- Preview long launchers with `--dry-run`.
- Do not delete, move, package, or rewrite an actively written log/checkpoint.
- Do not launch evaluation or another trainer merely to inspect a live run.
- Keep generated datasets, raw checkpoints, caches, and virtual environments
  out of source control.
- Do not resume either rejected codec as a substitute for the selected VQ-f4
  experiment.
- Regenerate summaries after runs finish; do not silently edit measured values.
- Treat a manual quality override as evidence to disclose, not as a passed gate.
