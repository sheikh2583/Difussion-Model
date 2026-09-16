# Implementation Log

## "How Fast Can Generative Models Get, and What Does It Cost?"

Last verified: 2026-09-17.

This describes the current repository, not an older zip bundle. Complete setup
before running the commands below.

## Verification

Run the same repository check on Windows or Linux:

```bash
python scripts/verify_workflow.py --dataset cifar10
```

Use `--dataset celeba` to download and validate a CelebA batch. Use
`--strict-prerequisites` when teacher checkpoints and generated Reflow pairs must
already exist. Preview long workflows without launching training:

```bash
./scripts/train_all.sh --dry-run                   # Linux
.\scripts\train_all.ps1 -DryRun                  # Windows
./scripts/evaluate_all.sh --dry-run                # Linux
.\scripts\evaluate_all.ps1 -DryRun               # Windows
```

## Current verified state

| Area | State | Evidence or blocker |
|---|---|---|
| Cross-platform bootstrap | Ready | Completed on Windows 11 with Python 3.12, CUDA 12.8, all dependencies, pretrained metric assets, CIFAR-10, and CelebA. POSIX launcher syntax/path handling was exercised through a Linux WSL environment; full Linux dependency installation still needs a general-purpose Linux host. |
| Algorithm registry | Ready | Six research algorithms plus the `mock` smoke utility are registered. |
| CIFAR-10 pipeline | Ready | Batch verified as `(32, 3, 32, 32)` with range `[-1, 1]`. |
| CelebA pipeline | Ready locally | The 1.44 GB image archive and official annotations downloaded successfully; a batch was verified as `(64, 3, 64, 64)` with range `[-1, 1]`. |
| Smoke pipeline | Ready | Two epochs, FID/IS, checkpoints, and checkpoint archives completed. |
| MF-Distill prerequisite | Ready locally | The expected FM epoch-100 teacher exists. |
| Consistency prerequisite | Ready locally | The expected FM epoch-100 teacher exists; tuning remains. |
| Reflow prerequisite | Ready locally | `data/reflow_pairs_cifar10.pt` contains 50,000 validated NFE-50 pairs and loads through `ReflowAlgorithm`. |
| Full tournament | Partial | Only one expected epoch-100 evaluation path is currently populated. |

## Phase 0: datasets

CIFAR-10 is downloaded by default. CelebA is optional:

```bash
python bootstrap.py --yes
python bootstrap.py --yes --datasets celeba
python scripts/verify_workflow.py --dataset celeba
```

Torchvision expects CelebA images and official annotation files under
`data/raw/celeba/`. If Google Drive is unavailable, obtain both images and
annotations from the official CelebA source and place them there.

## Phase 1: FM and MF on CelebA

Status: dataset verified; model training pending.

```bash
python train.py --algorithm fm --config config/fm_celeba64.json
python train.py --algorithm mf --config config/mf_celeba64.json
```

All six methods have CelebA presets. The teacher-based presets require the
CelebA FM checkpoint, and Reflow requires CelebA pairs generated from it.

## Phase 2: Mean Flow Distillation

Status: code and CIFAR-10 teacher ready; training pending.

```bash
python train.py --algorithm mf_distill --config config/mf_distill_full.json
```

Both `config/mf_distill_full.json` and `config/mf_distill_celeba64.json` are
provided. The CelebA preset becomes runnable after its FM teacher is trained.

## Phase 3: Consistency Distillation

Status: code and teacher ready; training and tuning pending.

```bash
python train.py --algorithm consistency --config config/consistency_full.json
```

Treat loss-weight, EMA-decay, and FID guidance as tuning hypotheses until an
empirical run confirms them.

## Phase 4: Rectified Flow Reflow

Status: implementation, generator, and CIFAR-10 pair artifact verified; training pending.

```bash
python scripts/generate_reflow_pairs.py \
  --checkpoint results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt \
  --config config/fm_full.json \
  --n-pairs 50000 --nfe 50 \
  --output data/reflow_pairs_cifar10.pt

python train.py --algorithm reflow --config config/reflow_full.json
```

The generator loaded the existing FM epoch-100 checkpoint and produced the full
50,000-pair NFE-50 artifact. Both tensors have shape `(50000, 3, 32, 32)` and
float32 dtype; generated endpoints are finite and within `[-1, 1]`. The saved
1.144 GB file was loaded through `ReflowAlgorithm`, and a finite Reflow loss was
computed successfully without performing a training update.

## Phase 5: evaluation

Status: partial; only checkpoints present at the requested epoch can run.

```bash
./scripts/evaluate_all.sh --epoch 100
.\scripts\evaluate_all.ps1 -Epoch 100
```

The scripts skip missing checkpoints and evaluate available ones. Run dry-run
mode first. Aggregate all available JSONL records into a summary table and plots
with `python scripts/aggregate_results.py`.

## Optional Mean Flow extensions

Adaptive-NFE and multiscale sampling are implemented as inference wrappers and
are intentionally not separate training algorithms. Inspect their portable CLI:

```bash
python scripts/sample_mean_flow_extensions.py adaptive --help
python scripts/sample_mean_flow_extensions.py multiscale --help
```

## Paths

Outputs follow `results/<experiment_name>_<dataset_name>/`. Run the documented
`scripts/...` commands from the repository root. Large datasets, Reflow pairs,
and results remain outside Git.

## Verification limits

- No single host can prove operation on every Windows/Linux installation.
- Windows runtime behavior is verified. The beginner POSIX launchers were parsed
  by an actual Linux `sh` through WSL and their missing-environment behavior was
  exercised, but full Linux dependency installation still needs a normal Linux
  machine or CI runner.
- CelebA download and batch validation are complete locally.
- Full model training remains intentionally excluded. CelebA teacher checkpoints
  and CelebA Reflow pairs therefore remain training-dependent.
- Model-quality and FID targets require real experiments; setup tests cannot
  guarantee them.

---

## [Antigravity] Track A — Verification results (2026-09-17)

### A1 — CelebA dataset
- Images: **202,599 jpg files** extracted in `data/raw/celeba/img_align_celeba/`
- All 5 annotation files present
- Loader smoke call (`fm_celeba64.json`, batch=4):
  shape `(4, 3, 64, 64)`, dtype `float32`, range `[-0.9922, 1.0000]` ✓
- **PASS**

### A2 — verify_workflow.py (CIFAR-10)
- 6 research algorithms + mock registered ✓
- All 14 config files valid ✓
- FM teacher checkpoint present ✓  (`results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt`)
- Reflow pairs present ✓  (`data/reflow_pairs_cifar10.pt`)
- CIFAR-10 batch: shape `(32, 3, 32, 32)`, range `[-1.000, 1.000]` ✓
- **PASS**

### A3 — aggregate_results.py
- Aggregated **262 records** from 6 JSONL files across 4 existing runs
- Outputs written to `results/aggregate/` (JSONL + CSV + plots)
- **PASS** — pipeline ready for real tournament results

### A4 — Linux checklist
- Created `docs/LINUX_VERIFICATION.md`

---

## [Codex] Track B — pipeline readiness (2026-09-17)

- Corrected the CelebA Reflow generation command to use the derived
  `fm_celeba` run directory and canonical `reflow_pairs_celeba.pt` output.
- Added atomic `results/.lock` enforcement, safe existing-output refusal,
  explicit `--overwrite`, and atomic output replacement to the Reflow pair
  generator.
- Generated two pairs from the real CIFAR-10 epoch-100 FM checkpoint in a
  temporary file and consumed them through `ReflowAlgorithm.training_step()`;
  tensor shapes were correct and the loss was finite. The temporary artifact
  was removed and the GPU lock was released.
- Validated MF-Distill and Consistency teacher paths for CIFAR-10 and CelebA.
  A non-optimizing Consistency forward pass and NFE-1 sample pass both produced
  finite outputs.
- Added `scripts/tune_consistency.py` with four isolated variants, shared GPU
  locking, dry-run validation, per-variant outputs, and JSON/Markdown result
  summaries. Only `--dry-run` was executed; model training remains manual.

Real pair-generation commands (run only after the matching teacher exists):

```bash
python scripts/generate_reflow_pairs.py \
  --checkpoint results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt \
  --config config/fm_full.json --n-pairs 50000 --nfe 50 \
  --output data/reflow_pairs_cifar10.pt --overwrite

python scripts/generate_reflow_pairs.py \
  --checkpoint results/fm_celeba/checkpoints/FlowMatchingAlgorithm_epoch100.pt \
  --config config/fm_celeba64.json --n-pairs 50000 --nfe 50 \
  --output data/reflow_pairs_celeba.pt
```
