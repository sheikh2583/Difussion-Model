# Implementation Log

## "How Fast Can Generative Models Get, and What Does It Cost?"

Last verified: 2026-09-16.

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
| Cross-platform bootstrap | Ready | Completed on Windows 11 with Python 3.12, CUDA 12.8, dependencies, and CIFAR-10. Linux still needs runtime validation on an actual Linux host or active container. |
| Algorithm registry | Ready | Six research algorithms plus the `mock` smoke utility are registered. |
| CIFAR-10 pipeline | Ready | Batch verified as `(32, 3, 32, 32)` with range `[-1, 1]`. |
| CelebA pipeline | Code ready | Loader and 64x64 presets for all six methods load; the 1.4 GB dataset is not present locally. |
| Smoke pipeline | Ready | Two epochs, FID/IS, checkpoints, and checkpoint archives completed. |
| MF-Distill prerequisite | Ready locally | The expected FM epoch-100 teacher exists. |
| Consistency prerequisite | Ready locally | The expected FM epoch-100 teacher exists; tuning remains. |
| Reflow prerequisite | Blocked | `data/reflow_pairs_cifar10.pt` has not been generated. |
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

Status: pending dataset verification and GPU time.

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

Status: implementation and generator ready; pair generation pending.

```bash
python scripts/generate_reflow_pairs.py \
  --checkpoint results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt \
  --config config/fm_full.json \
  --n-pairs 50000 --nfe 50 \
  --output data/reflow_pairs_cifar10.pt

python train.py --algorithm reflow --config config/reflow_full.json
```

The direct generator invocation now resolves project imports correctly. A
two-pair, one-step probe successfully loaded the existing legacy FM checkpoint,
generated tensors of shape `(2, 3, 32, 32)`, and loaded them into Reflow. The
full 50,000-pair artifact is still intentionally pending.

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
- Windows runtime behavior is verified. Linux syntax and path handling were
  reviewed, but this machine has no active Linux container or WSL distribution.
- CelebA download and batch validation remain pending.
- Full training and 50,000-pair generation are intentionally not part of a
  setup/workflow smoke test.
- Model-quality and FID targets require real experiments; setup tests cannot
  guarantee them.
