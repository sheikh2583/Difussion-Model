# Reconstructable thesis progression

This document is the canonical map from the project chronology to code,
artifacts, and claims. “Complete” means measured local evidence exists;
“partial” means the path is runnable but the measured comparison is incomplete;
“implementation-only” means code and CPU tests exist but no result should be
reported as an experiment.

Run the read-only audit at any time:

```bash
python scripts/verify_thesis_progression.py
```

The three training stages have dedicated entrypoints:

```bash
./scripts/linux/01_train_cifar10.sh
./scripts/linux/02_train_celeba_pixel.sh
./scripts/linux/03_train_celeba_latent.sh
```

They default to non-destructive fresh runs and automatically select the next
`checkpoints/run_N/` series. This workstation has one RTX 3090. The numbered
wrappers share a blocking queue lock, so scripts started in separate terminals
run one at a time and print elapsed wall time when each stage exits.

## 1. Shared SimpleUNet scaffold — complete

The thesis starts with the shared time-conditioned U-Net, generic trainer,
sampler, evaluator, configuration schema, and result logging. Algorithm classes
change the objective and sampler; they do not silently replace the backbone.

The frozen historical implementation remains available as
`legacy_cifar_unet`. Despite its historical name, it is fully convolutional and
is statically tested on both project pixel spaces:

- CIFAR-10 RGB: `3×32×32`;
- CelebA RGB: `3×64×64`.

It is deliberately rejected for the later `3×16×16` latent representation so
the pixel and latent stages cannot be conflated.

## 2. CIFAR-10 with the first three objectives — complete

Git chronology introduces the first three research objectives in this order:

1. Flow Matching;
2. Mean Flow;
3. Flow Matching with logit-normal time sampling.

The current reproducible presets are `fm_full.json`, `mf_full.json`, and
`fm_lognorm_full.json`. Completed CIFAR logs, metrics, and final checkpoints are
retained under their corresponding `results/*_cifar10/` directories.

## 3. CIFAR-10 six-method comparison — complete

MF-Distill, Consistency, and Reflow extend the comparison to six canonical
methods. Teacher and pair dependencies are explicit. The completed evidence is
aggregated at matched NFE values rather than comparing unlike training losses.

```bash
./scripts/linux/train_all.sh --dataset cifar10 --dry-run
```

## 4. CelebA 64×64 pixel transfer — partial

All six pixel-space CelebA presets parse and their dependency-aware launcher is
runnable. Only FM and FM-LN have completed measured results. MF, MF-Distill,
Consistency, and Reflow are templates/pending runs and must not be described as
completed CelebA pixel comparisons.

```bash
./scripts/linux/train_all.sh --dataset celeba --dry-run
```

Pixel Reflow additionally requires `data/reflow_pairs_celeba.pt`, which is not
currently present and is reported as a blocker by `verify_workflow.py`.

## 5. CelebA frozen-latent migration — complete

The accepted frozen VQ-f4 boundary maps CelebA RGB `3×64×64` images to
normalized `3×16×16` states. This is not a pretrained generative U-Net. The
completed latent configs also widen the trainable backbone from the CelebA
pixel model's 8,947,459 parameters to 24,026,627 parameters while reducing it
from four resolution stages to three. Representation, capacity, and
spatial-compute changes must therefore be reported separately. Codec identity,
cache identity, normalization statistics, and reconstruction limitations remain
explicit.

The six canonical latent methods have configurations and measured evidence.
Latent tensors remain unbounded until they are denormalized and decoded.

## 6. Latent fixes and MF-Hutchinson — partial diagnostic

The ordinary latent MF late-loss rise and the original Hutchinson divergence
remain negative evidence. The control-variate Hutchinson run started healthily
and produced metrics through epoch 91 plus an epoch-90 recovery checkpoint,
but it did not reach epoch 100. It is a seventh latent-only diagnostic—not a
member of the six-method cross-dataset leaderboard.

## 7. Further latent inference extensions — implementation-only

`AdaptiveMeanFlowSampler` and `MultiScaleMeanFlowPipeline` wrap a trained
latent MF checkpoint. They are single-model inference strategies, not training
algorithms and not registry entries. Their CPU contracts are tested, and the
sample CLI records JSON metadata beside generated grids. No FID/IS result exists
for either extension, so they belong in future-work or implementation sections.

```bash
python scripts/sample_mean_flow_extensions.py --help
python -m pytest -q tests/test_latent_extensions.py
```

## 8. Higher-resolution face generation — complete at 64×64

The documented resolution jump is CIFAR-10 `32×32` to CelebA `64×64` RGB.
Latent generation happens at `16×16` and the frozen decoder returns `64×64`
RGB. The repository does not contain a trained 256×256 generative experiment;
the codec’s CelebA-HQ source name must not be used to imply one.

## 9. Comparison graphs, metrics, and reasoning — complete with gaps shown

The aggregate layer preserves separate views for:

- algorithm progression relative to FM;
- cross-dataset direction and magnitude;
- CelebA pixel versus decoded latent representation;
- FID/NFE and FID/latency Pareto membership;
- training cost and peak memory;
- provenance and missing-comparison coverage.

FID and Inception Score are the measured quality metrics. Codec reconstruction
error is disclosed for latent results, and absent modern metrics or missing
experiment cells remain explicit rather than being inferred.

```bash
./scripts/linux/make_summary.sh --dry-run
```

Do not regenerate the final thesis context while a trainer is active. The
compact context contains code, configs, metrics, plots, logs, and provenance;
large checkpoints and tensor datasets are separate research artifacts.
