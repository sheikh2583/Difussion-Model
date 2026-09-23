# Training-flow sanity check and runtime estimates

Date: 2026-09-17

All twelve configured flows (six algorithms on CIFAR-10 and CelebA) completed
disposable synthetic AMP forward/backward/optimizer steps and NFE-1 sampling on
the available RTX 4070 Laptop GPU. No dataset training or checkpoint generation
was performed.

These are historical **pixel-space** estimates. They do not estimate the newer
CelebA VQ-f4 latent suite or its seventh MF-Hutchinson diagnostic; use observed
per-epoch latent logs for those jobs rather than extrapolating this table.

## Estimated end-to-end hours per model

These estimates include 100 training epochs, the configured periodic
evaluations, the tournament's final evaluation, approximate data/checkpoint
overhead, and Reflow pair generation where applicable.

| Dataset | Model | RTX 4070 Laptop 8 GB | RTX 3090 24 GB | RTX 3060 Laptop 6 GB |
|---|---|---:|---:|---:|
| CIFAR-10 | FM | 2.5 h | 1.6 h | 4.5 h |
| CIFAR-10 | FM-LogNorm | 2.5 h | 1.5 h | 4.4 h |
| CIFAR-10 | Mean Flow | 2.0 h | 1.2 h | 3.5 h |
| CIFAR-10 | Consistency | 3.0 h | 1.9 h | 5.4 h |
| CIFAR-10 | MF-Distill | 5.2 h | 3.2 h | 9.4 h |
| CIFAR-10 | Reflow, including pairs | 2.3 h | 1.4 h | 4.1 h |
| CelebA | FM | 23.9 h | 14.8 h | 43.0 h |
| CelebA | FM-LogNorm | 23.9 h | 14.8 h | 43.0 h |
| CelebA | Mean Flow | 26.5 h | 16.4 h | 47.7 h |
| CelebA | Consistency | 40.3 h | 25.0 h | 72.5 h |
| CelebA | MF-Distill | 64.7 h | 40.1 h | 116.5 h |
| CelebA | Reflow, including pairs | 24.7 h | 15.3 h | 44.5 h |

Approximate full sequential tournament totals:

- RTX 4070 Laptop: **about 222 hours (9.2 days)**; allow roughly 8–11 days.
- RTX 3090 desktop: **about 137 hours (5.7 days)**; allow roughly 5–7 days.
- RTX 3060 Laptop: **about 399 hours (16.6 days)**; allow roughly 15–20 days.

Laptop power limits and sustained cooling cause large variation. The RTX 3090
column refers specifically to the 24 GB desktop card. Its numbers and the RTX
3060 Laptop numbers are scaled projections rather than measurements. Re-run the
included benchmark on the destination machine for a tighter estimate.

## Measured RTX 4070 Laptop memory

Peak allocated training memory at the configured batch sizes:

| Model | CIFAR-10 | CelebA |
|---|---:|---:|
| FM | 2.19 GiB | 4.34 GiB |
| FM-LogNorm | 2.20 GiB | 4.34 GiB |
| Mean Flow | 2.16 GiB | 4.26 GiB |
| Consistency | 2.51 GiB | 2.58 GiB |
| MF-Distill | 1.18 GiB | 2.28 GiB |
| Reflow | 2.20 GiB | 2.24 GiB |

Allocated memory excludes the CUDA context, allocator reserve, desktop display,
and other applications. The 6 GB RTX 3060 Laptop is therefore excluded from the
controlled tournament. Reducing its batch sizes would alter optimizer-step
counts, gradient noise, and Consistency EMA updates, creating a different
training condition.

The final protocol uses the canonical config batch sizes without overrides:

1. Run the disposable exact-config checks on the 8 GB RTX 4070 Laptop and fix
   any mechanical training-flow errors.
2. Transfer the same commit and unchanged configs to the 24 GB desktop RTX 3090.
3. Run the full sequential tournament without `--batch-size` / `-BatchSize`.
4. Keep epochs, seeds, NFE values, generated sample counts, FID reference data,
   evaluation frequency, and checkpoint selection identical for every model.

The batch-size override remains available for diagnostics, but results produced
with it must not be mixed into the controlled comparison.

## Reproduce on another GPU

After initialization, run:

```bash
venv/bin/python scripts/benchmark_training_flows.py
```

The script writes `results/training_flow_benchmark.json`. It uses disposable
random inputs and temporary prerequisites, performs no real dataset training,
and holds `results/.lock` while benchmarking.

## Improvements applied before training

- Fixed MF-Distill's AMP dtype mismatch, which previously crashed its first
  training step.
- Updated Consistency EMA after every optimizer step and persisted/restored EMA
  weights in checkpoints. Legacy checkpoints fall back to student weights
  instead of an invalid random EMA.
- Batched 5,000-image evaluation generation in groups of 32 and moved completed
  batches to CPU, preventing a single enormous GPU allocation.
- Reused real FID statistics across all NFE values in one evaluation sweep.
- Removed duplicate final evaluation inside `run_full()`.
- Made teacher loading lazy, so teacher-dependent checkpoints can be sampled
  without the original teacher file and unused teachers do not occupy VRAM.
- Preallocated Reflow pair storage, avoiding a temporary host-RAM doubling, and
  reduced the generator's default GPU batch size from 256 to 64.
- Added a diagnostic batch-size override to the CLI and tournament scripts; the
  controlled tournament intentionally does not use it.

## Estimate limitations

- The RTX 4070 measurements used PyTorch 2.11.0 with CUDA 12.8 on the current
  laptop. Long-run thermal behavior was not simulated.
- Synthetic batches isolate GPU compute. Actual JPEG decoding, storage speed,
  checkpoint archives, antivirus scanning, and background applications can add
  substantial time, especially for CelebA.
- RTX 3090 and RTX 3060 Laptop numbers use conservative scaling from the measured
  4070 Laptop result. Exact laptop TGP matters more than the GPU name alone.
- Model convergence is not guaranteed by a flow sanity check. Inspect early loss
  curves and generated samples before committing to every 100-epoch CelebA run.

Official specification references:

- [NVIDIA laptop GPU comparison](https://www.nvidia.com/en-us/geforce/laptops/compare/)
- [NVIDIA RTX 30-series laptop specifications](https://www.nvidia.com/en-gb/geforce/laptops/30-series/)
- [NVIDIA RTX 3090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/)
