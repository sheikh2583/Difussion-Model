# Thesis Results Summary

Generated from the canonical experiment metric files by `scripts/aggregate_results.py`. Rerun the generator after active training finishes to produce the final snapshot.

## Evaluation metrics

| Dataset | Space | Experiment | Run identity | Algorithm | Train epoch | Eval epoch | Samples | FID@1 | FID@5 | FID@10 | FID@20 | FID@50 | IS@20 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | pixel | consistency_cifar10 | unknown@unknown | ConsistencyAlgorithm | 100 | 100 | 5000 | 137.084 | 111.758 | 107.464 | 107.285 | — | 4.160 |
| cifar10 | pixel | final_smoke_cifar10 | unknown@unknown | MockAlgorithm | 1 | 1 | unknown | — | 389.533 | — | — | — | — |
| cifar10 | pixel | fm_cifar10 | unknown@unknown | FlowMatchingAlgorithm | 100 | 100 | 5000 | 374.039 | 63.638 | 47.091 | 41.790 | 38.821 | 6.430 |
| unknown | pixel | fm_cifar10_5k | unknown@unknown | FlowMatchingAlgorithm | — | 100 | 5000 | 374.039 | 63.638 | 47.091 | 41.790 | 38.821 | 6.437 |
| cifar10 | pixel | fm_lognorm_cifar10 | unknown@unknown | FlowMatchingLognormAlgorithm | 100 | 100 | 5000 | 372.232 | 62.323 | 48.559 | 41.919 | 35.960 | 6.542 |
| cifar10 | pixel | fm_lognorm_rtx3060 | unknown@unknown | FlowMatchingLognormAlgorithm | 100 | 100 | unknown | 362.604 | 96.452 | 78.947 | 72.662 | — | 6.180 |
| cifar10 | pixel | mf_cifar10 | unknown@unknown | MeanFlowAlgorithm | 100 | 100 | 5000 | 93.351 | 77.766 | 76.200 | 75.163 | — | 4.335 |
| cifar10 | pixel | mf_distill_cifar10 | unknown@unknown | MeanFlowDistillAlgorithm | 100 | 100 | 5000 | 128.366 | 48.669 | 46.646 | 45.300 | — | 6.070 |
| cifar10 | pixel | mf_v2_exactjvp_probe_cifar10 | unknown@unknown | MeanFlowAlgorithm | 5 | — | unknown | — | — | — | — | — | — |
| cifar10 | pixel | mf_v2_probe_cifar10 | unknown@unknown | MeanFlowAlgorithm | 10 | — | unknown | — | — | — | — | — | — |
| cifar10 | pixel | reflow_cifar10 | unknown@unknown | ReflowAlgorithm | 100 | 100 | 5000 | 48.709 | 44.968 | 44.754 | 44.577 | — | 5.969 |
| cifar10 | pixel | smoke_cifar10 | unknown@unknown | MockAlgorithm | — | — | unknown | — | 383.377 | — | — | — | — |

## Controlled comparison artifacts

The aggregate directory contains protocol-matched FM-relative gains, cross-dataset transfer checks, pixel-versus-latent pairs, Pareto flags, and an experiment-coverage matrix. Positive `fid_improvement_percent` means lower FID than FM. Missing peers remain explicit instead of being silently compared.

- Transfer groups available: 22
- Pixel/latent pairs available: 0
- Latent FID includes codec reconstruction error; decoder time and codec quality must be reported separately.

## Interactive checkpoint demo and external handoff

The local inference UI can play saved checkpoints in epoch order using one selected algorithm, NFE, image count, and fixed seed. It advances the loss curve with the checkpoint, shows a conceptual noise-to-sample transition, and decodes latent outputs through the recorded codec. Backbone and decoder timings are reported separately.

Build the verified Claude Web handoff with `./scripts/linux/make_thesis_context.sh`. The command rebuilds this summary, refreshes the training-log catalog, validates comparison tables and Git-tracked implementation files, then verifies every ZIP member against its SHA-256 digest.

## Training cost

| Experiment | Algorithm | Epochs | Training hours | Peak GPU memory (MiB) |
|---|---|---:|---:|---:|
| consistency_cifar10 | ConsistencyAlgorithm | 100 | 2.66 | 4122.8 |
| final_smoke_cifar10 | MockAlgorithm | 1 | 0.01 | 845.7 |
| fm_cifar10 | FlowMatchingAlgorithm | 100 | 1.75 | 3423.2 |
| fm_lognorm_cifar10 | FlowMatchingLognormAlgorithm | 100 | 1.70 | 3916.5 |
| fm_lognorm_rtx3060 | FlowMatchingLognormAlgorithm | 100 | 3.60 | 1908.6 |
| mf_cifar10 | MeanFlowAlgorithm | 100 | 1.41 | 3656.3 |
| mf_distill_cifar10 | MeanFlowDistillAlgorithm | 100 | 3.12 | 2762.4 |
| mf_v2_exactjvp_probe_cifar10 | MeanFlowAlgorithm | 5 | 4.04 | 12521.2 |
| mf_v2_probe_cifar10 | MeanFlowAlgorithm | 10 | 0.20 | 4310.6 |
| reflow_cifar10 | ReflowAlgorithm | 100 | 1.72 | 4469.3 |
