# Thesis Results Summary

Generated from the canonical experiment metric files by `scripts/aggregate_results.py`. Rerun the generator after active training finishes to produce the final snapshot.

## Evaluation metrics

| Dataset | Space | Experiment | Run identity | Algorithm | Train epoch | Eval epoch | Samples | FID@1 | FID@5 | FID@10 | FID@20 | FID@50 | IS@20 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| celeba_latent | latent | consistency_celeba_latent | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@5269eeabfc91 | ConsistencyAlgorithm | 100 | 75 | 5000 | 302.530 | 299.796 | 297.435 | 293.102 | — | 2.335 |
| cifar10 | pixel | consistency_cifar10 | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@2ea2a5e2039b | ConsistencyAlgorithm | 100 | 100 | 5000 | 130.968 | 100.845 | 95.879 | 99.635 | — | 4.333 |
| celeba | pixel | fm_celeba | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@2ea2a5e2039b | FlowMatchingAlgorithm | 100 | 100 | 5000 | 223.126 | 44.357 | 25.975 | 19.808 | 16.779 | 2.410 |
| celeba | pixel | fm_celeba | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@c1f065aee704 | FlowMatchingAlgorithm | 100 | 100 | 5000 | 223.126 | 44.357 | 25.975 | 19.808 | 16.779 | 2.410 |
| celeba_latent | latent | fm_celeba_latent | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@1575e2beaf3f | FlowMatchingAlgorithm | 100 | 100 | 5000 | 329.205 | 305.048 | 302.265 | 299.849 | 297.914 | 2.639 |
| cifar10 | pixel | fm_cifar10 | unknown@unknown | FlowMatchingAlgorithm | 100 | — | unknown | — | — | — | — | — | — |
| cifar10 | pixel | fm_cifar10 | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@2ea2a5e2039b | FlowMatchingAlgorithm | — | 100 | 5000 | 380.620 | 58.568 | 47.243 | 45.467 | 43.427 | 6.008 |
| celeba | pixel | fm_lognorm_celeba | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@cef2edc1be6d | FlowMatchingLognormAlgorithm | 98 | 80 | 5000 | 215.889 | 35.888 | 23.292 | 19.557 | 17.349 | 2.598 |
| celeba | pixel | fm_lognorm_celeba | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@e04529a3f533 | FlowMatchingLognormAlgorithm | 100 | 100 | 5000 | 193.165 | 30.538 | 19.580 | 16.555 | 15.304 | 2.674 |
| celeba_latent | latent | fm_lognorm_celeba_latent | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@69763e69e3ba | FlowMatchingLognormAlgorithm | 100 | 100 | 5000 | 322.722 | 304.616 | 301.459 | 299.177 | 297.595 | 2.658 |
| cifar10 | pixel | fm_lognorm_cifar10 | unknown@unknown | FlowMatchingLognormAlgorithm | 100 | — | unknown | — | — | — | — | — | — |
| cifar10 | pixel | fm_lognorm_cifar10 | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@2ea2a5e2039b | FlowMatchingLognormAlgorithm | — | 100 | 5000 | 384.190 | 57.260 | 44.183 | 38.601 | 35.400 | 6.522 |
| celeba_latent | latent | mf_celeba_latent | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@69763e69e3ba | MeanFlowAlgorithm | 100 | 100 | 5000 | 304.972 | 304.027 | 304.469 | 303.576 | — | 2.594 |
| cifar10 | pixel | mf_cifar10 | unknown@unknown | MeanFlowAlgorithm | 100 | 100 | 5000 | 94.418 | 85.936 | 87.868 | 89.565 | — | 4.151 |
| celeba_latent | latent | mf_distill_celeba_latent | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@69763e69e3ba | MeanFlowDistillAlgorithm | 100 | 100 | 5000 | 314.526 | 303.177 | 301.536 | 300.414 | — | 2.670 |
| cifar10 | pixel | mf_distill_cifar10 | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@2ea2a5e2039b | MeanFlowDistillAlgorithm | 100 | 100 | 5000 | 106.062 | 56.285 | 55.801 | 54.664 | — | 5.722 |
| cifar10 | pixel | mf_distill_cifar10 | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@3ee8b280c0db | MeanFlowDistillAlgorithm | 13 | — | unknown | — | — | — | — | — | — |
| cifar10 | pixel | mf_distill_cifar10 | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@c1f065aee704 | MeanFlowDistillAlgorithm | — | 100 | 5000 | 106.062 | 56.285 | 55.801 | 54.664 | — | 5.728 |
| celeba_latent | latent | mf_hutchinson_celeba_latent | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@7893916c5fb5 | MeanFlowHutchinsonAlgorithm | 2 | — | unknown | — | — | — | — | — | — |
| cifar10 | pixel | mf_v3_exact_jvp_b128_cifar10 | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@86185312c104 | MeanFlowAlgorithm | 100 | 100 | 5000 | 86.588 | 69.054 | 69.014 | 67.890 | — | 4.788 |
| cifar10 | pixel | mf_v3_exact_jvp_b128_probe_cifar10 | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@86185312c104 | MeanFlowAlgorithm | 10 | — | unknown | — | — | — | — | — | — |
| cifar10 | pixel | reflow_cifar10 | linux-ndag-m-lab-nvidia-geforce-rtx-3090-24gb@2ea2a5e2039b | ReflowAlgorithm | 100 | 100 | 5000 | 53.573 | 50.510 | 50.220 | 50.020 | — | 5.511 |

## Controlled comparison artifacts

The aggregate directory contains protocol-matched FM-relative gains, cross-dataset transfer checks, pixel-versus-latent pairs, Pareto flags, and an experiment-coverage matrix. Positive `fid_improvement_percent` means lower FID than FM. Missing peers remain explicit instead of being silently compared.

- Transfer groups available: 35
- Pixel/latent pairs available: 25
- Latent FID includes codec reconstruction error; decoder time and codec quality must be reported separately.

## Interactive checkpoint demo and external handoff

The local inference UI can play saved checkpoints in epoch order using one selected algorithm, NFE, image count, and fixed seed. It advances the loss curve with the checkpoint, shows a conceptual noise-to-sample transition, and decodes latent outputs through the recorded codec. Backbone and decoder timings are reported separately.

Build the verified Claude Web handoff with `./scripts/linux/make_thesis_context.sh`. The command rebuilds this summary, refreshes the training-log catalog, validates comparison tables and Git-tracked implementation files, then verifies every ZIP member against its SHA-256 digest.

## Training cost

| Experiment | Algorithm | Epochs | Training hours | Peak GPU memory (MiB) |
|---|---|---:|---:|---:|
| consistency_celeba_latent | ConsistencyAlgorithm | 100 | 4.32 | 2392.0 |
| consistency_cifar10 | ConsistencyAlgorithm | 100 | 1.88 | 4569.8 |
| fm_celeba | FlowMatchingAlgorithm | 100 | 16.15 | 5230.9 |
| fm_celeba_latent | FlowMatchingAlgorithm | 100 | 1.82 | 2219.8 |
| fm_cifar10 | FlowMatchingAlgorithm | 100 | 0.59 | 2250.9 |
| fm_lognorm_celeba | FlowMatchingLognormAlgorithm | 100 | 16.10 | 5226.6 |
| fm_lognorm_celeba_latent | FlowMatchingLognormAlgorithm | 100 | 1.81 | 2444.0 |
| fm_lognorm_cifar10 | FlowMatchingLognormAlgorithm | 100 | 0.59 | 2250.9 |
| mf_celeba_latent | MeanFlowAlgorithm | 100 | 2.82 | 2095.9 |
| mf_cifar10 | MeanFlowAlgorithm | 100 | 0.80 | 2207.8 |
| mf_distill_celeba_latent | MeanFlowDistillAlgorithm | 100 | 5.58 | 2376.9 |
| mf_distill_cifar10 | MeanFlowDistillAlgorithm | 100 | 3.09 | 3316.2 |
| mf_hutchinson_celeba_latent | MeanFlowHutchinsonAlgorithm | 2 | 0.08 | 624.0 |
| mf_v3_exact_jvp_b128_cifar10 | MeanFlowAlgorithm | 100 | 1.51 | 4737.0 |
| mf_v3_exact_jvp_b128_probe_cifar10 | MeanFlowAlgorithm | 10 | 0.15 | 3517.6 |
| reflow_cifar10 | ReflowAlgorithm | 100 | 0.75 | 3809.3 |
