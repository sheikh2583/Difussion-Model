# Thesis Results Summary

Generated from the canonical experiment metric files by `scripts/aggregate_results.py`. Rerun the generator after active training finishes to produce the final snapshot.

## Evaluation metrics

| Dataset | Experiment | Algorithm | Train epoch | Eval epoch | Samples | FID@1 | FID@5 | FID@10 | FID@20 | FID@50 | IS@20 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | consistency_cifar10 | ConsistencyAlgorithm | 100 | 100 | 5000 | 130.968 | 100.845 | 95.879 | 99.635 | — | 4.333 |
| celeba | fm_celeba | FlowMatchingAlgorithm | 100 | 100 | 5000 | 223.126 | 44.357 | 25.975 | 19.808 | 16.779 | 2.410 |
| celeba | fm_celeba | FlowMatchingAlgorithm | 100 | 100 | 5000 | 223.126 | 44.357 | 25.975 | 19.808 | 16.779 | 2.410 |
| cifar10 | fm_cifar10 | FlowMatchingAlgorithm | 100 | — | unknown | — | — | — | — | — | — |
| cifar10 | fm_cifar10 | FlowMatchingAlgorithm | — | 100 | 5000 | 380.620 | 58.568 | 47.243 | 45.467 | 43.427 | 6.008 |
| celeba | fm_lognorm_celeba | FlowMatchingLognormAlgorithm | 98 | 80 | 5000 | 215.889 | 35.888 | 23.292 | 19.557 | 17.349 | 2.598 |
| celeba | fm_lognorm_celeba | FlowMatchingLognormAlgorithm | 92 | — | unknown | — | — | — | — | — | — |
| cifar10 | fm_lognorm_cifar10 | FlowMatchingLognormAlgorithm | 100 | — | unknown | — | — | — | — | — | — |
| cifar10 | fm_lognorm_cifar10 | FlowMatchingLognormAlgorithm | — | 100 | 5000 | 384.190 | 57.260 | 44.183 | 38.601 | 35.400 | 6.522 |
| cifar10 | mf_cifar10 | MeanFlowAlgorithm | 100 | 100 | 5000 | 94.418 | 85.936 | 87.868 | 89.565 | — | 4.151 |
| cifar10 | mf_distill_cifar10 | MeanFlowDistillAlgorithm | 100 | 100 | 5000 | 106.062 | 56.285 | 55.801 | 54.664 | — | 5.722 |
| cifar10 | mf_distill_cifar10 | MeanFlowDistillAlgorithm | 13 | — | unknown | — | — | — | — | — | — |
| cifar10 | mf_distill_cifar10 | MeanFlowDistillAlgorithm | — | 100 | 5000 | 106.062 | 56.285 | 55.801 | 54.664 | — | 5.728 |
| cifar10 | mf_v3_exact_jvp_b128_cifar10 | MeanFlowAlgorithm | 100 | 100 | 5000 | 86.588 | 69.054 | 69.014 | 67.890 | — | 4.788 |
| cifar10 | mf_v3_exact_jvp_b128_probe_cifar10 | MeanFlowAlgorithm | 10 | — | unknown | — | — | — | — | — | — |
| cifar10 | reflow_cifar10 | ReflowAlgorithm | 100 | 100 | 5000 | 53.573 | 50.510 | 50.220 | 50.020 | — | 5.511 |

## Training cost

| Experiment | Algorithm | Epochs | Training hours | Peak GPU memory (MiB) |
|---|---|---:|---:|---:|
| consistency_cifar10 | ConsistencyAlgorithm | 100 | 1.88 | 4569.8 |
| fm_celeba | FlowMatchingAlgorithm | 100 | 16.15 | 5230.9 |
| fm_cifar10 | FlowMatchingAlgorithm | 100 | 0.59 | 2250.9 |
| fm_lognorm_celeba | FlowMatchingLognormAlgorithm | 98 | 15.78 | 5226.6 |
| fm_lognorm_cifar10 | FlowMatchingLognormAlgorithm | 100 | 0.59 | 2250.9 |
| mf_cifar10 | MeanFlowAlgorithm | 100 | 0.80 | 2207.8 |
| mf_distill_cifar10 | MeanFlowDistillAlgorithm | 100 | 3.09 | 3316.2 |
| mf_v3_exact_jvp_b128_cifar10 | MeanFlowAlgorithm | 100 | 1.51 | 4737.0 |
| mf_v3_exact_jvp_b128_probe_cifar10 | MeanFlowAlgorithm | 10 | 0.15 | 3517.6 |
| reflow_cifar10 | ReflowAlgorithm | 100 | 0.75 | 3809.3 |
