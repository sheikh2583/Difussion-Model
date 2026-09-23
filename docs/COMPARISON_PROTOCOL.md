# Controlled comparison protocol

This project answers three different questions. They must be reported as
separate comparisons rather than collapsed into one leaderboard.

## 1. Algorithm progression

Within one dataset and representation, compare FM, FM-LN, Mean Flow,
MF-Distill, Consistency, and Reflow at matched epoch, NFE, seed, generated
sample count, machine, and backbone signature. Source identity is also checked:
new controlled suites should match it, while historical mismatches are retained
with `compatible_provenance_difference` rather than hidden. FM is the baseline.
The aggregate reports

```text
FID improvement (%) = 100 × (FM FID − method FID) / FM FID
```

Positive values are improvements. Training loss is diagnostic only because
the algorithms optimize different objectives and their raw loss scales are not
generally comparable.

MF-Hutchinson is a seventh, latent-only estimator diagnostic. Report it as an
ablation against latent MF; do not silently add it to the canonical six-method
cross-dataset leaderboard because no corresponding pixel run exists.

## 2. Dataset transfer

First establish each method's improvement relative to FM on CIFAR-10. Repeat
the controlled pixel experiment on CelebA and report whether the sign and
magnitude of that improvement transfer. A single seed is descriptive evidence,
not proof of consistency. The final claim should use at least three independent
training seeds and report uncertainty for every dataset-method-NFE cell.

`results/aggregate/cross_dataset_consistency.csv` records
`awaiting_second_dataset`, `consistent_improvement`,
`consistent_regression`, or `mixed_direction`. This makes missing experiments
visible instead of treating them as supporting evidence.

## 3. Pixel-to-latent architecture migration

Compare pixel and latent runs only when the semantic dataset, algorithm,
epoch, NFE, seed, and generated sample count match. New controlled runs should
also use the same machine label and source identity. Historical pairs with a
label or source mismatch remain available, but are explicitly marked
`paired_with_provenance_difference`. Report:

- FID and IS on decoded RGB images against the same raw-image validation split;
- generative-backbone time and decoder time separately;
- end-to-end sampling time, throughput, peak GPU memory, and training hours;
- state shape and state-value reduction;
- codec reconstruction metrics and exact codec identity.

Latent FID includes the codec's reconstruction ceiling. It is not a pure
measurement of the generative backbone. Pixel and latent results therefore
remain separate rankings, with paired deltas in
`results/aggregate/pixel_vs_latent.csv`.

A CIFAR-10 latent experiment requires a codec validated for CIFAR-10. The
accepted face-specific CelebA VQ-f4 codec must not be reused for CIFAR-10.
Freeze the CIFAR codec, posterior rule, scaling, normalization statistics,
content digest, cache identity, and reconstruction gate before comparing the
two representations. Preserve the algorithm and optimizer settings wherever
the tensor-shape change permits; disclose every unavoidable architecture
change.

## Aggregate outputs

Running `scripts/aggregate_results.py` or `scripts/linux/make_summary.sh`
produces:

| Artifact | Purpose |
|---|---|
| `summary.csv` | One wide row per run identity, including representation and state shape |
| `comparison_coverage.csv` | Completed training/evaluation cells |
| `algorithm_progression_vs_fm.csv` | Protocol-matched FM-relative deltas |
| `cross_dataset_consistency.csv` | Direction and magnitude of transfer across datasets |
| `pixel_vs_latent.csv` | Paired representation deltas and codec warning |
| `quality_compute_pareto.csv` | FID–NFE and FID–latency Pareto membership |
| `comparison_manifest.json` | Matching rules, metric directions, and missing modern metrics |

## Modern evaluation additions

FID and IS are retained for continuity with the completed experiments, but
they should not be the only evidence. The manifest marks unmeasured metrics as
not implemented rather than emitting empty or inferred values.

Recommended additions, after they are implemented and frozen across all runs:

- CMMD, which uses CLIP embeddings and an unbiased MMD estimator, as proposed
  in [Rethinking FID (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Jayasumana_Rethinking_FID_Towards_a_Better_Evaluation_Metric_for_Image_Generation_CVPR_2024_paper.html).
- KID plus precision, recall, density, and coverage to separate fidelity from
  mode coverage. Alternative self-supervised encoders such as DINOv2 should be
  reported explicitly rather than silently substituted; see
  [Exposing flaws of generative model evaluation metrics](https://openreview.net/forum?id=08zf7kTOoh).
- A dedicated diversity measurement. IRS is one recent option described in
  [Image Generation Diversity Issues and How to Tame Them](https://arxiv.org/abs/2411.16171).
- Confidence intervals across independent training seeds and a robustness
  check showing whether method rankings change with metric, NFE, or dataset.

Do not add a metric to the headline table until its feature extractor,
preprocessing, reference split, sample count, package revision, and direction
of improvement are recorded.
