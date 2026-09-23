# Consistency tuning harness

The tuning harness is ready, but no optimization run was started because model
training is explicitly reserved for the user. A dry run and a non-optimizing
forward/sample validation both pass.

## Variants

| Variant | EMA decay | Loss weight | Timesteps |
|---|---:|---:|---:|
| `baseline` | 0.999 | 1.0 | 18 |
| `lower_weight` | 0.999 | 0.5 | 18 |
| `faster_ema` | 0.995 | 1.0 | 18 |
| `finer_schedule` | 0.999 | 1.0 | 36 |

The variants isolate one change at a time. The default config remains unchanged
until empirical results exist; selecting a winner without training data would
not be defensible.

## Commands

Validate configuration, teacher availability, and planned variants without
training or creating outputs:

```bash
venv/bin/python scripts/tune_consistency.py --dry-run
```

Run the short CIFAR-10 sweep when training is authorized:

```bash
venv/bin/python scripts/tune_consistency.py --epochs 3 --batch-size 64
```

The harness acquires `results/.lock`, creates an isolated timestamped directory
under `results/consistency_tuning/`, disables evaluation during the short runs,
and writes `summary.json` plus `summary.md`. Its recommendation is the variant
with the lowest finite final-epoch loss. Check the full loss trajectory before
copying the recommended parameters into `config/consistency_full.json`.

## Mechanical validation completed

- Both CIFAR-10 teacher-dependent configs resolve to the existing epoch-100 FM
  checkpoint.
- Both CelebA configs use the matching `results/fm_celeba/...` convention.
- A Consistency training-step forward pass produced a finite scalar loss.
- NFE-1 Consistency sampling produced finite tensors with shape `(2, 3, 32, 32)`.
- No optimizer step or model-training run was performed.
