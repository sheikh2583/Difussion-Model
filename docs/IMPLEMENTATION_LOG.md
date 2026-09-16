# Implementation Log
# "How Fast Can Generative Models Get, and What Does It Cost?"
# Last updated: scaffold pass — all phases planned, Phase 0–2 implemented

---

## How to read this file

- ✅ DONE — code exists, tested or ready to run
- 🔧 PARTIAL — scaffolded, needs completion noted inline
- ⏳ PENDING — not yet started, prerequisites listed
- ❌ BLOCKED — cannot start until dependency resolves

Each entry lists: what file changed, what it does, what to verify.

---

## Repository layout (after this zip is applied)

```
project/
├── algorithms/
│   ├── base.py                    [ORIGINAL — do not touch]
│   ├── r_embed.py                 ✅ shared r-conditioning module
│   ├── flow_matching.py           [ORIGINAL — do not touch]
│   ├── flow_matching_lognorm.py   [ORIGINAL — do not touch]
│   ├── mean_flow.py               ✅ updated: FD-JVP + stochastic routing
│   ├── mean_flow_distill.py       ✅ new: teacher-distillation MF
│   ├── consistency.py             🔧 scaffolded: needs loss tuning
│   ├── reflow.py                  🔧 scaffolded: needs dataset gen step
│   └── __init__.py                ✅ updated: all 6 algorithms registered
│
├── data/
│   ├── cifar10.py                 [ORIGINAL — do not touch]
│   ├── dataset_registry.py        ✅ new: routes dataset name → loader fn
│   └── celeba.py                  ✅ new: CelebA 64×64 pipeline
│
├── config/
│   ├── config.py                  ✅ updated: DatasetConfig gets scale_factor
│   ├── fm_full.json               ✅ updated
│   ├── fm_lognorm_full.json       ✅ updated
│   ├── mf_full.json               ✅ updated
│   ├── mf_distill_full.json       ✅ updated
│   ├── consistency_full.json      ✅ new
│   ├── reflow_full.json           ✅ new
│   ├── fm_celeba64.json           ✅ new: FM on CelebA 64×64
│   └── mf_celeba64.json           ✅ new: MF on CelebA 64×64
│
├── training/
│   └── trainer.py                 ✅ updated: on_epoch_end hook
│
├── evaluation/
│   └── evaluator.py               ✅ updated: make_plots param wired
│
├── docs/
│   ├── IMPLEMENTATION_LOG.md      ← this file
│   └── THEORY_NOTES.md            ← mathematical reference for thesis
│
├── train.py                       ✅ updated: pulls from ALGORITHM_REGISTRY
├── evaluate.py                    ✅ updated: pulls from ALGORITHM_REGISTRY
├── train_all.sh                   ✅ updated: all 6 algorithms + ordering
├── train_all.ps1                  ✅ updated: Windows equivalent
└── evaluate_all.sh                ✅ updated: all 6 algorithms
```

---

## Phase 0 — Dataset upgrade

### Status: ✅ DONE (CelebA 64×64)

**Files added/changed:**
- `data/celeba.py` — CelebA auto-download, center-crop 178→64, normalize [-1,1]
- `data/dataset_registry.py` — maps `cfg.dataset.name` → loader function
- `config/config.py` — DatasetConfig gets `scale_factor` field (default 1.0)
- `config/fm_celeba64.json` — FM config for CelebA 64×64
- `config/mf_celeba64.json` — MF config for CelebA 64×64

**To verify:**
```bash
python -c "
from data.dataset_registry import get_dataloaders_for_config
from config.config import ExperimentConfig
cfg = ExperimentConfig.load('config/fm_celeba64.json')
train, test = get_dataloaders_for_config(cfg)
batch, _ = next(iter(train))
print('CelebA batch shape:', batch.shape)   # expect (B, 3, 64, 64)
print('Value range:', batch.min().item(), batch.max().item())  # expect ~[-1, 1]
"
```

**Notes:**
- CelebA download requires ~1.4GB. First run will download automatically.
- If CelebA download fails (Google Drive quota), use manual download:
  https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html → img_align_celeba.zip
  Place in `./data/raw/celeba/`.
- BackboneConfig for 64×64 uses `channel_mults: [1, 2, 2, 2]` (one extra
  downsampling stage) — already set in celeba configs.

---

## Phase 1 — Re-baseline FM and MF on CelebA 64×64

### Status: ⏳ PENDING (needs Phase 0 verify + GPU time)

**Prerequisites:** Phase 0 verified, ~8–12h GPU time per run.

**Commands:**
```bash
python train.py --algorithm fm --config config/fm_celeba64.json
python train.py --algorithm mf --config config/mf_celeba64.json
```

**What to watch:**
- Loss should decrease smoothly within first 5 epochs. If NaN: lower lr to 1e-4.
- Peak GPU memory at batch_size=64, 64×64: ~6GB on RTX 3060. If OOM: reduce
  batch_size to 32 in config.
- FID at epoch 50 should be < 50 for FM (rough sanity check). If much higher:
  check normalization in celeba.py (must be [-1,1], not [0,1]).

**Team assignment:** Person A (infra/data-focused).

---

## Phase 2 — Mean Flow Distillation

### Status: ✅ DONE (code), ⏳ PENDING (training run)

**Prerequisites:** FM checkpoint from Phase 1 (or existing CIFAR-10 FM checkpoint
for a first smoke-test).

**File:** `algorithms/mean_flow_distill.py`

**Key design decisions recorded:**
- Teacher is frozen FM (no grad, eval mode, excluded from optimizer).
- `teacher_nfe=4` Euler steps per training step for target construction.
- `p_same=0.25` diagonal shortcut maintained for training stability.
- Student backbone: identical parameter count to FM (fair comparison for
  student-only cost; report teacher cost separately).

**Commands:**
```bash
# CIFAR-10 smoke-test (uses existing FM checkpoint)
python train.py --algorithm mf_distill --config config/mf_distill_full.json

# CelebA 64×64 (after Phase 1 FM checkpoint exists)
python train.py --algorithm mf_distill --config config/mf_distill_celeba64.json
```

**Team assignment:** Person C.

---

## Phase 3 — Consistency Distillation

### Status: 🔧 SCAFFOLDED (training loop complete, loss needs tuning)

**File:** `algorithms/consistency.py`

**What's implemented:**
- `ConsistencyAlgorithm(BaseAlgorithm)` with full `training_step` and `sample`.
- Self-consistency loss: MSE between `f(z_t, t)` and `f(z_{t-1}, t-1).detach()`
  where `z_{t-1}` is one Euler step from teacher FM.
- EMA target network (exponential moving average of student weights) for
  training stability — standard in CM implementations.
- `algorithm_kwargs`: `ema_decay`, `consistency_weight`, `teacher_checkpoint`.

**What needs tuning (known issues with CM training):**
- Loss scale is sensitive: if `consistency_weight` is too high relative to
  the FM-velocity auxiliary loss, training collapses. Start at 1.0, halve
  if loss explodes after epoch 3.
- EMA decay: 0.999 is standard but may be too slow at 32×32. Try 0.995 first.
- Expect 2–3 debugging cycles before loss curve looks healthy.

**Reference:** Song et al. 2023 — see THEORY_NOTES.md §4 for loss derivation.

**Team assignment:** Person B (math/paper-reading-focused). Read §3 of the
CM paper before touching this code.

**Commands:**
```bash
python train.py --algorithm consistency --config config/consistency_full.json
```

---

## Phase 4 — Rectified Flow Reflow

### Status: 🔧 SCAFFOLDED (algorithm complete, needs data-gen script)

**Files:**
- `algorithms/reflow.py` — ReflowAlgorithm: trains on (noise, image) pairs
  generated by a pre-trained FM model instead of random (noise, data) pairs.
- `scripts/generate_reflow_pairs.py` — ⏳ NOT YET WRITTEN. Must be created
  before training. See note below.

**What's implemented in reflow.py:**
- `ReflowAlgorithm(BaseAlgorithm)`: loads pre-generated pair dataset,
  standard FM loss on those pairs.
- `sample()`: identical Euler integration to FM.

**What's missing — generate_reflow_pairs.py:**
Must do: load FM checkpoint → sample noise z_1 ~ N(0,I) → run full Euler
integration → save (z_1, x_0_hat) pairs as .pt tensors to disk → ReflowAlgorithm
reads them as its training set.

Rough script structure needed:
```python
# scripts/generate_reflow_pairs.py
# python scripts/generate_reflow_pairs.py \
#   --checkpoint results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt \
#   --config config/fm_full.json \
#   --n-pairs 50000 \
#   --output data/reflow_pairs_cifar10.pt
```

**Team assignment:** Person C (natural extension of FM work from Phase 1).

**Commands (once generate script exists):**
```bash
python scripts/generate_reflow_pairs.py \
  --checkpoint results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt \
  --config config/fm_full.json --n-pairs 50000 \
  --output data/reflow_pairs_cifar10.pt

python train.py --algorithm reflow --config config/reflow_full.json
```

---

## Phase 5 — Full tournament + write-up

### Status: ⏳ PENDING (needs Phases 1–4 complete)

**When all checkpoints exist, run:**
```bash
./evaluate_all.sh   # generates per-algorithm plots
python scripts/aggregate_results.py  # ⏳ NOT YET WRITTEN — produces master chart
```

**Master chart needed:** FID vs NFE with all 6 algorithms on shared axes.
`plots.py` already supports overlaid multi-algorithm curves from a merged JSONL.
`aggregate_results.py` just needs to merge per-experiment JSONL files and call
the existing plot functions.

**Master table columns:**
| Algorithm | Backbone params | Extra params | Train time (h) | FID@1 | FID@5 | FID@20 |
(all already logged in results.jsonl — just needs aggregation)

---

## Known risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| CelebA download fails (GDrive quota) | Medium | Manual download link in Phase 0 notes |
| MF OOM at 64×64 batch_size=64 | High | config already sets batch_size=32 for MF celeba |
| Consistency loss collapses | Medium | EMA decay + weight tuning notes in Phase 3 |
| Reflow pairs generation takes too long | Low | 50k pairs at NFE=10 is ~30min on RTX 3060 |
| Phase 3 runs long → cuts into deadline | High | Explicitly marked as first cut if behind |

---

## Checkpoint naming convention

All checkpoints follow: `results/<experiment_name>/checkpoints/<ClassName>_epoch<N>.pt`

Current checkpoints (pre-existing, do not delete):
- `results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt`
- `results/fm_lognorm_rtx3060/checkpoints/FlowMatchingLognormAlgorithm_epoch100.pt`
- `results/mf_cifar10/checkpoints/MeanFlowAlgorithm_epoch30.pt`

---

## How to add a new algorithm in the future

1. Create `algorithms/<name>.py` implementing `BaseAlgorithm`.
2. Add one line to `algorithms/__init__.py` ALGORITHM_REGISTRY.
3. Add `config/<name>_full.json` (copy nearest config, change experiment_name).
4. Add one block to `train_all.sh` and `evaluate_all.sh`.
5. Document in this log under a new Phase entry.
That's it — Trainer, Evaluator, Sampler, and plots.py need zero changes.
