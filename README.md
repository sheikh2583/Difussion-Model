# Diffusion Models — Comparative Study
### Flow Matching · Mean Flow · Consistency · Reflow · CIFAR-10 & CelebA

A fully operational research scaffold for a controlled comparison of **6 generative
algorithms** on CIFAR-10 (32×32) and CelebA (64×64). Every pipeline component —
data, backbone, training, sampling, evaluation, checkpoint archiving, and browser
inference UI — is operational.

---

## Quick start (any PC, any OS)

```bash
# 1. Clone
git clone <repo-url>
cd DiffusionProject

# 2. One-shot setup — detects GPU, installs the right PyTorch, creates venv
python bootstrap.py           # interactive
python bootstrap.py --yes     # non-interactive (lab PC / CI)

# 3. Activate environment
source venv/bin/activate      # Linux / macOS
.\venv\Scripts\Activate.ps1   # Windows PowerShell

# 4. Quick smoke test (~30 s, CPU only)
python train.py --algorithm mock --config config/smoke_fast.json
```

> **`bootstrap.py`** is a pure-stdlib Python 3.9+ script — no shell extensions,
> works identically on Windows, Linux, and macOS.  It auto-detects CUDA 11.8,
> CUDA 12.1, ROCm, or CPU and installs the correct PyTorch wheel.

---

## Algorithm status

| Key | Algorithm | Paper |
|---|---|---|
| `fm` | Flow Matching (uniform-t) | Lipman et al. 2022 |
| `fm_lognorm` | Flow Matching (logit-normal t) | Esser et al. 2024 |
| `mf` | Mean Flow | Geng et al. 2025 |
| `mf_distill` | Mean Flow Distillation | Geng & Salimans |
| `consistency` | Consistency Models | Song et al. 2023 |
| `reflow` | Rectified Flow Reflow | Liu et al. 2022 §3 |

> **Fairness note** — `mf_distill` and `consistency` require a pre-trained FM
> teacher checkpoint.  Their results are not directly comparable to from-scratch
> runs; report teacher training cost separately.

---

## 1 · Training

### Train a single algorithm
```bash
python train.py --algorithm fm --config config/fm_full.json
python train.py --algorithm mf --config config/mf_full.json
python train.py --algorithm consistency --config config/consistency_full.json
python train.py --algorithm reflow --config config/reflow_full.json
```

### Train all 6 algorithms in dependency order
```bash
bash scripts/train_all.sh            # Linux / macOS
.\scripts\train_all.ps1              # Windows PowerShell

# Dataset flag (CelebA instead of CIFAR-10):
bash scripts/train_all.sh --dataset celeba

# Skip / isolate one algorithm:
bash scripts/train_all.sh --skip-reflow
bash scripts/train_all.sh --only mf_distill
```

### Override CLI flags
```bash
python train.py --algorithm fm --config config/fm_full.json \
    --experiment-name fm_ablation --epochs 50
```

### Convenience scripts
```bash
bash scripts/run_train.sh -a mf -c config/mf_full.json -e 200
.\scripts\run_train.ps1 -Algorithm mf -Config config\mf_full.json -Epochs 200
```

### Config presets

| File | Description |
|---|---|
| `config/smoke_fast.json` | 2-epoch CPU smoke test |
| `config/fm_full.json` | FM, 100 epochs, batch=128, CIFAR-10 |
| `config/fm_lognorm_full.json` | FM-LN, 100 epochs, batch=128 |
| `config/fm_lognorm_budget.json` | FM-LN, 100 epochs, batch=32 (limited VRAM) |
| `config/mf_full.json` | Mean Flow, 100 epochs, batch=32 |
| `config/mf_distill_full.json` | MF Distillation (needs FM checkpoint) |
| `config/consistency_full.json` | Consistency Models |
| `config/reflow_full.json` | Rectified Flow Reflow (needs reflow pairs) |
| `config/fm_celeba64.json` | FM on CelebA 64×64 |
| `config/mf_celeba64.json` | MF on CelebA 64×64 |

---

## 2 · Checkpoints & transfer

After every checkpoint epoch, the trainer automatically creates a self-contained
zip archive at:

```
results/<experiment>/checkpoints/archive/<AlgoClass>_epoch<N>.zip
```

**Zip contents:**
```
checkpoint.pt   ← model weights + optimizer state
config.json     ← full experiment config
meta.json       ← epoch, algorithm, dataset, UTC timestamp
```

To continue training on a lab PC, copy or download one of these zips — it
contains everything needed to resume or evaluate without the rest of the run dir.

### Reflow pairs (prerequisite for reflow)
```bash
python scripts/generate_reflow_pairs.py \
  --checkpoint results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt \
  --config config/fm_full.json \
  --n-pairs 50000 \
  --output data/reflow_pairs_cifar10.pt
```

---

## 3 · Evaluation

### Re-evaluate a checkpoint (without retraining)
```bash
python evaluate.py \
    --algorithm fm \
    --checkpoint results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt \
    --config results/fm_cifar10/config.json \
    --make-plots
```

### Evaluate all 6 algorithms
```bash
bash scripts/evaluate_all.sh
```

### Convenience script
```bash
bash scripts/run_evaluate.sh -a fm \
    -k results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt \
    -c results/fm_cifar10/config.json --make-plots
```

**Plots generated** (under `results/<experiment>/metrics/plots/`):
- `fid_vs_nfe.png` · `is_vs_nfe.png` · `fid_vs_sampling_time.png`
- `loss_vs_epoch.png` · `sampling_time_vs_nfe.png` · `gpu_memory.png`

---

## 4 · Checkpoint sample grids

Generate 8×8 image grids for every saved checkpoint (useful for tracking
visual quality across training epochs):

```bash
# Auto-discover all runs under results/:
python scripts/generate_checkpoint_samples.py

# Specific experiments and NFE values:
python scripts/generate_checkpoint_samples.py \
    --experiments fm_cifar10,mf_cifar10 --nfe 1,5,20

# Output root (default: results/checkpoint_samples/):
python scripts/generate_checkpoint_samples.py --out-dir ./my_grids
```

---

## 5 · Inference UI

```bash
python web/inference_server.py                     # localhost:8000
python web/inference_server.py --host 0.0.0.0 --port 9000  # LAN
python web/inference_server.py --self-test         # smoke test + exit

# Convenience scripts:
bash scripts/run_inference.sh
.\scripts\run_inference.ps1 -Port 9000
```

Open **http://127.0.0.1:8000** in a browser.

| URL | Page |
|---|---|
| `/` | Landing page |
| `/infer` | **Inference UI** — generate images interactively |
| `/train` | Training dashboard |

---

## 6 · Project layout

```
DiffusionProject/
├── bootstrap.py               ← Universal setup (run once after clone)
├── train.py                   ← CLI: train any algorithm
├── evaluate.py                ← CLI: evaluate a saved checkpoint
├── requirements.txt
│
├── algorithms/                ← Algorithm implementations
│   ├── base.py                    BaseAlgorithm interface
│   ├── flow_matching.py           FM (uniform-t)
│   ├── flow_matching_lognorm.py   FM (logit-normal t)
│   ├── mean_flow.py               Mean Flow (JVP)
│   ├── mean_flow_distill.py       MF Distillation
│   ├── consistency.py             Consistency Models
│   ├── reflow.py                  Rectified Flow Reflow
│   └── mock.py                    Smoke-test stub
│
├── config/                    ← JSON experiment presets
├── data/                      ← Dataset loaders (CIFAR-10, CelebA, registry)
├── models/                    ← SimpleUNet backbone
├── training/                  ← Trainer (epoch loop, AMP, checkpoint+zip)
├── sampling/                  ← Sampler
├── evaluation/                ← FID / IS evaluator
├── experiments/               ← ExperimentRunner (wires all components)
├── utils/                     ← Logging, plotting, timing, seeding
│
├── scripts/                   ← Convenience launch scripts
│   ├── train_all.sh / .ps1        Train all 6 algorithms
│   ├── evaluate_all.sh            Evaluate all checkpoints
│   ├── run_train.sh / .ps1        Train one algorithm
│   ├── run_evaluate.sh / .ps1     Evaluate one checkpoint
│   ├── run_inference.sh / .ps1    Start inference server
│   ├── generate_reflow_pairs.py   Generate reflow training pairs
│   ├── generate_checkpoint_samples.py  Sample grids from checkpoints
│   ├── animate_training_loss.py   Animate loss curves from JSONL
│   └── setup.sh / .ps1            → delegates to bootstrap.py
│
├── web/                       ← Browser inference UI
│   ├── inference_server.py        HTTP server (auto-discovers checkpoints)
│   ├── inference_ui.html
│   ├── thesis_dashboard.html
│   └── index.html
│
├── docs/                      ← Documentation & thesis assets
│   ├── IMPLEMENTATION_LOG.md      Phase status, verify commands, known risks
│   ├── THEORY_NOTES.md            Maths derivations + paper citations
│   ├── report_main.tex
│   └── slides.pptx / .pdf
│
├── results/                   ← Experiment outputs (large files gitignored)
│   └── <experiment>/
│       ├── config.json
│       ├── checkpoints/
│       │   ├── <AlgoClass>_epoch<N>.pt
│       │   └── archive/           ← Self-contained zips for transfer
│       │       └── <AlgoClass>_epoch<N>.zip
│       ├── samples/
│       ├── metrics/
│       └── logs/
│
└── archive/                   ← Zip snapshots (gitignored)
```

---

## Naming policy

- **No hardware names in configs or scripts.** Use `fm_lognorm_budget` (intent)
  not `fm_lognorm_rtx3060` (machine).
- **Experiment names** follow `<algorithm>_<dataset>`, e.g. `fm_cifar10`, `mf_celeba64`.
- **Config files** follow `<algorithm>_<variant>.json`, e.g. `fm_lognorm_full.json`,
  `fm_lognorm_budget.json`.

---

## Fairness constraints (enforced at runtime)

- **Single backbone**: all algorithms share identical architecture from the same `BackboneConfig`.
- **Single dataloader**: `data/dataset_registry` routes to the correct loader; identical preprocessing for every algorithm.
- **`algorithm_kwargs` is guarded**: `experiments/runner.py` raises at runtime if any key shadows a shared experimental control (batch size, epochs, optimizer, etc.).
- **Narrow algorithm interface**: only `training_step(batch)` and `sample(n, nfe, device)`.
- **Wall-clock timing**: CUDA-synchronized timing and peak memory are logged for every epoch and sample call.
- **Shared FID reference**: computed once from real data, reused for all algorithms.
