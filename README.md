# Flow Matching vs Mean Flow — CIFAR-10 Comparative Study

A fully functional research scaffold for a controlled comparison between
**Flow Matching (FM)**, **Flow Matching + Logit-Normal sampling (FM-LN)**, and
**Mean Flow (MF)** on unconditional CIFAR-10 generation.  Every pipeline
component — data, backbone, training, sampling, evaluation, results logging,
and the inference UI — is operational.

---

## Status

| Component | Status | Notes |
|---|---|---|
| Config system | ✅ functional | JSON presets in `config/` |
| CIFAR-10 pipeline | ✅ functional | auto-downloads on first run |
| SimpleUNet backbone | ✅ functional | 6,352,899 parameters (full config) |
| MockAlgorithm | ✅ functional | smoke-test only, trivial L2 loss |
| FlowMatchingAlgorithm | ✅ functional | trained 100 epochs (`results/fm_cifar10/`) |
| FlowMatchingLognormAlgorithm | ✅ functional | trained 100 epochs (`results/fm_lognorm_rtx3060/`) |
| MeanFlowAlgorithm | ✅ functional | trained 30 epochs (`results/mf_cifar10/`) |
| Trainer | ✅ functional | AMP, checkpointing, JSONL logging |
| Sampler | ✅ functional | per-NFE timing + grid images |
| Evaluator (FID / IS) | ✅ functional | shared reference cache |
| Inference UI server | ✅ functional | browser UI at `http://127.0.0.1:8000` |
| Plots | ✅ functional | `utils/plots.py` + `evaluate.py --make-plots` |

---

## Setup

You can automatically set up the Python virtual environment, install dependencies, and create necessary workspace folders by running the provided setup scripts:

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**Linux / macOS:**
```bash
./setup.sh
```

*(Optional)* Run with the `-DownloadAssets` (Windows) or `--download-assets` (Linux/Mac) flag if you want to automatically download pre-trained checkpoints and datasets (URLs must be configured inside the script first).

Once setup completes, ensure your virtual environment is active for all subsequent commands:
- **Windows:** `.\venv\Scripts\Activate.ps1`
- **Linux/macOS:** `source venv/bin/activate`

---

## 1. Quick Smoke Test (no GPU required — ~30 s on CPU)

Runs the complete pipeline (dataset → backbone → train → checkpoint →
sample → FID/IS evaluate → JSONL results) using `MockAlgorithm` on a tiny
configuration. No real generative learning occurs — this just verifies every
interface works end-to-end.

```bash
python train.py --algorithm mock --config config/smoke_fast.json
```

**Expected output:**
```
epoch=1 loss=0.XXXXXX time=X.XXXs peak_mem=0.0MB
epoch=2 loss=0.XXXXXX time=X.XXXs peak_mem=0.0MB
training complete: total_time=X.XXXs
```

**Output files produced:**
```
results/smoke_fast/
├── config.json                          # saved experiment config
├── checkpoints/
│   ├── MockAlgorithm_epoch1.pt
│   └── MockAlgorithm_epoch2.pt
├── samples/
│   ├── MockAlgorithm_nfe2.png           # 64-image grid
│   └── MockAlgorithm_nfe5.png
├── logs/
│   └── trainer.MockAlgorithm.log
└── metrics/
    └── smoke_fast.jsonl                 # machine-readable results
```

---

## 2. Training

### Flow Matching (100 epochs, batch=128, AMP)
```bash
python train.py --algorithm fm --config config/fm_full.json
```
Outputs to `results/fm_cifar10/`. Checkpoints every 10 epochs.
Evaluation at epoch 50 and 100 (NFE 1, 5, 10, 20).

### Flow Matching + Logit-Normal time sampling (100 epochs, batch=32, RTX 3060)
```bash
python train.py --algorithm fm_lognorm --config config/fm_lognorm_rtx3060.json
```
Outputs to `results/fm_lognorm_rtx3060/`.
Uses logit-normal t-sampling (Esser et al. 2024) instead of uniform.

### Mean Flow (30 epochs, batch=32)
```bash
python train.py --algorithm mf --config config/mf_full.json
```
Outputs to `results/mf_cifar10/`. One-step generation via displacement identity.

### Optional overrides
```bash
# Override epoch count or experiment name at the CLI:
python train.py --algorithm fm --config config/fm_full.json \
    --experiment-name fm_run2 --epochs 50
```

### Windows convenience script
```powershell
.\scripts\run_train.ps1 -Algorithm fm -Config config\fm_full.json
.\scripts\run_train.ps1 -Algorithm mf -Config config\mf_full.json -Epochs 50
.\scripts\run_train.ps1 -Algorithm mock   # smoke test, no config needed
```

---

## 3. Evaluation (checkpoint re-evaluation without retraining)

Use `evaluate.py` to run sampling + FID/IS on an existing checkpoint.
Useful for re-evaluating at different NFE values or regenerating plots.

### Flow Matching — epoch 100
```bash
python evaluate.py \
    --algorithm fm \
    --checkpoint results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt \
    --config     results/fm_cifar10/config.json
```

### Flow Matching + Logit-Normal — epoch 100
```bash
python evaluate.py \
    --algorithm fm_lognorm \
    --checkpoint results/fm_lognorm_rtx3060/checkpoints/FlowMatchingLognormAlgorithm_epoch100.pt \
    --config     results/fm_lognorm_rtx3060/config.json
```

### Mean Flow — epoch 30
```bash
python evaluate.py \
    --algorithm mf \
    --checkpoint results/mf_cifar10/checkpoints/MeanFlowAlgorithm_epoch30.pt \
    --config     results/mf_cifar10/config.json
```

### With plots
Add `--make-plots` to any of the above. Plots are saved under
`results/<experiment>/metrics/plots/`:

```bash
python evaluate.py \
    --algorithm fm \
    --checkpoint results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt \
    --config     results/fm_cifar10/config.json \
    --make-plots
```

**Plots generated:**
```
results/fm_cifar10/metrics/plots/
├── loss_vs_epoch.png
├── training_time.png
├── sampling_time_vs_nfe.png
├── fid_vs_nfe.png
├── is_vs_nfe.png
├── gpu_memory.png
└── fid_vs_sampling_time.png
```

### Windows convenience script
```powershell
.\scripts\run_evaluate.ps1 `
    -Algorithm fm `
    -Checkpoint results\fm_cifar10\checkpoints\FlowMatchingAlgorithm_epoch100.pt `
    -Config     results\fm_cifar10\config.json `
    -MakePlots
```

---

## 4. Inference UI (interactive image generation in the browser)

The inference server auto-discovers all trained checkpoints and serves a
browser UI for interactive image generation.

### Start the server
```bash
python inference_server.py
```

Then open **http://127.0.0.1:8000** in a browser.

**Available UI pages:**

| URL | Page |
|---|---|
| `http://127.0.0.1:8000/` | Landing page |
| `http://127.0.0.1:8000/infer` | **Inference UI** — generate images |
| `http://127.0.0.1:8000/train` | Training dashboard |

### Server options
```bash
# Custom host/port:
python inference_server.py --host 0.0.0.0 --port 9000

# Point at a non-default results directory:
python inference_server.py --results-dir /path/to/other/results

# Smoke test (generate 1 image per available model then exit):
python inference_server.py --self-test
```

**Expected self-test output:**
```json
{
  "model": "fm",
  "model_label": "Flow Matching",
  "checkpoint_epoch": 100,
  "nfe": 1,
  "device": "cuda",
  "checkpoint_load_seconds": X.XXXX,
  "generation_seconds": X.XXXX,
  "images_per_second": X.XXX
}
```

### API endpoints (used by the UI internally)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/models` | JSON catalog of all models + checkpoint epochs |
| `POST` | `/api/generate` | Generate images; body: `{"model","checkpoint_epoch","nfe","num_images","seed"}` |

### Windows convenience script
```powershell
.\scripts\run_inference.ps1                          # defaults
.\scripts\run_inference.ps1 -Port 9000               # custom port
.\scripts\run_inference.ps1 -SelfTest                # smoke test
.\scripts\run_inference.ps1 -ResultsDir D:\models    # custom results dir
```

---

## 5. Trained Model Catalogue

Checkpoints already present in `results/`:

| Model key | Experiment dir | Epochs | Checkpoint files |
|---|---|---|---|
| `fm` | `results/fm_cifar10/` | 100 | epoch 10, 20, …, 100 |
| `fm_lognorm` | `results/fm_lognorm_rtx3060/` | 100 | epoch 1, 2, 3, 10, 20, …, 100 |
| `mf` | `results/mf_cifar10/` | 30 | epoch 10, 20, 30 |

---

## 6. Results & Metrics

All training, sampling and evaluation records are appended in real-time
to JSONL files under `results/<experiment>/metrics/`:

```
results/fm_cifar10/metrics/fm_cifar10.jsonl     # 127 records
results/fm_lognorm_rtx3060/metrics/...jsonl
results/mf_cifar10/metrics/...jsonl
```

Each line is a flat JSON record with fields:
`algorithm`, `seed`, `record_type` (`train_epoch` | `sampling` | `evaluation`),
`epoch`, `loss`, `training_time`, `nfe`, `fid`, `is_mean`, `is_std`,
`sampling_time`, `peak_gpu_memory`, etc.

---

## Project Layout

```
DiffusionProject/
├── train.py                    # CLI: train → checkpoint → sample → evaluate
├── evaluate.py                 # CLI: re-evaluate a saved checkpoint
├── inference_server.py         # HTTP server + browser inference UI
│
├── config/                     # JSON experiment config presets
│   ├── config.py               # ExperimentConfig dataclass
│   ├── smoke_fast.json         # tiny 2-epoch CPU smoke test
│   ├── fm_full.json            # FM, 100 epochs, batch=128
│   ├── fm_lognorm.json         # FM-LN, 100 epochs, batch=128
│   ├── fm_lognorm_rtx3060.json # FM-LN, 100 epochs, batch=32 (VRAM tuned)
│   └── mf_full.json            # MF, 30 epochs, batch=32
│
├── algorithms/                 # Algorithm implementations (BaseAlgorithm subclasses)
│   ├── base.py                 # Abstract interface: training_step + sample
│   ├── flow_matching.py        # FM: linear path, uniform-t, Euler ODE
│   ├── flow_matching_lognorm.py# FM-LN: same + logit-normal time sampling
│   ├── mean_flow.py            # MF: mean velocity, JVP, displacement identity
│   └── mock.py                 # Trivial smoke-test algorithm
│
├── models/
│   └── backbone.py             # SimpleUNet + build_backbone + count_parameters
│
├── training/
│   └── trainer.py              # Generic training engine (no algorithm math)
│
├── sampling/
│   └── sampler.py              # Generic sampling engine (no algorithm math)
│
├── evaluation/
│   ├── evaluator.py            # Evaluator + ensure_fid_reference
│   └── metrics.py              # compute_fid + compute_inception_score
│
├── experiments/
│   └── runner.py               # ExperimentRunner: wires all components
│
├── data/
│   └── cifar10.py              # CIFAR-10 dataloaders (shared by all algorithms)
│
├── utils/
│   ├── device.py               # resolve_device — CUDA fallback logic
│   ├── logging.py              # setup_logger + JsonlLogger
│   ├── plots.py                # Matplotlib plotting from JSONL results
│   ├── results.py              # ResultRecord schema + ResultsWriter
│   ├── seed.py                 # set_seed + set_deterministic
│   └── timing.py               # timer context manager + GPU memory helpers
│
├── scripts/                    # Convenience launch scripts
│   ├── run_train.ps1 / .sh     # Train any algorithm
│   ├── run_evaluate.ps1 / .sh  # Evaluate a checkpoint
│   └── run_inference.ps1 / .sh # Start the inference UI server
│
├── results/                    # Experiment outputs (gitignored large files)
│   ├── fm_cifar10/             # Flow Matching run (100 epochs)
│   ├── fm_lognorm_rtx3060/     # FM + Logit-Normal run (100 epochs)
│   └── mf_cifar10/             # Mean Flow run (30 epochs)
│
├── requirements.txt            # Minimum version pins
├── requirements_frozen.txt     # Exact reproducible pins (thesis environment)
└── .env                        # Developer setup notes (venv, PYTHONPATH, CUDA)
```

---

## Fairness Constraints (enforced by construction)

- **Single backbone constructor** (`models/backbone.build_backbone`): all algorithms share identical architecture and parameter count from the same `BackboneConfig`.
- **Single dataloader constructor** (`data/cifar10.get_dataloaders`): identical preprocessing and seeded shuffling for every algorithm.
- **`algorithm_kwargs` is guarded**: `experiments/runner.py` raises at runtime if any key in `algorithm_kwargs` shadows a shared experimental control (batch size, epochs, optimizer, lr, scheduler, AMP, backbone, dataset, seed, evaluation settings).
- **Narrow algorithm interface**: `BaseAlgorithm` exposes only `training_step(batch)` and `sample(n, nfe, device)`.  No `prepare_batch` hook exists — algorithm-specific preprocessing is contained entirely within the algorithm.
- **Wall-clock, not step-count**: Trainer and Sampler record synchronized CUDA timing and peak memory. NFE is logged but never assumed to represent equal computation across algorithms.
- **Shared FID reference**: `evaluation.ensure_fid_reference` computes reference statistics once from real CIFAR-10 data and reuses them for every algorithm's evaluation run.
