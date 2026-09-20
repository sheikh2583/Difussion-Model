# Linux Verification Checklist

> **Purpose:** Step-by-step checklist to verify the full DiffusionProject
> workflow on a fresh Linux machine. Run once on the lab PC before starting
> real training runs. Check each box as you go.

---

## Prerequisites

- [ ] Python 3.9+ installed (`python3 --version`)
- [ ] Git installed (`git --version`)
- [ ] NVIDIA driver installed if using CUDA (`nvidia-smi`)
- [ ] `zip` utility installed (`zip --version`) — needed for checkpoint archives
- [ ] Enough disk space: ~2 GB for deps + 170 MB CIFAR-10 + 1.4 GB CelebA

---

## Step 1 — Clone and bootstrap

```bash
git clone https://github.com/sheikh2583/Difussion-Model.git DiffusionProject
cd DiffusionProject

# Auto-detect GPU, create venv, install deps, download CIFAR-10
python3 bootstrap.py --yes

# Or with both datasets:
python3 bootstrap.py --yes --datasets all

# Activate environment (required for all commands below)
source venv/bin/activate
```

**Verify:**
- [ ] `venv/` directory created
- [ ] `python -c "import torch; print(torch.__version__)"` prints a version
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` prints `True` on GPU machines
- [ ] `data/raw/cifar-10-batches-py/` exists (CIFAR-10 downloaded)

---

## Step 2 — Workflow verification

```bash
python scripts/verify_workflow.py --dataset cifar10
```

**Expected output (all lines should say `[OK]`):**
```
[OK] algorithms: ['consistency', 'fm', 'fm_lognorm', 'mf', 'mf_distill', 'reflow']
[OK] additional utility algorithms: ['mock']
[OK] dataset registry: cifar10, celeba
[OK] config: config/fm_full.json
... (one line per config)
[OK] cifar10 batch: shape=(32, 3, 32, 32), range=[-1.000, 1.000]
Workflow code and configuration checks passed.
```

- [ ] All lines print `[OK]`
- [ ] No import errors

> **Known risk:** If `torch.func.jvp` is unavailable (PyTorch < 2.0), MeanFlow
> will fail. The bootstrap installs `torch>=2.0` but double-check with
> `python -c "from torch.func import jvp; print('jvp OK')"`.

---

## Step 3 — Smoke test (CPU, ~30 seconds)

```bash
python train.py --algorithm mock --config config/smoke_fast.json
```

**Expected:** Training completes, output ends with something like:
```
epoch=1 loss=... time=...
epoch=2 loss=... time=...
training complete: total_time=...
```

**Verify:**
- [ ] `results/smoke_cifar10/` directory created
- [ ] `results/smoke_cifar10/checkpoints/MockAlgorithm_epoch2.pt` exists
- [ ] `results/smoke_cifar10/metrics/smoke.jsonl` exists and is non-empty

---

## Step 4 — CelebA loader check (read-only, no training)

```bash
# Only needed if CelebA was downloaded
python -c "
from config.config import ExperimentConfig
from data.dataset_registry import get_dataloaders_for_config
cfg = ExperimentConfig.load('config/fm_celeba64.json')
cfg.batch_size = 4
train_loader, _ = get_dataloaders_for_config(cfg)
batch = next(iter(train_loader))
imgs = batch[0] if isinstance(batch, (list,tuple)) else batch
print(f'Shape: {tuple(imgs.shape)}, min={imgs.min():.3f}, max={imgs.max():.3f}')
assert imgs.shape[1:] == (3, 64, 64), 'Shape wrong'
assert imgs.min() >= -1.1 and imgs.max() <= 1.1, 'Normalization wrong'
print('[PASS] CelebA OK')
"
```

- [ ] Prints `[PASS] CelebA OK`

---

## Step 5 — Aggregation pipeline

```bash
python scripts/aggregate_results.py
```

- [ ] Prints `Aggregated N records from M files.` (N>0 if any runs exist)
- [ ] `results/aggregate/combined_metrics.jsonl` created
- [ ] `results/aggregate/summary.csv` created

---

## Step 6 — Results browser and inference server

```bash
python web/inference_server.py --self-test   # exits after smoke test
# OR
python web/inference_server.py               # stays running, open http://127.0.0.1:8000
```

- [ ] Server starts without import errors
- [ ] Results page lists runs using config-derived algorithm and dataset names
- [ ] `--self-test` prints one JSON object for each discovered checkpoint run

---

## Known Linux-specific risks

| Risk | File(s) affected | Mitigation |
|---|---|---|
| Path separator `\` vs `/` | `scripts/*.sh`, `bootstrap.py` | All scripts use `Path()` or POSIX strings — should be safe, but check if any `os.path.join` with hardcoded `\` leaks |
| `zip` not installed | `training/trainer.py` `_zip_checkpoint()` | Run `sudo apt install zip` before training |
| CelebA Google Drive quota blocks download | `bootstrap.py` | Pre-download zip from Drive manually → place at `data/raw/celeba/img_align_celeba.zip` → rerun bootstrap |
| `nvidia-smi` not in PATH | `bootstrap.py` | Install NVIDIA driver properly; bootstrap falls back to CPU wheels if missing |
| `nvcc` not installed | `bootstrap.py` | Not required — bootstrap uses driver version to select wheels |
| `torch.func.jvp` missing (PyTorch < 2.0) | `algorithms/mean_flow.py` | `bootstrap.py` installs `torch>=2.0`; verify manually if using system Python |
| Workers > 0 hang on some Linux setups | `config/*.json` `dataset.num_workers` | Set to `0` in config if DataLoader hangs indefinitely |

---

## Verification log

| Date | Machine | OS | CUDA | Result | Notes |
|---|---|---|---|---|---|
| 2026-09-17 | Dev PC (Windows) | Windows 11 | 12.x | PASS | All steps verified |
| _(add row when run on Linux lab PC)_ | | | | | |
