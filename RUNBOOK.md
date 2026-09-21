# Latent CelebA operator runbook

Run every command from the repository root. Stop at the first failed command.
Do not resume the rejected scratch VAE or any pixel-space CelebA run.

## Phase 0 — Environment check (CPU)

This confirms that the pretrained-codec loader and CelebA pipeline dependencies
are importable and prints their installed versions.

```bash
venv/bin/python -c "import diffusers, huggingface_hub, torch, torchvision; print('diffusers', diffusers.__version__); print('huggingface_hub', huggingface_hub.__version__); print('torch', torch.__version__); print('torchvision', torchvision.__version__)"
```

If an import fails, install the declared environment and repeat the check:

```bash1
venv/bin/pip install -r requirements.txt
```

Proceed only when all four imports and version lines succeed.

## Phase 1 — Download face-specific pretrained VQ-f4 weights (CPU/network)

This resolves `main` to an immutable Hugging Face commit, downloads only the
frozen `vqvae` component from `CompVis/ldm-celebahq-256`, and records the
resolved revision. The previously tested SD VAE is rejected evidence and is
not reused.

```bash
venv/bin/python codec/download_pretrained_vq.py \
  --repo-id CompVis/ldm-celebahq-256 \
  --revision main \
  --output-dir ./data/pretrained/ldm-celebahq-256
```

Expected files:

```bash
test -f ./data/pretrained/ldm-celebahq-256/vqvae/source_manifest.json
test -f ./data/pretrained/ldm-celebahq-256/vqvae/config.json
test -f ./data/pretrained/ldm-celebahq-256/vqvae/diffusion_pytorch_model.bin
venv/bin/python -m json.tool ./data/pretrained/ldm-celebahq-256/vqvae/source_manifest.json
```

Visually verify that `repo_id` is `CompVis/ldm-celebahq-256`, `subfolder` is
`vqvae`, and
`resolved_revision` is a non-empty commit hash. If download fails, check network
access and Hugging Face availability, then rerun the same command; the snapshot
downloader resumes cached files.

## Phase 2 — Validate codec and freeze training statistics (CUDA)

This loads the frozen VAE, computes native latent statistics from the complete
CelebA training split, and evaluates reconstruction FID and PSNR on 5,000
validation images.

```bash
venv/bin/python codec/validate_codec.py \
  --codec-source CompVis/ldm-celebahq-256 \
  --codec-source-path ./data/pretrained/ldm-celebahq-256/vqvae \
  --codec-source-revision auto \
  --celeba-root ./data/raw \
  --output-dir ./results/codecs/celeba_vq_f4 \
  --batch-size 32 \
  --n-val-images 5000 \
  --num-workers 4 \
  --device cuda \
  --accept-quality-failure \
  --acceptance-reason "User selected the face-specific VQ-f4 latent experiment despite the measured reconstruction ceiling"
```

Expected outputs:

- `./results/codecs/celeba_vq_f4/accepted_codec.pt` — after all structural
  checks pass and either the quality gates pass or the explicit override is recorded.
- `./results/codecs/celeba_vq_f4/validation_report.json` — metrics and identity.
- `./results/codecs/celeba_vq_f4/reconstruction_grid.png` — originals/reconstructions.

Required project gates are reconstruction FID `< 5`, PSNR `> 30 dB`, and
minimum native latent channel standard deviation `> 0.1`. Open
`results/codecs/celeba_vq_f4/reconstruction_grid.png` and verify that identities, eyes, skin boundaries,
and hair structure remain recognizable without severe blur or color shifts.
Also inspect the report:

```bash
venv/bin/python -m json.tool ./results/codecs/celeba_vq_f4/validation_report.json
```

For this selected experiment, the measured rFID/PSNR shortfall is accepted as
a documented codec limitation. Continue only if the command prints either
`ALL QUALITY GATES PASSED — codec accepted` or
`QUALITY GATE OVERRIDDEN — codec accepted for latent training`, the report has
`accepted_for_latent_training: true`, and `accepted_codec.pt` exists. The
report must retain the measured failures and `quality_gate_passed: false` when
the override is used. Structural, finite-value, and minimum-latent-std failures
remain hard stops. The rejected scratch VAE is not a fallback.

## Phase 3 — Cache normalized latents (CUDA)

This deterministically encodes and codebook-quantizes both CelebA splits,
normalizes them using the frozen training statistics, and atomically publishes
a content-addressed cache.

```bash
venv/bin/python codec/cache_latents.py \
  --codec-path ./results/codecs/celeba_vq_f4/accepted_codec.pt \
  --celeba-root ./data/raw \
  --output-dir ./data/latent_cache \
  --split both \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda
```

The command prints the exact final directory, for example
`./data/latent_cache/latents_<hash>`, containing:

- `latents_train.pt`
- `latents_valid.pt`
- `manifest.json`

Set the printed directory exactly, then verify hashes/metadata by rerunning the
same cache command (it should print `Existing cache verified — nothing to do`).
Check the normalized training tensor statistics explicitly:

```bash
CACHE_DIR=./data/latent_cache/latents_<hash>
venv/bin/python -c "import sys,torch; z=torch.load(sys.argv[1]+'/latents_train.pt',map_location='cpu',weights_only=True); m=z.mean((0,2,3)); s=z.std((0,2,3),unbiased=False); print('shape',tuple(z.shape)); print('mean',m.tolist()); print('std',s.tolist()); assert tuple(z.shape[1:])==(3,16,16); assert bool((m.abs()<=0.1).all()); assert bool(((s>=0.8)&(s<=1.2)).all())" "$CACHE_DIR"
```

Proceed only if the shape is `(N,3,16,16)`, every mean is within ±0.1, and every
standard deviation is in `[0.8,1.2]`. On failure, do not hand-edit cache files;
inspect the manifest/checkpoint identity and rerun caching after correcting the
underlying input.

## Phase 4 — Confirm checkpoint paths

The six latent configs have been aligned with Phase 2's output. Verify all six:

```bash
rg 'codec_checkpoint' config/*_celeba_latent.json
```

Every line must contain
`./results/codecs/celeba_vq_f4/accepted_codec.pt`. Do not proceed if any latent
config still names the rejected SD codec checkpoint.

## Phase 5 — Smoke tests

The first command is the synthetic CPU-safe contract test:

```bash
venv/bin/python scripts/smoke_latent.py --mode static
```

Expected final line: `ALL STATIC CHECKS PASSED`.

Then use the exact content-addressed directory printed in Phase 3 for the full
CUDA boundary test:

```bash
venv/bin/python scripts/smoke_latent.py --mode full \
  --codec-path ./results/codecs/celeba_vq_f4/accepted_codec.pt \
  --latent-cache-dir "$CACHE_DIR" \
  --celeba-root ./data/raw \
  --device cuda
```

Expected final line: `ALL FULL SMOKE CHECKS PASSED`. This is the live script's
actual success text; it does not print `OVERALL: PASS`. Stop and resolve every
reported failure before training.

## Phase 6 — Training order (one GPU job at a time)

Use explicit fresh-run lifecycle control. Start with FM and monitor at least the
first five epochs for finite, decreasing loss and acceptable GPU memory. The
initial planning estimate is 10–14 hours for 100 FM epochs on an RTX 3090, but
measure the first epochs and revise the estimate from observed throughput.

```bash
venv/bin/python train.py --algorithm fm \
  --config config/fm_celeba_latent.json --mode fresh
```

FM-LN and ordinary MF have no teacher dependency, but serialize them on the
single GPU:

```bash
venv/bin/python train.py --algorithm fm_lognorm \
  --config config/fm_lognorm_celeba_latent.json --mode fresh

venv/bin/python train.py --algorithm mf \
  --config config/mf_celeba_latent.json --mode fresh
```

After FM epoch 100 exists at
`results/fm_celeba_latent/checkpoints/run_1/FlowMatchingAlgorithm_epoch100.pt`,
the teacher-dependent runs may be launched separately:

```bash
venv/bin/python train.py --algorithm mf_distill \
  --config config/mf_distill_celeba_latent.json --mode fresh

venv/bin/python train.py --algorithm consistency \
  --config config/consistency_celeba_latent.json --mode fresh
```

Generate Reflow pairs only after the same FM checkpoint exists:

```bash
venv/bin/python scripts/generate_reflow_pairs_latent.py \
  --checkpoint results/fm_celeba_latent/checkpoints/run_1/FlowMatchingAlgorithm_epoch100.pt \
  --config config/fm_celeba_latent.json \
  --output data/reflow_pairs_celeba_latent.pt \
  --n-pairs 50000 \
  --nfe 50 \
  --batch-size 64 \
  --chunk-size 1000 \
  --seed 0

venv/bin/python train.py --algorithm reflow \
  --config config/reflow_celeba_latent.json --mode fresh
```

If a training job is interrupted, inspect its logs and use `--mode continue`
for that same run rather than starting another fresh run. Never run two CUDA
jobs concurrently on the RTX 3090.
