# Prompt for Codex: Scratch Codec and Latent-Native Training Side

You are the Codex agent working concurrently with Antigravity/Claude in the
same repository and filesystem. There is no Git workflow in this phase.

Read `AGENTS.md`, this file, and `CROSS_TRACK.md` completely before writing.
Obey the static ownership table in `AGENTS.md`. Never edit an
Antigravity-owned path; post dependencies to `CROSS_TRACK.md`.

## Objective

Implement the self-trained factor-4 KL-VAE fallback and the latent-native data,
configuration, algorithm-output, and Reflow-pair side of the CelebA latent
extension. The scratch codec is implemented but not trained by the agent. The
same latent infrastructure must accept Antigravity's frozen pretrained codec as
the primary backend without branching inside the algorithms.

Do not train, download weights, cache CelebA, generate Reflow pairs, run CUDA
sampling/evaluation, or execute the full smoke test.

## Mandatory pre-read

Read these files before changing anything:

1. `algorithms/base.py`
2. all six concrete algorithm files
3. `models/backbone.py`
4. `data/celeba.py`
5. `data/dataset_registry.py`
6. `config/config.py`
7. `training/trainer.py`
8. `experiments/runner.py`
9. `scripts/generate_reflow_pairs.py`
10. all six `config/*_celeba64.json` files
11. Antigravity's `codec/base.py` if it already exists

Confirm behavior from live code. In particular, the trainer expects each
loader item to unpack as `(batch, label)`, and all current algorithms clamp
samples to `[-1,1]`, which is invalid for normalized latents.

## Locked architecture

- Scratch fallback: factor-4 KL-VAE, RGB 64x64 to four-channel 16x16 latent.
- Posterior means are used for the cached generative dataset.
- Frozen training-set per-channel mean/std define normalized latents.
- The latent U-Net is randomly initialized and trained separately for each
  algorithm.
- Existing pixel behavior, configs, results, and CIFAR-10 remain untouched.
- The scratch codec uses the exact common interface/checkpoint schema in
  `AGENTS.md` and `codec/base.py`.

## Your exclusive write set

- `codec/scratch_vae.py`
- `codec/train_scratch_vae.py`
- `data/celeba_latent.py`
- `data/dataset_registry.py`
- `config/config.py`
- six `config/*_celeba_latent.json` files
- `algorithms/base.py`
- all six concrete algorithm files listed in `AGENTS.md`
- `scripts/generate_reflow_pairs_latent.py`
- tests dedicated to these components

Do not edit codec factory/cache/pretrained files, evaluator, sampler, results
schema, or Antigravity's smoke command.

## Action plan

### 1. Bind to the shared codec contract

If `codec/base.py` is not ready, implement independent scratch internals and
post `NEEDS_INTERFACE`; do not create or edit the other track's file. Once the
`READY` entry appears, make `ScratchKLVAE` conform exactly.

### 2. Implement the scratch factor-4 KL-VAE

Create `codec/scratch_vae.py` with:

- encoder: 3→64, residual block, stride-2 64→128, residual block,
  stride-2 128→128, residual block, normalized activation, projection to eight
  channels split into four-channel mean/log-variance;
- decoder: four-channel input, 128-channel residual stack, two stride-2
  transposed-convolution stages, final RGB `tanh` output;
- residual skip projections when channel counts differ;
- `encode()` returning `(sample, mean, logvar)` for codec training;
- deterministic `encode_mean()` for latent caching;
- standard mean KL loss with numerically bounded log variance;
- codec normalization, denormalization, frozen-stat computation, save/load,
  and metadata exactly matching the common contract.

Avoid ambiguous `GroupNorm(min(8, channels), channels)` if divisibility is not
guaranteed; choose a valid divisor explicitly.

### 3. Implement, but do not run, scratch codec training

Create `codec/train_scratch_vae.py` as an operator command:

- CelebA train/validation pixels use the existing crop/resize/range;
- AdamW, learning rate `1e-4`, weight decay `1e-4`, batch 128, 60 epochs;
- KL warmup `1e-5 → 1e-4` over epochs 0–20, then constant;
- save resumable checkpoints atomically every five epochs;
- validate every five epochs with reconstruction FID (5,000), PSNR, and
  per-channel latent statistics;
- print the requested strict gate: rFID `<5`, PSNR `>30 dB`, minimum channel
  std `>0.1`, clearly labeled as a project acceptance gate;
- exit nonzero when the final gate fails;
- after acceptance, compute/freeze training-set latent statistics and save a
  factory-loadable codec checkpoint.

The requested MSE+KL objective is the baseline. Document that it may produce
blurry reconstructions and fail the strict perceptual gate; do not silently add
an adversarial/perceptual loss that would change the experiment. Such a change
requires user approval.

### 4. Extend configuration additively

In `config/config.py`, add backward-compatible fields required by latent runs.
At minimum:

```text
DatasetConfig.cache_dir
DatasetConfig.split
DatasetConfig.codec_checkpoint
BackboneConfig.sample_clamp
```

Defaults must preserve all existing behavior (`sample_clamp=True`). Do not add
duplicate optimizer/scheduler schemas. Load every existing JSON config in a
CPU-only regression test.

### 5. Implement the latent dataset correctly

Create `data/celeba_latent.py` and register `celeba_latent` additively.

- Resolve the cache matching the configured codec identity and split; reject
  ambiguity.
- Verify the complete manifest hash and metadata before exposing tensors.
- Load with supported read-only/mmap behavior where available, with a clear
  compatibility fallback.
- Validate float32 shape `(N,4,16,16)` and finite values.
- Return `(latent, -1)` from `__getitem__`, not the bare tensor, so the current
  trainer remains unchanged.
- Return train/validation DataLoaders with deterministic train shuffling,
  `drop_last=True` for train to mirror pixel CelebA, and no shuffle for val.

Do not modify `data/celeba.py`.

### 6. Make algorithm sample finalization representation-aware

Add one backward-compatible sample-finalization helper to `BaseAlgorithm`.

- Pixel configs continue clamping to `[-1,1]`.
- Latent configs set `backbone.sample_clamp=false` and return finite,
  unbounded normalized latents without clamping.
- Route the final return of all six algorithms through this helper.
- Do not change their losses, paths, teacher behavior, EMA behavior, NFE, or
  existing pixel results.

Do not add the original prompt's proposed teacher preflight: the current
constructors already call `resolve_checkpoint_reference`, which checks and
resolves the teacher checkpoint before lazy weight loading.

### 7. Create six isolated latent configs

Create:

- `config/fm_celeba_latent.json`
- `config/fm_lognorm_celeba_latent.json`
- `config/mf_celeba_latent.json`
- `config/mf_distill_celeba_latent.json`
- `config/consistency_celeba_latent.json`
- `config/reflow_celeba_latent.json`

For every file:

- copy batch size, epochs, optimizer, scheduler, seed, checkpoint cadence, and
  algorithm kwargs from its corresponding CelebA-64 pixel config;
- set `dataset.name="celeba_latent"`, `image_size=16`, configured cache dir and
  codec checkpoint;
- set backbone `in_channels=4`, existing CelebA channel multipliers, and
  `sample_clamp=false`;
- set `experiment_name` to only the algorithm key so runner produces
  `results/<algo>_celeba_latent`;
- use a separately named latent FID reference cache;
- point MF-Distill and Consistency to the latent FM epoch-100 checkpoint under
  `checkpoints/run_1`;
- point Reflow to `data/reflow_pairs_celeba_latent.pt`.

Do not change any existing config.

### 8. Implement latent Reflow-pair generation

Create `scripts/generate_reflow_pairs_latent.py` without modifying the pixel
script.

Correct the earlier brief: Reflow pairs do not require encoding each CelebA
image or reading the latent cache. They consist of Gaussian `z1` and the
trained latent FM model's generated `x0` endpoint.

- Load latent FM config/checkpoint with provenance/shape checks.
- Generate a requested number of `(z1, x0)` pairs using NFE 50 by default.
- Store float32 CPU tensors shaped `(N,4,16,16)` atomically using keys exactly
  `z1` and `x0` for the existing `ReflowAlgorithm`.
- Include a sidecar/embedded metadata schema with FM checkpoint digest, NFE,
  seed, shape, config identity, and pair count without breaking the two
  required tensor keys.
- Support restart-safe chunking so a long operator run is recoverable.
- The codec is not needed unless an optional decoded preview is requested; do
  not make pair generation depend on the latent dataset.

Do not execute pair generation.

### 9. Focused CPU/static tests

Test without real datasets, checkpoints, downloads, or CUDA:

- scratch encode/sample/mean/decode shapes at tiny batch size;
- KL finiteness and normalization round trip;
- scratch checkpoint metadata/save/load using temporary files;
- all existing and new configs parse;
- latent run-directory derivation is collision-free;
- latent dataset accepts valid fake cache, rejects wrong shape/hash/identity,
  and returns `(tensor, label)`;
- all algorithms retain pixel clamping and skip latent clamping on synthetic
  outputs;
- Reflow output schema/chunk assembly using a mocked FM model.

## Operator commands to document, not execute

Use the current CLI correctly. Training requires both algorithm and lifecycle:

```bash
venv/bin/python train.py --algorithm fm \
  --config config/fm_celeba_latent.json --mode fresh
```

Corresponding later runs use `--algorithm fm_lognorm`, `mf`, `mf_distill`,
`consistency`, or `reflow`. FM-LN and MF have no FM-teacher prerequisite;
MF-Distill and Consistency require latent FM; Reflow requires latent FM plus
operator-generated latent pairs.

The scratch trainer is a fallback only after the operator rejects the
pretrained codec gate. Never launch it yourself.

Record cross-track needs and completion in `CROSS_TRACK.md` without editing
the other track's files.
