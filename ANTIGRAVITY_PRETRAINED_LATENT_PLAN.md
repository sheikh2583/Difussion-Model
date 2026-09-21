# Prompt for Antigravity: Frozen Pretrained Codec and Pixel-Boundary Integration

> **Superseded architecture note (2026-09-21):** The user explicitly selected
> the standard frozen pretrained VAE path after the scratch experiment failed
> its gate. The active implementation uses `stabilityai/sd-vae-ft-mse`, a
> factor-8 `AutoencoderKL`, producing `(B,4,8,8)` from CelebA-64. The text below
> is retained as historical planning context and is no longer the live contract.

You are the Antigravity/Claude agent working concurrently with Codex in the
same repository and filesystem. There is no Git workflow in this phase.

Read `AGENTS.md`, this file, and `CROSS_TRACK.md` completely before writing.
Obey the static ownership table in `AGENTS.md`. Never edit a Codex-owned path,
even to make a convenient fix; post the dependency to `CROSS_TRACK.md`.

## Objective

Implement the primary frozen pretrained factor-4 codec path and the shared
pixel-boundary integration needed to train all six algorithms on normalized
CelebA latents while keeping existing pixel-space behavior unchanged. Also
implement backend-neutral validation/caching so the Codex-owned scratch codec
can be selected as a fallback through the same interface.

Do not train, download weights, cache CelebA, run GPU evaluation, or execute the
full smoke test. The operator performs all artifact-producing commands.

## Mandatory pre-read

Read these files before changing anything:

1. `config/config.py`
2. `algorithms/base.py`
3. `models/backbone.py`
4. `data/celeba.py`
5. `data/dataset_registry.py`
6. `training/trainer.py`
7. `evaluation/evaluator.py`
8. `evaluation/metrics.py`
9. `experiments/runner.py`
10. `sampling/sampler.py`
11. `utils/results.py`
12. all six `config/*_celeba64.json` files

Confirm from the live code rather than trusting line numbers in an older
prompt. Preserve every existing public call path.

## Locked architecture

- Primary representation: a genuinely factor-4 pretrained KL autoencoder.
- Pixel shape/range: `(B,3,64,64)` in `[-1,1]`.
- Normalized latent shape: `(B,4,16,16)`, approximately zero mean/unit standard
  deviation per channel using statistics frozen from the CelebA training set.
- Posterior policy: deterministic posterior mean.
- Encoder/decoder: frozen, evaluation mode, no gradient tracking.
- Generative backbone: trained from scratch by every algorithm; this track does
  not add pretrained U-Net weights.
- FID/IS: decoded RGB samples versus raw CelebA validation pixels.
- NFE excludes decoder execution; decoder time is separately recorded.

Do not use `CompVis/stable-diffusion-v1-4` while calling it KL-f4. The SD 1.x
VAE is normally factor-8 and would produce 8x8 latents from 64x64 input. Use an
actual factor-4 CompVis KL checkpoint or another explicitly identified,
reproducible factor-4 checkpoint. Pin and record the exact source/revision and
verify output shape. If no reliable factor-4 weights are locally available,
finish the implementation with a clear operator-facing prerequisite; never
silently fall back to factor-8.

## Your exclusive write set

- `codec/__init__.py`
- `codec/base.py`
- `codec/pretrained_vae.py`
- `codec/codec_factory.py`
- `codec/validate_codec.py`
- `codec/cache_latents.py`
- `evaluation/evaluator.py`
- `sampling/sampler.py`
- `utils/results.py`
- `scripts/smoke_latent.py`
- tests dedicated to these components
- dependency declarations strictly required by the pretrained codec

Do not edit configs, algorithms, dataset registry, latent dataset, scratch VAE,
scratch trainer, or Reflow-pair generator.

## Action plan

### 1. Establish the codec contract first

Create `codec/base.py` with the interface and checkpoint schema agreed in
`AGENTS.md`. Validate metadata strictly:

- reject missing or unsupported schema versions;
- reject `stats_frozen != True` for inference/cache use;
- reject latent channel/factor/pixel-size mismatches;
- require positive finite latent standard deviations;
- preserve native scaling metadata rather than assuming `0.18215`;
- record a stable weight/source digest and revision.

After this interface compiles and has CPU tests, append a `READY` entry to
`CROSS_TRACK.md` so Codex can bind the scratch implementation to it.

### 2. Implement the frozen pretrained factor-4 wrapper

Implement `PretrainedKLVAE` using the common contract:

- load the exact pinned factor-4 architecture and weights;
- keep encoder and decoder frozen and in eval mode;
- `encode_mean()` must return the deterministic posterior mean in the codec's
  native scaled latent convention;
- `decode()` must invert native scaling correctly and return RGB in `[-1,1]`;
- statistics are shaped `(1,4,1,1)` and computed over all batch/spatial axes;
- normalization and denormalization must round-trip numerically;
- saving need not duplicate large pretrained weights only if the checkpoint
  stores enough immutable identity to reload the exact local/source weights.

Use optional imports and actionable errors so existing pixel-only workflows do
not require the new codec dependency.

### 3. Implement backend-neutral factory and validation

`codec_factory.load_codec(path, device)` dispatches by `codec_type`, supports
both the pretrained backend and Codex's `ScratchKLVAE`, and performs schema and
frozen-stat validation. Use lazy imports so either file can be developed
concurrently.

`validate_codec.py` is an operator-run command that:

- loads raw CelebA validation pixels;
- confirms exact encode/decode shapes and ranges;
- computes frozen train-set latent statistics when creating the accepted codec
  checkpoint;
- reports reconstruction FID on 5,000 validation images, PSNR, and per-channel
  latent statistics;
- writes a reconstruction grid, JSON report, and codec checkpoint only after
  successful structural validation;
- uses the gate `rFID < 5`, `PSNR > 30 dB`, and minimum channel std `> 0.1` as
  the requested strict gate, while clearly printing that it is a project gate,
  not a universal literature threshold;
- exits nonzero on failure.

Decode/evaluate in bounded batches. Do not hold decoder activations for all
5,000 images simultaneously.

### 4. Implement content-addressed latent caching

`cache_latents.py` must work with either codec backend through the factory.

- Encode CelebA train and validation splits with posterior means.
- Normalize with already frozen training-set statistics.
- Store float32 `(N,4,16,16)` tensors and complete metadata.
- Build cache identity from codec content/source digest, preprocessing schema,
  split, posterior mode, and normalization schema—not absolute path alone.
- Hash the complete tensor file and verify the complete hash when consuming it.
- Allow multiple codec caches to coexist; never reject a valid cache merely
  because another codec's cache exists.
- Publish cache directories atomically so interruptions cannot expose a
  half-written cache.

Do not execute this command.

### 5. Add the latent evaluation boundary

Modify `evaluation/evaluator.py` additively:

- For `celeba_latent`, FID reference metadata describes decoded output:
  `5000`, `64x64`, RGB.
- Build a missing latent-experiment FID reference from the raw CelebA
  validation loader, never from cached latents or reconstructions.
- Retain the separately named latent cache path supplied by latent configs.
- Decode generated normalized latents in configurable bounded batches.
- Feed only `(B,3,64,64)` decoded images in the expected range to FID/IS.
- Record backbone sampling time and decoder time separately through additive,
  optional fields in `utils/results.py`.
- Lazily load one codec instance and reuse it.
- Leave the pixel path equivalent in behavior.

Do not decode all 5,000 samples in one VAE call.

### 6. Make grid saving safe

Modify `sampling/sampler.py` additively so a 4-channel latent is never passed
to `torchvision.save_image`. It may skip the raw grid with an explicit log
message; decoded reconstruction/sample grids are produced by codec validation
and the integration smoke command. Preserve current RGB behavior.

### 7. Implement an operator-run integration smoke command

`scripts/smoke_latent.py` must support two modes:

- a CPU/static mode usable by agents with mocked codec/data and no real
  checkpoint;
- a full operator mode requiring the real codec checkpoint, latent cache,
  prerequisites, CelebA, and CUDA.

The full mode checks:

1. codec schema, identity, frozen state, and exact factor/channel values;
2. real encode/normalize/decode shape, range, and reconstruction grid;
3. complete latent-cache manifest verification and batch shape;
4. config-derived run directory names;
5. one training step and one sample for each ready algorithm, skipping
   dependency-bound algorithms with explicit reasons;
6. latent sample remains unclamped and decodes correctly;
7. sampler never writes a meaningless four-channel grid;
8. every existing and latent config parses;
9. FID reference metadata is 64x64 RGB and sourced from raw pixels.

The agent must not execute full mode.

## Corrections retained from the original brief

- FID reference remains raw pixels, not `decode(encode(real))`.
- Latent FID uses a separate cache filename even though its metadata describes
  the same raw RGB protocol.
- Decode happens only at output/evaluation boundaries.
- Existing pixel runs and CIFAR-10 remain untouched.
- Pretrained codec is validated before the scratch fallback is considered.

## Static verification required

Run only CPU/static checks:

- compile every owned Python file;
- test checkpoint schema failures and normalization round trips;
- test factory dispatch with mocked backends;
- test batched decode with a fake codec;
- test raw-pixel FID-loader selection without calculating FID;
- test four-channel grid suppression and unchanged RGB behavior;
- ensure imports remain optional for pixel-only workflows.

Record any cross-track dependency in `CROSS_TRACK.md`. Finish with an operator
runbook, but do not execute it.
