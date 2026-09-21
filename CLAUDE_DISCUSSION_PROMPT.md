# Claude Discussion Brief: Moving the Algorithms to Latent Space

## Paste this message after uploading `thesis_context.zip`

I uploaded a compact snapshot of my thesis project. Please read the project
structure and this brief before proposing code changes. For now, I want a
technical architecture discussion and a concrete migration plan, not an
implementation.

The intended change is to move the existing generative algorithms from pixel
space to a compressed latent space. Please correct any mistaken assumptions in
my proposal, identify the affected modules and interfaces, and recommend the
smallest scientifically defensible design. Preserve the ability to compare all
algorithms fairly and preserve the completed pixel-space runs as baselines.

## Current architecture

- Datasets produce normalized RGB tensors in `[-1, 1]`.
- `models/backbone.py` builds one shared, time-conditioned `SimpleUNet`.
- The U-Net is trained from scratch and maps `(x_t, t)` to a tensor with the
  same shape as the input.
- FM, FM with logit-normal time sampling, Mean Flow, Mean-Flow Distillation,
  Consistency, and Reflow currently define their paths, losses, and sampling
  procedures directly in pixel space.
- `BaseAlgorithm.training_step()` receives image tensors, and
  `BaseAlgorithm.sample()` returns image tensors.
- Evaluation computes FID and Inception Score on generated RGB images.
- The shared backbone and experimental controls are deliberately kept equal
  across algorithms for fair comparisons.

## Terminology correction

A generative U-Net is not normally the component that compresses an image into
a persistent latent representation or reconstructs the final image. A latent
generative pipeline usually has two distinct model roles:

1. A pretrained autoencoder or VAE provides an encoder `E` and decoder `D`:
   `x -> z = E(x)` and `z -> x_hat = D(z)`.
2. A time-conditioned U-Net or transformer predicts velocity, noise, or another
   algorithm-specific target while operating on the latent tensor `z_t`.

Generation therefore follows:

```text
Gaussian latent noise z_1
        |
        v
FM / FM-LN / MF / CM / Reflow dynamics using a latent-space backbone
        |
        v
generated latent z_0
        |
        v
frozen pretrained decoder D(z_0)
        |
        v
RGB image for saving and FID/IS evaluation
```

Training follows:

```text
RGB image x -> frozen encoder E(x) -> scaled latent z
             -> algorithm-specific interpolation/loss in latent space
             -> train the shared latent-space backbone
```

The default proposal is to freeze the pretrained encoder and decoder and train
the algorithm backbone in latent space. This prevents reconstruction training
from becoming an uncontrolled difference between algorithms.

## Decisions I want help making

Please evaluate these as two separate choices:

### 1. Pretrained latent codec

- Which pretrained VAE/autoencoder is technically suitable for both CIFAR-10
  32x32 and CelebA 64x64?
- Many Stable Diffusion VAEs were trained at much higher resolutions. Would
  their stride and training domain make them unsuitable for this thesis?
- Should the project instead use a small dataset-appropriate pretrained
  autoencoder, and what credible checkpoint/source would support that choice?
- What latent shape, channel count, scaling factor, posterior sampling policy,
  and normalization should be recorded in the configuration and checkpoints?
- Should training use posterior samples or posterior means? Discuss the effect
  on reproducibility and the learned target distribution.
- Should latents be encoded online or cached? Compare storage, augmentation,
  reproducibility, and throughput implications.

### 2. Pretrained generative backbone

- Is it defensible to initialize the latent U-Net from pretrained weights, or
  should only the VAE be pretrained while the shared latent backbone is trained
  from scratch?
- A third-party diffusion U-Net may expect different conditioning, prediction
  targets, latent scaling, and resolutions. Explain what can and cannot be
  transferred safely to FM, Mean Flow, Consistency, and Reflow.
- If pretrained U-Net weights are used, all compared algorithms must begin from
  the same initialization. Explain how to keep the comparison fair.

## Proposed software boundary

Please critique this boundary rather than assuming it is correct:

- Add a `LatentCodec` abstraction responsible only for `encode(images)` and
  `decode(latents)`.
- Add codec configuration: implementation/checkpoint identity, frozen status,
  latent channels, spatial downsampling factor, scaling factor, posterior mode,
  and optional cache settings.
- Keep each algorithm's mathematics unchanged in form, but replace pixel data
  `x` with scaled latent data `z` when constructing paths and targets.
- Build the shared backbone with latent channels and latent spatial size.
- Decode only at external boundaries: sampling previews, saved grids, and
  FID/IS evaluation.
- Keep the trainer generic. Decide whether encoding belongs in the runner/data
  pipeline or in a shared algorithm wrapper; avoid copying codec logic into
  every algorithm.
- Save codec identity and latent conventions in checkpoint provenance, and
  reject incompatible resumes.
- Use new experiment names/result directories so existing pixel-space metrics
  and checkpoints are never overwritten or silently combined with latent runs.

## Scientific and evaluation constraints

- Preserve completed pixel-space experiments as baselines.
- Use the same frozen codec and latent backbone architecture for every latent
  algorithm unless a difference is explicitly part of an ablation.
- Report codec parameters separately from trainable generative-model
  parameters, and report trainable versus frozen parameter counts.
- FID and Inception Score must be computed on decoded RGB images with the same
  sample counts and preprocessing as the pixel-space baselines.
- Measure reconstruction quality of `D(E(x))` first. The codec places a ceiling
  on attainable image quality, especially at 32x32 and 64x64.
- Keep NFE defined as generative-backbone evaluations. Report decoder time
  separately so sampling-time comparisons remain interpretable.
- Record whether latent encoding was cached and exclude one-time caching cost
  from per-epoch training time while reporting it separately.
- Add a small smoke test before any full run: encode/decode shape and range,
  reconstruction grid, one training step for each algorithm, sampling decode,
  checkpoint round trip, and FID input validation.

## What I want in your response

Please provide:

1. A corrected architecture recommendation.
2. A comparison of realistic codec/backbone choices for 32x32 CIFAR-10 and
   64x64 CelebA, including important risks.
3. A file-by-file migration plan based on the uploaded repository.
4. The exact tensor shapes and value/scaling conventions through training and
   sampling.
5. A checkpoint/config migration strategy that cannot corrupt existing runs.
6. A validation and ablation plan suitable for a thesis.
7. A rough GPU-memory, storage, and runtime impact analysis for the RTX 3090
   environment recorded in the project.
8. Any reasons this latent-space change could weaken comparison with the
   existing results, and how to present that limitation honestly.

Do not write code yet. End with the decisions you need from me before an
implementation begins.
