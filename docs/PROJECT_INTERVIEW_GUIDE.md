#.-=0/ Flow Matching, Mean Flow, and Logit-Normal Flow Matching

## A repository and mathematics interview guide

This document explains what the project actually does, from the command-line entry point to the saved artifacts and browser inference service. It distinguishes the mathematical intent of each algorithm from the implementation details that are shared by all algorithms.

## 1. The one-minute answer

The project is an unconditional CIFAR-10 image generator. It uses one shared conditional U-Net and swaps only the algorithm-specific interpretation of that network's output:

- **Flow Matching (FM):** predict the instantaneous velocity of a straight path from a real image to Gaussian noise, then solve the learned ODE backward with Euler steps.
- **Flow Matching with logit-normal time sampling (FM-LN):** exactly the same path, target, network, and sampler as FM, but draw training times as $t=\sigma(u)$ with $u\sim\mathcal N(0,1)$ instead of $t\sim\mathcal U(0,1)$. This is a training-distribution change, not a new sampler.
- **Mean Flow (MF):** predict the average velocity over an interval $[r,t]$, using a Jacobian-vector product (JVP) to construct the Mean Flow Identity target. Sampling uses the learned displacement directly and can take a one-step noise-to-data jump.

All three use the same CIFAR-10 preprocessing and the same backbone configuration. MF additionally owns a 323-parameter $r$-embedding because the shared backbone accepts one explicit time input but MF needs two endpoints.

The complete intended path is:

```text
train.py
  -> ExperimentConfig
  -> ExperimentRunner
  -> CIFAR-10 DataLoader + shared SimpleUNet + algorithm wrapper
  -> Trainer.training_step repeatedly
  -> checkpoints/*.pt and metrics/*.jsonl
  -> Sampler.algorithm.sample for each NFE
  -> Evaluator: FID / Inception Score
  -> samples/*.png, metrics, logs
```

## 2. Repository map

| Layer                    | Files                                                                                                                                                                                          | Responsibility                                                  |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Entry points             | [train.py](../train.py), [evaluate.py](../evaluate.py), [web/inference_server.py](../web/inference_server.py)                                                                                  | Start training, evaluate a checkpoint, or serve the browser UI  |
| Configuration            | [config/config.py](../config/config.py), [config/fm_full.json](../config/fm_full.json), [config/fm_lognorm_full.json](../config/fm_lognorm_full.json), [config/mf_full.json](../config/mf_full.json) | Shared controls and algorithm-only options                      |
| Data                     | [data/cifar10.py](data/cifar10.py)                                                                                                                                                             | CIFAR-10 download, resize, tensor conversion, normalization     |
| Backbone                 | [models/backbone.py](models/backbone.py)                                                                                                                                                       | Shared time-conditioned SimpleUNet                              |
| Algorithms               | [algorithms/flow_matching.py](algorithms/flow_matching.py), [algorithms/flow_matching_lognorm.py](algorithms/flow_matching_lognorm.py), [algorithms/mean_flow.py](algorithms/mean_flow.py)     | FM, FM-LN, and MF mathematics                                   |
| Shared engine            | [training/trainer.py](training/trainer.py), [sampling/sampler.py](sampling/sampler.py)                                                                                                         | Optimization, checkpointing, sampling timing, and batching      |
| Experiment orchestration | [experiments/runner.py](experiments/runner.py)                                                                                                                                                 | Builds the common objects and runs train/sample/evaluate        |
| Metrics                  | [evaluation/evaluator.py](evaluation/evaluator.py), [evaluation/metrics.py](evaluation/metrics.py)                                                                                             | Cached real reference, FID, and Inception Score                 |
| Artifacts                | [results](results)                                                                                                                                                                             | Config snapshots, checkpoints, samples, logs, and JSONL metrics |

## 3. Start-to-end execution trace

### 3.1 CLI and registry

`train.py` parses `--algorithm`, loads a JSON configuration, selects the class from `ALGORITHM_REGISTRY`, creates an `ExperimentRunner`, saves the effective configuration, and calls `run_full`. The registry includes `mock`, `fm`, `fm_lognorm`, and `mf` in [train.py](train.py#L19-L56).

`ExperimentRunner.run_full` performs training, sampling, and evaluation in that order in [experiments/runner.py](experiments/runner.py#L83-L86). The runner is intentionally algorithm-agnostic: it constructs the same dataloaders and backbone for every algorithm, then passes the algorithm object to the generic trainer and sampler.

### 3.2 Configuration and fairness

`ExperimentConfig` defines the shared controls: seed, device, AMP, batch size, epochs, checkpoint frequency, dataset, backbone, optimizer, and evaluation settings in [config/config.py](config/config.py#L35-L62).

`algorithm_kwargs` is supposed to contain only algorithm-inherent parameters. The runner rejects an algorithm keyword that shadows shared controls in [experiments/runner.py](experiments/runner.py#L22-L35). This is important experimentally: changing batch size or optimizer through an algorithm-specific dictionary would invalidate an FM/MF comparison.

The fairness claim is about architecture and data construction, not automatically about every recorded run. The saved configs must still be compared. For example, the repository contains FM-LN configurations with batch size 128 and 32, while MF's saved full configuration uses batch size 32, 30 epochs, and a lower learning rate.

### 3.3 Data

`build_transforms` resizes images to 32x32, converts them to tensors, and applies mean/std $(0.5,0.5,0.5)$ normalization, mapping ordinary $[0,1]$ pixels to approximately $[-1,1]$ in [data/cifar10.py](data/cifar10.py#L11-L18).

Both train and test datasets are CIFAR-10. The training loader shuffles, drops the final incomplete batch, and uses the configured batch size in [data/cifar10.py](data/cifar10.py#L21-L44). Thus the algorithm receives a batch $x_0\in[-1,1]^{B\times3\times32\times32}$ and no labels are used by FM, FM-LN, or MF.

### 3.4 Shared network

`build_backbone` creates the same `SimpleUNet` for every algorithm and records the configured image size in [models/backbone.py](models/backbone.py#L107-L124). The network maps

$$
f_\theta(x,t):\mathbb R^{3\times32\times32}\times[0,1]\to\mathbb R^{3\times32\times32}.
$$

The scalar $t$ is encoded by sinusoidal features and projected into residual blocks. The U-Net's output has the same image shape, but the backbone itself does not know whether that output means noise, instantaneous velocity, or average velocity. That semantic meaning is assigned by the algorithm wrapper. The relevant implementation is [models/backbone.py](models/backbone.py#L13-L31) and [models/backbone.py](models/backbone.py#L53-L105).

The standard 32x32 configuration records 6,352,899 backbone parameters in the supplied metric artifacts. MF reports those same backbone parameters plus `algorithm_extra_parameter_count = 323` for its $r$-embedding.

## 4. The common generative path

The algorithms use the same coupling for training:

1. Select a real image $x_0$.
2. Sample independent Gaussian noise $\epsilon\sim\mathcal N(0,I)$.
3. Define a straight interpolation

$$
z_t=(1-t)x_0+t\epsilon,\qquad t\in[0,1].
$$

4. Its instantaneous velocity is

$$
v=\frac{d z_t}{dt}=\epsilon-x_0.
$$

At generation time, start from $z_1\sim\mathcal N(0,I)$ and move toward $t=0$, where the data distribution is intended to live. The shared path is visible in [algorithms/flow_matching.py](algorithms/flow_matching.py#L48-L82) and [algorithms/mean_flow.py](algorithms/mean_flow.py#L137-L158).

This is related to diffusion models, but it is not the standard DDPM implementation. A DDPM normally learns a denoising/noise score parameterization over a stochastic reverse-time Markov chain. Here the project learns a deterministic vector field and generates by solving an ODE. Flow matching can use diffusion-like probability paths, but this repository specifically uses a linear interpolation between data and Gaussian noise.

## 5. Flow Matching (FM)

### 5.1 Objective

FM trains the network to approximate the conditional velocity $v=\epsilon-x_0$ at random points on the path:

$$
\mathcal L_{FM}(\theta)
=\mathbb E_{x_0,\epsilon,t}\left[
\left\|f_\theta(z_t,t)-(\epsilon-x_0)\right\|_2^2
\right],
\qquad t\sim\mathcal U(0,1).
$$

The code sequence is exact:

- noise: [algorithms/flow_matching.py](algorithms/flow_matching.py#L50-L52)
- uniform time: [algorithms/flow_matching.py](algorithms/flow_matching.py#L54-L56)
- interpolant: [algorithms/flow_matching.py](algorithms/flow_matching.py#L59-L64)
- target velocity: [algorithms/flow_matching.py](algorithms/flow_matching.py#L66-L69)
- prediction and MSE: [algorithms/flow_matching.py](algorithms/flow_matching.py#L72-L78)

The MSE is a conditional-flow-matching regression objective. In the idealized infinite-data/infinite-capacity limit, the learned field represents the appropriate conditional expectation of the target velocity given $(z_t,t)$, which is the vector field used by the probability-flow ODE.

### 5.2 Sampling

The learned field is defined in the forward direction, data $\to$ noise. Generation reverses it. With $h=1/N$ and current time $t_i=1-ih$:

$$
z_{t_{i+1}}\approx z_{t_i}-h f_\theta(z_{t_i},t_i).
$$

This is backward Euler integration in the time orientation used by the code, although the numerical update is the explicit Euler formula applied with $\Delta t=-h$. The implementation samples Gaussian noise, loops exactly `nfe` times, evaluates the backbone once per step, and clamps the output to $[-1,1]$ in [algorithms/flow_matching.py](algorithms/flow_matching.py#L93-L142).

Therefore, for FM, `nfe` is also the number of backbone evaluations per sample, apart from sampler chunking and framework overhead.

## 6. Flow Matching with logit-normal time sampling (FM-LN)

FM-LN changes one line of the training distribution. It samples

$$
u\sim\mathcal N(0,1),\qquad t=\sigma(u)=\frac{1}{1+e^{-u}}.
$$

The induced density on $t$ is the logit-normal density

$$
p(t)=\frac{1}{\sqrt{2\pi}\,t(1-t)}
\exp\left[-\frac12\left(\log\frac{t}{1-t}\right)^2\right],\qquad 0<t<1.
$$

The path and target remain

$$
z_t=(1-t)x_0+t\epsilon,\qquad v=\epsilon-x_0,
$$

and the loss remains the same MSE. The only change is `u = torch.randn(...)` followed by `t = torch.sigmoid(u)` in [algorithms/flow_matching_lognorm.py](algorithms/flow_matching_lognorm.py#L54-L62). Sampling is byte-for-byte conceptually the same reverse Euler loop as FM in [algorithms/flow_matching_lognorm.py](algorithms/flow_matching_lognorm.py#L91-L142).

The important interview answer is: **logit-normal matching is not a new flow equation here; it is a non-uniform Monte Carlo weighting of the same flow-matching objective.** It spends more training samples in the central/intermediate part of the trajectory and fewer extremely close to the endpoints. It can improve practical learning because the regression problem is often more informative away from degenerate endpoints, but the repository does not implement an explicit loss reweighting or a changed ODE.

## 7. Mean Flow (MF)

### 7.1 What is being predicted?

FM predicts instantaneous velocity $v(z_t,t)$. MF predicts an interval-average velocity. For a trajectory $z_\tau$ and $r\le t$:

$$
u(z_t,r,t)=\frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,d\tau.
$$

The displacement identity is

$$
(t-r)u(z_t,r,t)=z_t-z_r,
$$

so the ideal interval jump is

$$
z_r=z_t-(t-r)u(z_t,r,t).
$$

This is why MF can use a small number of network calls at inference: it predicts a displacement over the whole requested interval rather than approximating that interval with many instantaneous-velocity evaluations.

### 7.2 Conditioning on both endpoints

The shared backbone signature is only `model(x, t)`, but MF needs $(z_t,r,t)$. MF therefore creates `r_embed`, a small MLP `Linear(1 -> 64) -> SiLU -> Linear(64 -> C)`, and adds its channel-wise signal to the image before calling the unchanged backbone in [algorithms/mean_flow.py](algorithms/mean_flow.py#L60-L88) and [algorithms/mean_flow.py](algorithms/mean_flow.py#L103-L128).

The explicit time argument supplies $t$; the additive embedding supplies $r$. `trainable_modules()` returns both the shared backbone and `r_embed` in [algorithms/mean_flow.py](algorithms/mean_flow.py#L90-L101). This preserves the shared backbone parameter count but means the total trainable parameter count is not literally identical.

### 7.3 Mean Flow Identity and JVP target

Differentiating the displacement/average-velocity relation along the trajectory gives the Mean Flow Identity:

$$
u(z_t,r,t)=v(z_t,t)-(t-r)\frac{d}{dt}u(z_t,r,t).
$$

The total derivative is

$$
\frac{d}{dt}u
=\frac{\partial u}{\partial z_t}v+\frac{\partial u}{\partial t},
$$

with $r$ held fixed. The code obtains this derivative through a JVP with tangent $(v,0,1)$:

$$
\operatorname{JVP}_u[(v,0,1)] = \frac{d}{dt}u(z_t,r,t).
$$

The implementation samples $t$, constructs $z_t$ and $v$, samples $r\sim\mathcal U(0,t)$, and sometimes sets $r=t$ with probability `p_same` in [algorithms/mean_flow.py](algorithms/mean_flow.py#L137-L172). It then calls `torch.func.jvp` with exactly the tangent $(v,0,1)$ in [algorithms/mean_flow.py](algorithms/mean_flow.py#L174-L204).

The target and loss are

$$
u_{target}=\operatorname{stopgrad}\left(v-(t-r)\frac{d u}{dt}\right),
\qquad
\mathcal L_{MF}=\mathbb E\left[\|u_\theta(z_t,r,t)-u_{target}\|_2^2\right].
$$

Those operations are [algorithms/mean_flow.py](algorithms/mean_flow.py#L206-L217). The stop-gradient is deliberate: the JVP-generated target is treated as a regression target and gradients flow through `u_pred`, not through the target construction.

When $r=t$, the interval length is zero and the target reduces to $v$. This is the diagonal case explicitly included for stability by `p_same`, defaulting to 0.25 in [algorithms/mean_flow.py](algorithms/mean_flow.py#L56-L58).

### 7.4 MF sampling

MF creates `nfe + 1` times from 1 to 0. For each adjacent interval it evaluates $u(z_t,r,t)$ once and applies

$$
z\leftarrow z-(t-r)u(z,r,t).
$$

The code is [algorithms/mean_flow.py](algorithms/mean_flow.py#L235-L279). At `nfe=1`, this is the headline jump

$$
z_0=z_1-u(z_1,0,1).
$$

The word “exact” must be qualified in an interview: the displacement update is exact **if the learned mean field is exact**. The neural approximation and the choice of interval partition still create modeling error. There is no Euler substep inside each MF interval.

## 8. Training engine and checkpoint semantics

`Trainer` moves every algorithm-owned module to the device, builds AdamW/Adam/SGD, optionally builds a scheduler, and controls AMP in [training/trainer.py](training/trainer.py#L35-L83). The batch loop calls only `algorithm.training_step(batch)`, backpropagates `out["loss"]`, and updates the optimizer in [training/trainer.py](training/trainer.py#L89-L116).

At epoch end it records loss, wall-clock time, optimization steps, samples seen, parameter counts, extra algorithm parameters, and peak GPU memory. It checkpoints at the configured frequency in [training/trainer.py](training/trainer.py#L128-L161). A checkpoint contains the state dictionaries for all algorithm-owned modules, optimizer/scheduler state, counters, and the serialized config.

This means an MF checkpoint must load two module state dictionaries in the correct order: backbone first, then `r_embed`. The generic loader uses `algorithm.trainable_modules()` to preserve that contract.

## 9. Sampling, evaluation, and metrics

The generic sampler deliberately knows no FM/MF math. It calls `algorithm.sample(batch_size, nfe, device)`, chunks evaluation requests into batches of 32, times synchronized GPU work, records throughput/memory, and saves image grids in [sampling/sampler.py](sampling/sampler.py#L17-L80).

The evaluator first ensures one cached real-image reference set. The cache is validated against image count, resolution, and channel count in [evaluation/evaluator.py](evaluation/evaluator.py#L16-L72). For each NFE it generates the configured number of images, computes FID and/or Inception Score, and appends an evaluation record in [evaluation/evaluator.py](evaluation/evaluator.py#L75-L111).

Images are converted from $[-1,1]$ to uint8 $[0,255]$ before torchmetrics receives them in [evaluation/metrics.py](evaluation/metrics.py#L29-L37). FID is the Fréchet distance between Inception feature Gaussians:

$$
\operatorname{FID}=\|\mu_r-\mu_g\|_2^2
+\operatorname{Tr}\left(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\right).
$$

Lower FID is better. Inception Score is based on

$$
\exp\left(\mathbb E_x\left[\operatorname{KL}(p(y\mid x)\|p(y))\right]\right),
$$

where higher is usually better, though it should not be treated as a substitute for visual inspection or FID.

## 10. What the saved results say

The artifacts give strong evidence of what was run, but not of who originally trained it. The source and filenames support the following provenance statement:

### FM

`results/checkpoint_samples/fm_cifar10` contains a `FlowMatchingAlgorithm` run with checkpoints, config, logs, samples, and metrics. Its supplied metrics show 100-epoch-style training and a 6,352,899-parameter backbone. It is a locally consumable checkpoint run, but the repository does not contain an external experiment manifest or Git history proving its training origin.

### MF

`results/checkpoint_samples/mf_cifar10` contains `MeanFlowAlgorithm` checkpoints through epoch 30, plus config, logs, samples, and metrics. Its records show the shared 6,352,899 backbone parameters and 323 extra MF parameters. The configuration uses batch size 32, 30 epochs, learning rate $5\times10^{-5}$, and zero data-loader workers. Therefore it is not a matched 100-epoch/batch-128 comparison with the FM config merely because the architecture is shared.

### FM-LN

`results/fm_lognorm_cifar10` contains a complete 100-epoch `FlowMatchingLognormAlgorithm` run with checkpoints at ten-epoch intervals. Its configuration uses batch size 32, learning rate $2\times10^{-4}$, and the same backbone. The records show lower training loss than the supplied FM run and evaluation values at NFE 1, 5, 10, and 20. `scripts/generate_checkpoint_samples.py` can load this run and write visual grids under `results/checkpoint_samples/`.

The results are useful for describing observed behavior, but the guide reader should not infer causality from raw loss alone. FM-LN, FM, and MF differ in training time, epoch count, batch size, learning rate, dataloader workers, numerical path, and in MF's extra module. A rigorous comparison must report those confounders or rerun a controlled study.

## 11. Interview questions and model answers

### Architecture and pipeline

**Q: What is the smallest algorithm interface?**  
**A:** `training_step(batch)` returns a scalar loss dictionary, and `sample(n_samples, nfe, device)` returns images. The interface is defined in [algorithms/base.py](algorithms/base.py#L22-L55).

**Q: Why is the backbone not called a diffusion model by itself?**  
**A:** It is only a conditional image-to-image function approximator. The algorithm decides whether its output is an instantaneous velocity or a mean velocity.

**Q: What is held constant in a fair comparison?**  
**A:** The construction path for the backbone and dataloaders, preprocessing, seed policy, evaluation reference, and shared config controls. The actual saved configs must still be inspected for matched budgets.

### Flow mathematics

**Q: Derive the FM target.**  
**A:** Differentiate $z_t=(1-t)x_0+t\epsilon$ with respect to $t$, obtaining $v=\epsilon-x_0$; regress the network output against that vector with MSE.

**Q: Why integrate from 1 to 0?**  
**A:** Training defines the vector field in the data-to-noise orientation. Generation begins at the known prior $\mathcal N(0,I)$ and reverses the ODE toward the data distribution.

**Q: What does logit-normal change?**  
**A:** Only the sampling law for $t$. It changes how often different regions of the same path contribute to the empirical regression objective; it does not change the target formula or the reverse Euler sampler.

### Mean Flow mathematics

**Q: Why does MF need $r$ and $t$?**  
**A:** Its output is an average velocity over an interval. The same current state can require different average displacements depending on the requested left endpoint $r$.

**Q: What does the JVP compute?**  
**A:** It computes the directional derivative of $u(z,r,t)$ along the trajectory tangent $(dz/dt,dr/dt,dt/dt)=(v,0,1)$, which is the total derivative $du/dt$ in the Mean Flow Identity.

**Q: Why is `u_tgt.detach()` used?**  
**A:** To stop the regression target from creating a second-order or target-side gradient path. The optimizer fits the predicted mean velocity to the constructed target.

**Q: Is MF one-step generation exact?**  
**A:** The displacement identity is exact for the true mean velocity. In code, one step has no internal Euler discretization, but learned-field approximation error remains.

### Evaluation and evidence

**Q: What does NFE mean here?**  
**A:** It is the requested number of algorithm-level network evaluations/interval updates. The project records wall-clock time separately because equal NFE does not imply equal training or inference work, especially with MF's JVP during training.

**Q: Why can FM-LN have better FID without a different sampler?**  
**A:** Its network may learn the same vector field more effectively because its training samples emphasize different times. At inference, better field approximation can improve the same Euler discretization.

**Q: Can the supplied results prove FM-LN beats MF?**  
**A:** No. They show observations under different saved configurations and run histories. A causal claim requires matched data, architecture, total training budget, optimizer schedule, evaluation reference, seeds, and checkpoint selection.

## 12. Repository issues a reviewer should know

1. The original [README.md](README.md) is stale: it says FM and MF are `NotImplementedError` placeholders, while the actual algorithm files and artifacts implement FM, MF, and FM-LN.
2. [evaluate.py](evaluate.py#L21-L24) registers `mock`, `fm`, and `mf` but omits `fm_lognorm`, even though [train.py](train.py#L19-L24) and [inference_server.py](inference_server.py#L68-L91) support it.
3. The artifact name `fm_lognorm_rtx3060` is evidence of intended hardware provenance, not a cryptographic or experiment-manifest proof of external training.
4. JSONL files are append-only. Repeated train/evaluate invocations can place multiple runs in one metrics file; readers should group records by algorithm, configuration, and run provenance before drawing curves.
5. MF training is materially more expensive than FM because the JVP differentiates through the conditioned network, so comparing only epoch count or NFE is misleading.

## 13. A rigorous presentation order for a defense or interview

1. State the common data path and shared backbone.
2. Define the coupling $z_t=(1-t)x_0+t\epsilon$.
3. Derive $v=\epsilon-x_0$.
4. Explain FM as instantaneous-velocity regression plus reverse Euler.
5. Explain FM-LN as a change in the time sampling measure, not a new ODE.
6. Define MF's interval average and displacement identity.
7. Derive the JVP tangent $(v,0,1)$ and detached target.
8. Explain why the MF $r$ embedding adds 323 parameters.
9. Follow the generic trainer, checkpoint, sampler, and evaluator.
10. End with artifact provenance and fairness limitations.

The central conceptual sentence to remember is:

> FM learns a local direction; MF learns an interval displacement; FM-LN changes where FM spends its training effort along the same path.
