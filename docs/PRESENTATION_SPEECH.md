# Presentation Speech and Viva Guide

## A Controlled Comparison of Flow Matching and Mean Flow for Unconditional CIFAR-10 Image Generation

This is a full defense script, so reading every line would exceed 10 minutes. For a 10-minute delivery, use the first one or two paragraphs of each “Main speech,” state only the central displayed equations, and reserve every “Deeper mathematical explanation” and extended answer for the viva.

Suggested timing: slides 1-2, 30 seconds; slide 3, 45 seconds; slides 4-5, 75 seconds; slides 6-9, 3 minutes; slides 10-12, 75 seconds; slides 13-15, 90 seconds; slide 16 demo, 75 seconds; slide 17, 40 seconds; slides 18-20, 20 seconds. This leaves a small transition margin.

## Slide 1 - Welcome

### Main speech

Assalamu Alaikum and welcome to our CSE 4610 Design Project presentation. Our project studies modern flow-based generative models for unconditional image generation. In particular, we compare standard Flow Matching, Mean Flow, and a Flow Matching variant that changes the training-time distribution using logit-normal sampling.

The central question is simple: how does the definition of the learned velocity field affect training behavior, image quality, and the number of neural-network evaluations required during generation?

### Likely questions

**Why did you choose CIFAR-10?**

CIFAR-10 is small enough for controlled experimentation but still diverse enough to reveal differences in optimization and generation quality. Its 32 by 32 RGB images make repeated training, checkpoint sampling, and FID evaluation computationally practical.

**What does unconditional generation mean?**

The model generates an image without receiving a class label or text prompt. It learns the marginal image distribution rather than a class-conditional distribution.

## Slide 2 - Project title and team

### Main speech

Our title is “A Controlled Comparison of Flow Matching and Mean Flow for Unconditional CIFAR-10 Image Generation.” The team members are Sheikh Mosheul Akbar, Md. Samiul Islam, and Rad Shahmad Daiyan.

Our implementation uses a shared time-conditioned U-Net and changes the algorithm wrapper around it. This is important because the network architecture alone does not decide whether the output represents an instantaneous velocity or an interval-average velocity. That meaning comes from the training objective.

### Transition

Before introducing our models, we first need to establish how diffusion-based generation works and why flow-based methods are related to it.

## Slide 3 - Related work

### Main speech

“Our comparison is built on two foundational papers. First, Lipman and colleagues introduced *Flow Matching for Generative Modeling* at ICLR 2023. Their key idea is to train a continuous normalizing flow without simulating the ODE during training. A conditional probability path is fixed, and the network regresses its instantaneous vector field. At inference, that learned field is integrated numerically from noise to data.

Second, Geng and colleagues introduced *Mean Flows for One-step Generative Modeling* in 2025. Instead of predicting only the instantaneous velocity at one time, Mean Flow predicts the average velocity across an interval. The Mean Flow Identity relates that interval average to the instantaneous field and introduces a Jacobian-vector product in training. This permits a direct noise-to-image displacement when the interval spans from one to zero.

Our contribution is not a claim to have invented either base algorithm. We implement Flow Matching, logit-normal time-sampled Flow Matching, and Mean Flow within one CIFAR-10 codebase and evaluate their quality, training behavior, and inference cost using one reporting pipeline.”

### Mathematical distinction

Flow Matching learns a local field,

\[
v_\theta(z_t,t) \approx \frac{d z_t}{dt}.
\]

Mean Flow learns an interval average,

\[
u_\theta(z_t,r,t) \approx \frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,d\tau.
\]

The first normally requires several numerical integration steps. The second can represent the total interval displacement directly.

### Likely questions

**Q: Is Mean Flow simply Flow Matching with one Euler step?**  
No. A one-step Euler update uses an instantaneous velocity over a large interval. Mean Flow is trained to predict the interval-averaged velocity, so its target and training identity are different.

**Q: What is original in this project?**  
The project contribution is a transparent implementation and empirical comparison on CIFAR-10, including a logit-normal time-sampling variant of Flow Matching. We do not claim a new foundational generative algorithm.

**Q: Why are these papers comparable?**  
Both describe deterministic velocity-based transport between data and noise. They differ primarily in whether the model learns a local instantaneous field or an interval-average field.

### Transition

“With those two sources established, I will first connect them to the more familiar diffusion viewpoint.”

## Slide 4 - What diffusion models do

### Main speech

A diffusion model defines a forward process that gradually corrupts data with Gaussian noise and a learned reverse process that reconstructs data from noise.

A common closed-form forward marginal is

\[
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

Here, \(x_0\) is a real image, \(x_t\) is its noisy version, and \(\bar\alpha_t\) controls the signal-to-noise ratio. A conventional DDPM learns a denoising or noise-prediction model and applies many reverse transitions.

Our project is related to diffusion because it also connects data and Gaussian noise over time. However, it does not implement the standard stochastic DDPM reverse chain. It learns a deterministic vector field and solves an ordinary differential equation, or ODE, from noise back to data.

### Likely questions

**Is Flow Matching a diffusion model?**

It belongs to the broader family of continuous-time generative models and can use diffusion-like probability paths, but this repository does not implement a standard DDPM. It learns a deterministic velocity field for an ODE using a straight interpolation between data and noise.

**What is the difference between an ODE and the DDPM reverse chain?**

An ODE evolves a state deterministically once the initial noise is fixed. A DDPM reverse chain normally includes stochastic transitions. ODE sampling can still be discretized into many numerical steps, but each step follows a learned deterministic vector field.

## Slide 5 - Flow Matching as continuous transport

### Main speech

Flow Matching learns how probability mass should move continuously between two distributions. In our implementation, the coupling between a real image \(x_0\) and independent Gaussian noise \(\epsilon\) is the straight path

\[
z_t=(1-t)x_0+t\epsilon, \qquad t\in[0,1].
\]

At \(t=0\), \(z_0=x_0\), so we are at the data. At \(t=1\), \(z_1=\epsilon\), so we are at Gaussian noise.

Differentiating the path gives the conditional velocity:

\[
\frac{d z_t}{dt}=-x_0+\epsilon=\epsilon-x_0.
\]

The velocity is constant for an individual straight-line pair, although the marginal vector field learned by the neural network is state- and time-dependent because many different data-noise pairs can pass through nearby regions.

The learned ODE is

\[
\frac{dz}{dt}=v_\theta(z,t).
\]

Training defines the path from data to noise. Generation begins from the known noise distribution at \(t=1\) and solves the ODE backward toward \(t=0\).

### Deeper mathematical explanation

The distribution \(p_t\) transported by a velocity field satisfies the continuity equation

\[
\frac{\partial p_t(x)}{\partial t}
+\nabla\cdot\left(p_t(x)v_t(x)\right)=0.
\]

Flow Matching avoids directly evaluating this divergence during the basic regression objective. It samples points on a conditional path and regresses the network toward the corresponding conditional velocity. Under the standard conditional Flow Matching argument, the conditional regression produces the appropriate marginal velocity after averaging over pairs consistent with \((z_t,t)\).

### Likely questions

**Why is the target \(\epsilon-x_0\)?**

Because it is the exact derivative of the selected interpolation. Differentiating \((1-t)x_0+t\epsilon\) with respect to \(t\) gives \(-x_0+\epsilon\).

**Why is the generation direction reversed?**

The easy distribution to sample is Gaussian noise at \(t=1\). The data distribution is at \(t=0\). Therefore, generation integrates the learned field with a negative time increment.

**Does every sample follow the same straight line?**

Each sampled data-noise pair defines its own straight conditional path. The neural network learns the marginal field obtained from the population of such conditional paths.

## Slide 6 - Standard Flow Matching

### Main speech

Standard Flow Matching samples time uniformly and trains the network using mean squared error:

\[
\mathcal L_{FM}(\theta)=
\mathbb E_{x_0,\epsilon,t}
\left[
\left\|f_\theta(z_t,t)-(\epsilon-x_0)\right\|_2^2
\right],
\qquad t\sim\mathcal U(0,1).
\]

The model therefore learns a local, instantaneous direction at each state and time.

At inference, we start with \(z_1\sim\mathcal N(0,I)\). With \(N\) function evaluations and step size \(h=1/N\), the reverse-time explicit Euler update used by the code is

\[
z_{t-h}=z_t-h f_\theta(z_t,t).
\]

The minus sign appears because time moves from 1 toward 0. Each step calls the backbone once, so for FM the number of function evaluations, or NFE, is approximately the number of neural-network calls per generated sample.

### Likely questions

**Why use MSE?**

For a fixed \((z_t,t)\), minimizing MSE makes the optimal prediction the conditional expectation of the target velocity. This gives the regression interpretation behind conditional Flow Matching.

**Is the sampler backward Euler?**

It is best described as explicit Euler applied in reverse time. The implementation evaluates the field at the current state and uses a negative time step. Calling it “backward Euler” can be confusing because implicit backward Euler evaluates the field at the next state.

**Why does increasing NFE help FM?**

The neural field may be accurate, but numerical integration introduces discretization error. Smaller steps usually approximate the continuous trajectory better, although the improvement can saturate when modeling error dominates.

## Slide 7 - Mean Flow as interval displacement

### Main speech

Flow Matching predicts a local velocity. Mean Flow instead predicts the average velocity over an interval \([r,t]\):

\[
u(z_t,r,t)=\frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,d\tau.
\]

Multiplying both sides by the interval length gives the displacement identity:

\[
(t-r)u(z_t,r,t)=z_t-z_r.
\]

Rearranging gives the sampling update

\[
z_r=z_t-(t-r)u(z_t,r,t).
\]

This is the reason Mean Flow is attractive for low-NFE generation. Instead of approximating a long interval with many local Euler steps, it attempts to predict the average displacement over that interval directly.

For one-step generation, \(t=1\) and \(r=0\), giving

\[
z_0=z_1-u(z_1,0,1).
\]

However, one-step generation is not automatically exact. The identity is exact for the true mean field. A learned neural approximation still has modeling error.

### Likely questions

**What is the main conceptual difference between FM and MF?**

FM learns a local derivative. MF learns an average displacement over a requested interval.

**Why does Mean Flow need both \(r\) and \(t\)?**

The required average displacement depends on the interval. The same current state at time \(t\) can require a different prediction depending on how far back we want to move, which is specified by \(r\).

**Is one-step Mean Flow exact?**

Only if the learned mean velocity equals the true interval-average velocity. The update itself has no internal Euler discretization, but neural approximation error remains.

## Slide 8 - Mean Flow Identity and JVP

### Main speech

The Mean Flow target is constructed from the Mean Flow Identity. Begin with

\[
(t-r)u(z_t,r,t)=z_t-z_r.
\]

Differentiate both sides with respect to \(t\), keeping \(r\) fixed and moving \(z_t\) along the trajectory. The product rule gives

\[
u+(t-r)\frac{du}{dt}=v.
\]

Therefore,

\[
u=v-(t-r)\frac{du}{dt}.
\]

Because \(u\) depends on the state \(z_t\), the left endpoint \(r\), and the current time \(t\), the total derivative is

\[
\frac{du}{dt}
=\frac{\partial u}{\partial z_t}\frac{dz_t}{dt}
+\frac{\partial u}{\partial r}\frac{dr}{dt}
+\frac{\partial u}{\partial t}\frac{dt}{dt}.
\]

During this differentiation, \(r\) is fixed, so \(dr/dt=0\), \(dz_t/dt=v\), and \(dt/dt=1\). Thus the directional derivative uses the tangent

\[
(v,0,1).
\]

The implementation obtains this derivative with a Jacobian-vector product, or JVP, without materializing the complete Jacobian.

The target is

\[
u_{target}=\operatorname{stopgrad}
\left(v-(t-r)\frac{du}{dt}\right),
\]

and the loss is

\[
\mathcal L_{MF}=\mathbb E\left[
\|u_\theta-u_{target}\|_2^2
\right].
\]

The stop-gradient prevents the target construction from creating a target-side or higher-order gradient path. Gradients flow through the predicted \(u_\theta\).

The shared backbone has 6,352,899 parameters. Mean Flow adds a small embedding for \(r\). Its parameter count is

\[
(1\times64+64)+(64\times3+3)=128+195=323.
\]

### Likely questions

**What is a JVP?**

For a function \(f(x)\) and a direction \(a\), the JVP computes \(J_f(x)a\), the directional derivative of \(f\) along \(a\). It avoids constructing the entire Jacobian matrix.

**Why is the tangent exactly \((v,0,1)\)?**

The state changes with derivative \(v\), the left endpoint \(r\) is fixed, and the current time changes with derivative 1.

**Why sometimes set \(r=t\)?**

This includes the diagonal or zero-length interval. In the limiting case, the mean velocity reduces to the instantaneous velocity, providing an anchoring condition during training. The code uses this case with probability 0.25.

**Why detach the target?**

The constructed value is treated as a regression target. Detaching prevents optimization through the target and avoids an unintended higher-order feedback path.

**Is MF training as cheap as FM training?**

No. MF uses a JVP during training and has additional conditioning. Low-NFE inference does not imply equally cheap training.

## Slide 9 - Logit-normal Flow Matching

### Main speech

Our Flow Matching with logit-normal time sampling changes one training choice. Standard FM uses

\[
t\sim\mathcal U(0,1).
\]

FM-LN instead samples

\[
s\sim\mathcal N(0,1),
\qquad t=\sigma(s)=\frac{1}{1+e^{-s}}.
\]

The resulting density is

\[
p(t)=\frac{1}{\sqrt{2\pi}\,t(1-t)}
\exp\left[-\frac12
\left(\log\frac{t}{1-t}\right)^2\right],
\qquad 0<t<1.
\]

This distribution is symmetric around 0.5 and samples intermediate times more frequently than points extremely close to 0 or 1.

The path remains \(z_t=(1-t)x_0+t\epsilon\). The target remains \(\epsilon-x_0\). The MSE loss form and reverse-time Euler sampler are also unchanged.

Therefore, FM-LN is not a new ODE. It changes how the empirical training objective allocates samples across time. Since the code does not apply importance weights to recover the uniform-time expectation, it is a genuine reweighting of the training objective.

### Likely questions

**Why might intermediate times be useful?**

Near the endpoints, inputs can be almost clean data or almost pure noise. Intermediate states contain both signal and corruption, which can provide informative gradients for learning transport. This is a motivation, not a proof that it will always improve performance.

**Does FM-LN change inference?**

No. It uses the same reverse-time Euler sampler as standard FM. Any difference at inference comes from the learned field, not a new sampler.

**Is this importance sampling?**

Not in the strict unbiased-estimation sense, because the implementation does not divide by the sampling density or otherwise correct back to a uniform-time objective. It changes the effective weighting of training times.

**Does a different time density change the ideal vector field?**

If the model had infinite capacity and optimized every time independently with full support, the pointwise conditional target is unchanged. In finite training, changing how frequently times appear changes optimization emphasis and approximation quality.

## Slide 10 - Shared backbone

### Main speech

All three methods use the same time-conditioned SimpleUNet backbone. It receives an image-like state and a scalar time, encodes the time with sinusoidal features, processes information through encoder, bottleneck, and decoder blocks, and outputs a three-channel field with the same spatial dimensions.

The backbone does not know the semantic meaning of that field. Under FM, it is trained as an instantaneous velocity. Under FM-LN, it has the same target but sees a different distribution of times. Under MF, it represents an interval-average velocity and also receives information about \(r\) through the additional embedding.

The input images are CIFAR-10 RGB images resized to 32 by 32 and normalized approximately from \([0,1]\) to \([-1,1]\). No class labels are supplied.

### Likely questions

**Does using the same backbone make the comparison fully fair?**

It controls architecture construction, but it does not make the recorded runs fully matched. Batch size, epoch count, learning rate, and training cost must also be compared.

**Why use a U-Net?**

Its multi-scale encoder-decoder structure and skip connections preserve both global context and local spatial detail, which is useful for image-to-image vector-field prediction.

**Are the total parameter counts identical?**

The backbone count is identical. MF has 323 additional parameters for the \(r\) embedding, so the total trainable count is not literally identical.

## Slide 11 - Algorithm-agnostic pipeline

### Main speech

The project separates infrastructure from algorithm mathematics. Configuration, CIFAR-10 loading, backbone construction, training, checkpointing, sampling, evaluation, and the browser demo are shared.

Each algorithm mainly provides two operations:

1. `training_step(batch)`, which returns a scalar loss.
2. `sample(n, NFE, device)`, which returns generated images.

The experiment runner builds the shared objects and calls the selected algorithm through this interface. This design reduces duplicated code and helps ensure that differences come from the algorithm wrapper rather than separate training pipelines.

MF checkpoints contain both the backbone state and the \(r\)-embedding state. The loader must restore these modules in the same order used during saving.

### Likely questions

**Why is modularity scientifically useful?**

It reduces accidental differences in preprocessing, model construction, checkpointing, and evaluation. It also makes algorithm-specific assumptions easier to audit.

**What does one checkpoint contain?**

It contains state dictionaries for algorithm-owned trainable modules, optimizer and scheduler state where applicable, counters, and the serialized experiment configuration.

## Slide 12 - Evaluation and fairness

### Main speech

The saved runs use the same dataset and backbone family but different training budgets.

The recorded FM run uses 100 epochs, batch size 128, and learning rate \(2\times10^{-4}\). FM-LN uses 100 epochs, batch size 32, and the same learning rate. Mean Flow uses 30 epochs, batch size 32, and learning rate \(5\times10^{-5}\).

Evaluation is recorded at NFE 1, 5, 10, and 20 using 1,000 generated images.

Therefore, the results accurately describe these checkpoints, but they are not a causal proof that one algorithm is universally superior. A strictly controlled comparison would match samples seen, optimizer schedule, compute budget, checkpoint-selection policy, evaluation reference, and random seeds.

### Likely questions

**Why is epoch count not enough for fairness?**

An epoch depends on batch size and dataset traversal, while training cost also depends on the algorithm. MF includes a JVP, so one MF update may cost more than one FM update. Samples seen and GPU-hours are stronger budget measures.

**Why evaluate only 1,000 images?**

It is computationally practical for the project, but it makes FID and IS noisier than standard large-sample evaluation. A stronger final evaluation should use around 50,000 generated images and multiple seeds.

**Can you claim FM-LN beats MF?**

We can say FM-LN has the best recorded result among these saved configurations. We cannot claim universal algorithmic superiority without matched runs.

## Slide 13 - Training-loss findings

### Main speech

The logged training loss decreases for both FM variants. From the first to the last retained epoch record, standard FM decreases by approximately 44.1 percent and FM-LN decreases by approximately 30.9 percent.

The Mean Flow loss increases by approximately 17.1 percent over its recorded run. This suggests that the current MF configuration is optimization-limited or unstable. It does not prove that the Mean Flow method itself is fundamentally ineffective.

The loss scales should also not be compared as if they were identical objectives. FM regresses an instantaneous velocity, while MF uses a JVP-constructed interval target. The most defensible interpretation is the within-run trend and its agreement with generated samples.

### Likely questions

**Why can FM-LN have a smaller absolute loss than FM?**

The time-sampling distributions and recorded batch sizes differ. Because the objectives weight time differently, absolute loss values are not a perfectly controlled cross-method metric.

**Why might MF loss rise?**

Possible causes include an unstable self-referential JVP target, learning-rate choice, insufficient epochs, interval-sampling difficulty, target variance, or the extra conditioning pathway. The recorded artifacts do not isolate a single cause.

**How were duplicate epochs handled?**

The JSONL logs are append-only and can contain repeated epoch records. The plotted canonical history retains the last record for a duplicated epoch.

## Slide 14 - Quantitative generation findings

### Main speech

We evaluate generation using Fréchet Inception Distance, or FID. FID approximates real and generated Inception features as Gaussians and computes

\[
\operatorname{FID}=
\|\mu_r-\mu_g\|_2^2+
\operatorname{Tr}
\left(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\right).
\]

Lower is better because the generated feature distribution is closer to the real feature distribution.

At 20 NFE, FM-LN records the best FID of 72.66, while standard FM records 76.10. Mean Flow's best displayed value is 210.99 at one NFE. Increasing the number of Mean Flow intervals degrades this checkpoint rather than refining it.

The project also records Inception Score. For FM-LN at 20 NFE, the recorded score is approximately \(6.18\pm0.52\), compared with \(6.12\pm0.72\) for FM. Mean Flow at one NFE records approximately \(2.54\pm0.21\).

Inception Score is based on

\[
\exp\left(
\mathbb E_x
\left[\operatorname{KL}(p(y\mid x)\|p(y))\right]
\right).
\]

It rewards confident individual predictions and diversity across predicted classes. Higher is generally better, but it does not compare generated images directly with real images, so it should not replace FID or visual inspection.

### Likely questions

**Why can Mean Flow become worse with more steps?**

The model is trained to predict interval averages, but approximate interval predictions may not compose consistently. Multiple updates can accumulate error or move states into regions not well represented during training. More NFE is therefore not guaranteed to help an imperfect MF checkpoint.

**Why is FID lower better?**

FID is a distance between fitted feature distributions. Zero would indicate identical fitted means and covariances, although finite samples and feature limitations prevent interpreting it as a complete perceptual measure.

**Is FID 72.66 good?**

It demonstrates learning and is the best among the recorded runs, but it is not competitive with large state-of-the-art CIFAR-10 systems. The model is small, unconditional, and evaluated using only 1,000 generated images.

## Slide 15 - Qualitative findings

### Main speech

The sample grids support the quantitative trend. Standard FM produces recognizable CIFAR-10-like objects and varied scenes. FM-LN is slightly cleaner in the final grid. The Mean Flow checkpoint retains substantial high-frequency noise and weak object structure.

The epoch progression uses fixed grid positions, so changes are easier to compare across checkpoints. These images are qualitative evidence; they support the metric results but do not replace quantitative evaluation.

### Likely questions

**Were the same seeds used for visual comparison?**

The checkpoint progressions use fixed grid positions and are intended for consistent visual tracking. During the live demo, we explicitly keep the inference seed fixed when comparing NFE settings.

**Can visual quality alone prove one model is better?**

No. Small selected grids can be unrepresentative. We combine visual inspection with FID and IS and disclose the evaluation limitations.

**Why are the images blurry?**

CIFAR-10 images are only 32 by 32 pixels, the backbone is relatively small, and the training runs are limited. Upscaling them for presentation makes their low native resolution more visible.

## Slide 16 - Live demo

### Main speech

For the demo, we start the local inference server and open the browser interface.

First, on the training-history view, we select an algorithm and play the saved epochs. The image grid and the endpoint of the loss curve should move to the same epoch. This verifies that the visualization is synchronized with the logged checkpoint sequence.

Second, on the inference view, we select a trained model, checkpoint, NFE, image count, and random seed. We generate a sample batch, then keep the seed fixed and change NFE. Holding the seed constant isolates the effect of the inference budget rather than changing the starting noise.

The command is:

```powershell
.\venv\Scripts\python.exe .\inference_server.py
```

Then we open `http://localhost:8000`.

### Demo discipline

- Start the server before the presentation.
- Preload the page once so model discovery is complete.
- Use a small image count to avoid delay.
- Keep one known-working checkpoint selected.
- Use the same seed for comparisons.
- If generation is slow, explain that NFE counts network calls and show the prepared result grids.

### Likely questions

**What is the input to inference?**

The main stochastic input is Gaussian noise determined by the random seed. The user also selects the trained model, checkpoint, NFE, and number of images. There is no text prompt or class label.

**What is the output?**

The output is a batch of generated RGB images, converted from the model's normalized \([-1,1]\) range into displayable pixel values.

**Why does training history begin at epoch 10?**

Epoch 10 is the first saved trained checkpoint for those runs. Epoch 0 can be represented as pure starting noise, but there are no trained checkpoint images for epochs 1 through 9.

## Slide 17 - Future direction

### Main speech

We propose two future directions.

The first is Adaptive MeanFlow for Dynamic NFE. Instead of using the same number of inference steps for every sample, the system would estimate sample difficulty or confidence and allocate more refinement only where needed. Easy samples could preserve one-step or low-step efficiency, while difficult samples could receive additional corrections.

This requires a stopping or confidence criterion. Possible signals include the norm of the predicted residual displacement, disagreement between consecutive predictions, estimated local truncation error, or an auxiliary confidence head. The main research question is whether adaptive computation improves expected image quality without losing Mean Flow's low average inference cost.

The second direction is Multi-Scale MeanFlow for high-resolution generation. A coarse-to-fine system would first generate global structure at low resolution, then refine progressively at higher resolutions. This reduces the cost of operating at full spatial resolution throughout the entire trajectory.

The key question is whether Mean Flow's low-NFE advantage survives across scales. Each stage must preserve consistency with the previous resolution, and evaluation should report both quality and end-to-end computation.

### Likely questions

**How would you measure sample difficulty?**

We could use the magnitude or change of the predicted displacement, consistency between one large jump and two smaller jumps, a learned confidence estimator, or an image-quality proxy. The criterion must be calibrated without using unavailable ground-truth images at inference.

**Would dynamic NFE make batching difficult?**

Yes. Different samples may stop at different times, causing branch divergence and inefficient GPU batches. A practical design could group active samples by remaining steps or use a small set of discrete budgets such as 1, 2, 4, and 8 NFE.

**How would Multi-Scale MeanFlow work?**

One option is a cascade: generate a low-resolution sample, upsample it, and condition a higher-resolution Mean Flow model on the coarse image. Another option is a single multi-scale network with resolution-specific heads. The cascade is simpler to evaluate and train independently.

**How would you test whether the low-NFE advantage is preserved?**

We would compare quality against total network evaluations, wall-clock latency, memory, and floating-point operations at each resolution, using matched baselines and multiple seeds.

## Slide 18 - GitHub

### Main speech

This slide is reserved for the final GitHub repository link. Before submission, we will place the verified public URL here. The repository should include source code, configuration files, checkpoint-loading instructions, evaluation commands, and clear notes about which artifacts are included.

### Likely questions

**What is needed for reproducibility?**

Exact configuration snapshots, random seeds, dependency versions, dataset preprocessing, checkpoint files or download instructions, evaluation sample count, real-statistics cache details, and commands for training, evaluation, and inference.

**Are all large checkpoints stored directly in Git?**

Large binary checkpoints should normally use Git LFS or an external release/download location rather than ordinary Git history. The repository should provide hashes and clear paths so the UI can discover them.

## Slide 19 - Questions and answers

### Main speech

Thank you for listening. We welcome questions about the flow mathematics, training objectives, experimental design, evaluation, and live inference system.

## Slide 20 - Thank you

### Main speech

Thank you. This project demonstrated one shared generative pipeline with three interpretations of the learned field: Flow Matching learns a local direction, Mean Flow learns an interval displacement, and logit-normal Flow Matching changes where standard FM spends its training effort along the same path.

Our recorded results favor FM-LN at 20 NFE, while the current Mean Flow run remains optimization-limited. The most important conclusion is therefore both technical and experimental: changing the objective can change low-step behavior, but strong comparisons require matched compute, multiple seeds, and larger evaluation sets.

# High-probability viva questions

## 1. Give the entire project in one minute.

The project is an unconditional CIFAR-10 image generator using a shared time-conditioned SimpleUNet. Standard FM predicts instantaneous velocity along a straight data-to-noise path and generates by reverse-time Euler integration. FM-LN keeps the same path, target, loss form, and sampler but samples training time from a logit-normal distribution. Mean Flow predicts interval-average velocity and uses a JVP-derived target so it can attempt low-NFE or one-step displacement. The recorded FM-LN checkpoint gives the best 20-NFE FID, while the Mean Flow run shows rising loss and noisy samples. Because the saved training budgets differ, these are checkpoint observations rather than a universal ranking.

## 2. State the single most important conceptual sentence.

FM learns a local direction; MF learns an interval displacement; FM-LN changes where FM spends its training effort along the same path.

## 3. Derive the Flow Matching target.

Start from

\[
z_t=(1-t)x_0+t\epsilon.
\]

Differentiate:

\[
\frac{dz_t}{dt}=-x_0+\epsilon=\epsilon-x_0.
\]

Therefore, regress \(f_\theta(z_t,t)\) toward \(\epsilon-x_0\) with MSE.

## 4. Derive the Mean Flow Identity.

Start from

\[
(t-r)u=z_t-z_r.
\]

Differentiate with respect to \(t\), holding \(r\) fixed:

\[
u+(t-r)\frac{du}{dt}=\frac{dz_t}{dt}=v.
\]

Rearrange:

\[
u=v-(t-r)\frac{du}{dt}.
\]

The derivative is computed using the JVP tangent \((v,0,1)\).

## 5. Why does FM-LN emphasize the middle?

The sigmoid maps a standard normal variable to \((0,1)\). Values of the normal variable are most concentrated near zero, and \(\sigma(0)=0.5\). Very small or very large times require large-magnitude normal samples, so they occur less frequently.

## 6. What exactly remains unchanged in FM-LN?

The interpolation \(z_t\), target \(\epsilon-x_0\), MSE form, shared backbone, and reverse-time Euler sampler. Only the distribution used to draw training time changes.

## 7. Why can more NFE hurt Mean Flow?

An approximate interval field need not be compositionally consistent. Splitting one interval into several predicted intervals can accumulate approximation error and expose the model to off-distribution intermediate states. More steps only help when the learned predictions compose accurately.

## 8. What are the main threats to validity?

One seed, 1,000-image metric estimates, non-matched batch sizes, non-matched epoch counts and learning rates, different training costs, append-only JSONL logs, and limited checkpoint provenance. These prevent a universal causal ranking.

## 9. Why not compare raw training loss directly across methods?

FM and MF optimize different targets, and FM-LN uses a different time weighting. Absolute loss magnitudes therefore have different meanings. Within-run convergence trends and downstream samples are more defensible.

## 10. What would a stronger experiment look like?

Use at least three seeds; match architecture, data order, samples seen, optimizer schedule, and GPU-hours; evaluate the same selected checkpoints with 50,000 generated images; report FID, IS, latency, throughput, memory, and uncertainty; and preserve fixed-seed visual comparisons.

# Final delivery advice

- Memorize the three central differences, not the entire script.
- Write the FM target, MF displacement identity, Mean Flow Identity, and logit-normal transformation on a board if asked.
- Never say that one-step Mean Flow is unconditionally exact.
- Never claim that the recorded runs prove universal superiority.
- Say “reverse-time explicit Euler” rather than “implicit backward Euler.”
- When discussing FM-LN, emphasize that inference is unchanged.
- When discussing the percentages, say they compare the first and last retained logged epochs.
- During the demo, keep the seed fixed when changing NFE.
