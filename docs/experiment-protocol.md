# Metal Cloud Renderer: Experiment Protocol

This protocol applies to every comparison between `pt`, `analytic` and `neural`. Scene, renderer and RTE definitions
are in [architecture.md](architecture.md).

## 1. Scenes and splits

All scenes share: AABB [-500,0,-500]–[500,800,500] m, grid 256×192×256, albedo 0.99, $g=0.8$, film 960×540,
scene `version` 3.

| Scene file | Cloud | Role |
|---|---|---|
| `scenes/cumulus.json` | Hand-designed 11-lobe cumulus (the article's look), $\sigma_{t,max}=0.05$ | Sun sweep (§8); same-cloud track |
| `scenes/cumulus_heldout.json` | `makeRandomCumulus(2500)`, a validation-range shape never used for training | Generalization track |

Cameras: `front` and `side` (horizontal views from 2300 m, 90° apart) and `below` (looking up at the base).

Two neural tracks, reported separately:

| Model | Training data | Valid comparison |
|---|---|---|
| `cumulus_single` (`configs/neural_single*.json`) | `scenes/cumulus.json` itself with random suns (`--fixed-shape`), the article's single-cloud setup | Same-cloud only: the sun sweep. Its results on `cumulus.json` are **not** generalization results. |
| `cumulus_net` (`configs/neural*.json`) | Random shapes, seeds 1000–1999; validation seeds 2000–2999 | Both scenes; `cumulus.json` is hand-designed (not from the generator) and `cumulus_heldout.json` is a held-out seed |

Shape seed ranges (random-shape mode): evaluation 1–999 (reserved), training 1000–1999, validation and held-out
2000–2999. Per training shape the sun (elevation −10…90°, any azimuth) and $\sigma_{t,max}$ (log-uniform
[0.02, 0.1]) are random.

Rules:

- The generalist never trains on evaluation or held-out seeds; the model sidecar and every bench record store the seed
  range, and fixed-shape datasets are flagged in their header and sidecar.
- Any change to a scene file requires a bump of its `version`. Results from different scene versions are never compared.

## 2. Reference generation

For each scene/camera configuration of the experiment (`experiments/eval.json`: 2 scenes × 2 cameras):

1. Render two independent halves A and B with `pt`, using seeds 1 and 2 and N spp each, starting at N = `reference.initial_spp` of the experiment (1024 in `experiments/eval.json`).
2. The reference is $R=(A+B)/2$.
3. The estimated reference error is $\widehat{\mathrm{relMSE}}(R)=\mathrm{relMSE}(A,B)/4$. A−B has twice the per-half variance, and averaging the halves halves it, so the ratio is 1/4.
4. Accept the reference only if $\widehat{\mathrm{relMSE}}\le10^{-4}$ **and** $\widehat{\mathrm{relMSE}}\le0.1\times$ the smallest relMSE of any compared method on that configuration. Otherwise re-render with N' = the smallest power of two $\ge\max(2N,\ N\,\widehat{\mathrm{relMSE}}/\text{target})$ (the estimate falls as 1/N), capped at `reference.max_spp` (65536).
5. Store `ref.pfm`, `refA.pfm`, `refB.pfm`, `ref.png`, `convergence.csv` and `ref.json` (N, seeds, estimated error, `meets_abs_gate`, convergence slope, wall and GPU time, bounce-cap hits, renderer version). `cloudnr suite` keeps references in `results/refs/<scene>_<camera>/` and reuses one only if scene version, camera, resolution, seeds, pt settings and renderer version match.

An unaccepted reference is never used as ground truth: if the cap is reached first, the suite still writes the
metrics but marks the configuration **NOT ACCEPTED** in the report, CSV (`reference_accepted`) and records, and exits
with status 2.

## 3. Image metrics

Let $x$ be the test image and $r$ the reference. Both are linear RGB with $P$ pixels, and the exposure
$2^{EV}$ is applied to both before evaluation.

- **relMSE** on linear radiance: $\mathrm{relMSE}=\frac{1}{3P}\sum_{p,c}\frac{(x_{pc}-r_{pc})^2}{r_{pc}^2+0.01}$.
- **PSNR** on display-mapped images in [0,1] (Reinhard plus sRGB OETF, architecture §4): $\mathrm{PSNR}=10\log_{10}(1/\mathrm{MSE}_{display})$.
- **SSIM** on the same display-mapped images: 11×11 Gaussian window with $\sigma=1.5$, $K_1=0.01$, $K_2=0.03$, dynamic range $L=1$, computed per channel and averaged over channels and pixels.
- **Heat maps**: PNGs of absolute error $|x-r|$ and relative error $|x-r|/(|r|+0.1)$, averaged over channels.

The C++ host (`src/metrics.{h,mm}`) is the only implementation, used by `cloudnr compare` and `bench`. Its unit tests
(`tests/test_*.mm`) check identical images (relMSE 0, PSNR ∞, SSIM 1), a known constant offset with the closed-form
relMSE and PSNR, and that the display mapping matches the PNG writer on a fixed set of values.

## 4. Timing

The protocol measures steady state only.

- A frame is one command buffer that contains only that renderer's dispatches for one full image. For `pt`, a frame is 1 spp.
- At most 2 frames are in flight, synchronized with `MTLSharedEvent`. There is no readback during measurement.
- Each run has 30 warm-up frames, then 300 measured frames.
- GPU time per frame is `GPUEndTime − GPUStartTime` from the `MTL4CommitFeedback` of that commit.
- Report the median, p5 and p95 per run. Do 3 runs. The headline number is the median of the three run medians.
- CPU preparation time is `std::chrono::steady_clock` around encode and commit, reported the same way.
- For the path tracer, report GPU ms per spp, and the *equal-quality spp*: the spp at which the progressive `pt` relMSE first falls to the relMSE of each other method. The time to reach that quality is equal-quality spp × ms/spp.

These costs are excluded from steady state and reported separately:

| Cost | Measured as |
|---|---|
| Pipeline compile | Wall time of `MTL4Compiler` library and pipeline creation |
| Density bake | GPU time of the bake and mip-generation commit |
| Model load | Wall time of `.cnrm` read, validation and buffer upload |
| Reference accumulation | Wall and GPU time recorded in `ref.json` |

Conditions: AC power, Low Power Mode off, the display awake, no other GPU applications running, and
`MTL_DEBUG_LAYER` and `MTL_SHADER_VALIDATION` unset. Validation runs are separate and are never timed.

## 5. Memory

- **Renderer memory**: the sum of `allocatedSize` over every resource the renderer owns (pipelines are excluded; buffers, textures and the MLP weight buffer are included).
- **Density memory**: the shared density texture with its mips, reported separately because every method uses it.
- Also report the delta in `MTLDevice.currentAllocatedSize` around renderer setup.

## 6. Benchmark record

`cloudnr bench` prints one JSON record (and writes it with `--out`); `cloudnr suite` writes
`<out>/records/<scene>_<camera>_<config>_<timestamp>.json`, where `<config>` is the config file stem (so two configs
of the same renderer stay distinct). Keys:

| Key | Content |
|---|---|
| `renderer_version` | `git describe --always --dirty` captured at CMake configure time, or `"nogit"` |
| `renderer` | `pt`, `analytic` or `neural` |
| `config` | The full `RenderConfig` object |
| `scene`, `scene_version`, `camera` | Scene name, its `version`, and the camera identifier |
| `width`, `height` | Output resolution |
| `gpu_name`, `os_version` | `MTLDevice.name` and the OS version string |
| `spp_or_quality` | spp for `pt`; step counts (and octaves) for `analytic`/`neural` |
| `seed` | RNG seed, or null for deterministic renderers |
| `gpu_ms` | `{median, p5, p95, runs}`, where `runs` lists the per-run statistics |
| `cpu_prep_ms` | Same structure as `gpu_ms` |
| `memory_bytes` | `{renderer, density, device_delta}` |
| `excluded_costs_ms` | `{compile, bake, model_load}` |
| `model` | `{path, format_version, sha256, train_seed_range}` or null |
| `metrics` | `{relMSE, psnr, ssim}` against `reference_id` |
| `reference_id` | The path of the accepted `ref.json` used for `metrics` |
| `bias` | `{bounce_cap_hits, mean_bounces}` for `pt`; `{notes}` naming the approximation sources otherwise |
| `timing`, `validation_enabled`, `gpu_ms_per_spp` (pt), `config_path` | Timing parameters, debug-layer state (must be false), pt cost per sample, config file |
| `reference_accepted`, `reference_spp_per_half`, `reference_est_relMSE`, `equal_quality_pt_spp` | Added by `suite` |

Percentiles are nearest-rank (rank $=\lceil p\,n/100\rceil$); the median of an even count is the lower median.

`cloudnr suite --experiment experiments/eval.json` runs the fixed configuration (scenes, cameras, method configs,
reference gates, timing and equal-quality settings all come from that file) and exports: `records/*.json`,
`summary.csv` (one row per scene/camera/method), `report.md` (environment, per-configuration tables, excluded costs,
bias sources, neural model provenance), `images/` (method, reference and heat-map PNGs plus method PFMs) and a copy of
the experiment file. Equal-quality spp uses a progressive `pt` run with `equal_quality.seed` (independent of the
reference seeds) at power-of-two checkpoints; its GPU time is the cumulative GPU time of that run. `--quick`
(3+20 frames, 1 run, references 64–256 spp) is a smoke test and its report is labelled not protocol-compliant.

## 7. Reporting rules

- Compare methods only at their recorded configurations. Label any comparison where quality settings differ.
- Never use an unaccepted reference (§2).
- Report bias sources next to the results: bounce-cap hits (`pt`), march discretization, early termination and the octave model (`analytic`), and Monte Carlo noise in the training targets (`neural`).
- Report neural results with the model version, dataset seed ranges (or the fixed-shape flag), and validation loss/correlation. Never present `cumulus_single` results on `scenes/cumulus.json` as generalization results.
- Report excluded costs (§4) in a separate table, never added to steady-state numbers.

## 8. Sun sweep (the article's evaluation)

`cloudnr sweep --experiment experiments/sweep.json` swings the sun around the cloud and scores each method against
a path-traced reference at every position, as in the article.

- Scene `scenes/cumulus.json`, camera `front`, 480×270 (about 29k cloud pixels; the article used ~10.7k).
- Sun angle $a\in\{0,15,\dots,180\}°$ at elevation 20°: $\omega_{sun}=\mathrm{normalize}(\cos e\,(-\cos a\,\hat f+\sin a\,\hat r)+\sin e\,\hat y)$
  with $\hat f,\hat r$ the camera's horizontal forward/right. 0° = sun behind the camera (front-lit), 180° = sun behind
  the cloud (backlit).
- Reference per angle: two-half `pt` at 4096 spp per half (`refs/a<angle>/`, estimated error recorded).
- Mask: pixels where the analytic render's alpha $1-T>0.05$.
- **Score** $=\sum_{mask}\mathrm{lum}(x)\,/\sum_{mask}\mathrm{lum}(r)$ (Rec.709 luminance, linear): 1 matches, <1 too
  dark, >1 too bright. **RMSE** of luminance over the mask. Full-image relMSE/PSNR/SSIM and GPU time (§4) as well.
- Methods: `analytic`, `neural_single` (scalar), `neural_single_shaderml` (Shader ML), `neural` (generalist). The
  same-cloud specialist is the article's setup; the generalist column shows what is lost without training on the cloud.
- Outputs: `sweep.csv`, `report.md`, per-angle image strips (reference, then each method), records, experiment copy.
- The `ml_encoder` variant is not in the sweep: its image equals the scalar variant's (relMSE 2.3e-15, enforced by
  `test_neural`), so its scores are identical, and its frame cost (~1.8 s at 480×270) would make a 13-angle protocol
  sweep take hours. Its timing is measured separately with `bench` at 240×135 (`results/mlencoder/`), next to the
  scalar and Shader ML variants under the same protocol.
