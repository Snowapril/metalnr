# Metal Cloud Renderer: Architecture

Status: implemented. The project replicates Charles Grassi's
[Neural Shaders, Explained With Clouds](https://charlesgrassi.dev/blog/neural-volumetric-clouds/) on Metal 4.
The benchmark and evaluation rules are in [experiment-protocol.md](experiment-protocol.md). Build and run
instructions are in §12.

## 1. Goal and scope

This is a research renderer for one volumetric cumulus cloud. Three methods render the same scene:

1. **Path-traced reference** (`pt`): unbiased apart from a documented bounce cap, with progressive accumulation.
2. **Analytic** (`analytic`): the article's "usual way": a ray march with Henyey–Greenstein single scattering, three
   Wrenninge multiple-scattering octaves and a two-direction ambient term.
3. **Neural** (`neural`): the same march, with the in-scattered radiance at each step predicted by a 13-input MLP.

The deliverables are the CLI `cloudnr` and a minimal interactive viewer (`cloudnr view`). It is not an engine.
There is no render graph, ECS, plugin system, scripting, or editor.

Decisions:

- Follow the article: the cumulus shape function, the march, the analytic constants, the 13 descriptor inputs, the
  13→16→16→3 squared-ReLU network trained on raw targets, and the sun-sweep evaluation.
- Run the network three ways to exercise Metal 4: hand-written MSL loops (the article's HLSL equivalent), Metal 4
  Shader ML (`matmul2d` from MetalPerformancePrimitives inside the march), and `MTL4MachineLearningCommandEncoder`
  (a `.mtlpackage` network fed by a wavefront march, §8–§9).
- Density: the procedural cumulus is baked into a 3D texture with mips. No VDB import.
- The CLI and an interactive viewer (AppKit + `CAMetalLayer`, a tone-mapping render pass, orbit camera) are both
  part of Milestone 1. Benchmarks stay headless.

## 2. Toolchain decisions

Verified environment (observed on the target machine before writing this document):

| Item | Value |
|---|---|
| Machine | Apple M1, 8-core GPU, 16 GB RAM |
| OS | macOS 27.2 (26B123) |
| GPU families | `MTLGPUFamilyMetal4`=true, `Apple7`=true, `Apple8`=false |
| `queryTimestampFrequency` | 24000000 Hz |
| `recommendedMaxWorkingSetSize` | ~12124 MB |
| `maxBufferLength` | ~9093 MB |
| `threadExecutionWidth` | 32 |
| Xcode | 27.2 at `/Applications/Xcode.app`. The license is not accepted, so `xcodebuild`, `xcrun metal` and Xcode `swift` fail. There is no offline `metal` compiler. |
| Command Line Tools | SDK `MacOSX27.2.sdk` with all `MTL4*.h`, `MTLTensor.h` and `MetalPerformancePrimitives.framework` |
| Host tools | Apple clang 21, cmake 4.4.5, system Python 3.9.6 (no torch). The directory is not a git repository. |

A probe run with `DEVELOPER_DIR=/Library/Developer/CommandLineTools swift -` on the M1 confirmed the following:
`MTL4Compiler newLibraryWithDescriptor:` compiled MSL from a source string, a compute pipeline sampling a
`texture3d<float>` was built through `MTL4ComputePipelineDescriptor`/`MTL4LibraryFunctionDescriptor`, and
`newMTL4CommandQueue` succeeded. With `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`, the device was
`MTLDebugDevice`, the queue was `MTL4DebugCommandQueue`, and the log contained "Metal API Validation Enabled".

Decisions:

- **Build**: CMake with the Command Line Tools `clang++`. Host code is C++20, plus Objective-C++ (`.mm`) wherever it calls Metal.
- **Shaders are compiled at runtime from source** with `MTL4Compiler`. The host concatenates
  `src/shared_types.h`, `src/shaders/common.metal` and the kernel file into one source string
  (`MetalContext::buildShaderSource`; shader files never `#include` project files).
  This needs neither the Xcode license nor the offline toolchain, it was verified on the target,
  and it keeps host and shader structs in a single header.
- Compile time is measured and reported separately from rendering (protocol §4).
- Rejected alternative: an offline `metallib` built by Xcode. It is blocked by the license and the missing
  toolchain. Revisit it only if runtime compile time becomes a measured problem.
- **Python** is used only for training (`tools/train.py`), never by the renderer or for evaluation; image metrics
  are computed in C++. It runs in a venv from `tools/requirements.txt` (torch 2.8.0, numpy 2.0.2, system Python 3.9.6).

## 3. Module layout

`cloudnr_core` is a static library of every `src/*.mm` except `main.mm`; `cloudnr` links it. Each module owns its
files; renderers are reached only through `makeRenderer`.

| Path | Responsibility |
|---|---|
| `src/main.mm` | CLI dispatch: `info`, `render`, `view`, `compare`, `reference`, `dump-training`, `neural-check`, `bench`, `suite`, `sweep` |
| `src/metal_context.{h,mm}` | Device, `MTL4CommandQueue`, `MTL4Compiler`, `MTL4CommandAllocator`s, `MTLResidencySet`, `MTLSharedEvent`, command feedback errors, pipeline creation from source |
| `src/scene.{h,mm}` | `Scene` and `RenderConfig` structs. JSON load/save uses `NSJSONSerialization` (native, so no JSON library). |
| `src/shared_types.h` | POD structs shared by C++ and MSL, each with an explicit `static_assert` on its size |
| `src/image_io.{h,mm}` | PFM read/write (linear float RGB), display mapping, PNG through ImageIO `CGImageDestination` |
| `src/metrics.{h,mm}` | relMSE, PSNR, SSIM and error heat maps (protocol §3) |
| `src/density_volume.{h,mm}` | Density texture ownership, bake dispatch and mip generation |
| `src/renderer.h`, `src/renderer_registry.mm` | `Renderer` interface; `makeRenderer` maps `config.renderer` to a factory |
| `src/density_renderer.{h,mm}` | `density` renderer: density visualization (M1) |
| `src/viewer.{h,mm}` | AppKit window, `CAMetalLayer`, display pass, orbit camera |
| `src/shaders/common.metal` | Camera ray, AABB clip, density sample, $\sigma_t$, HG eval and sample, PCG RNG, environment lookup, ratio/delta tracking, random walk, and the shared march `march_view_ray`. This is the only copy of the shared physics. |
| `src/shaders/bake_density.metal` | Cumulus / constant density bake |
| `src/shaders/density_view.metal` | Density visualization kernel |
| `src/shaders/display.metal` | Viewer tone-mapping pass (`display_vs`, `display_fs`) |
| `src/analytic_renderer.{h,mm}`, `src/shaders/analytic.metal` | `analytic` renderer (§7) |
| `src/path_tracer.{h,mm}`, `src/shaders/path_tracer.metal` | `pt` renderer (§6) |
| `src/reference.{h,mm}` | Two-half references, convergence log (`reference`, protocol §2) |
| `src/neural_types.h`, `src/neural_model.{h,mm}`, `src/neural_renderer.{h,mm}`, `src/training_dump.{h,mm}` | `.cnrm` loader, `neural` renderer, `neural-check`, `dump-training` (§8) |
| `src/shaders/neural_common.metal`, `neural.metal`, `neural_dump.metal` | Descriptor builder and MLP (shared by inference and data generation), render and dataset kernels |
| `src/bench.{h,mm}`, `src/suite.{h,mm}` | Steady-state benchmark records, full comparison suite and reports (protocol §4–§7) |
| `src/sweep.{h,mm}` | The article's sun sweep (`sweep`, protocol §8) |
| `experiments/*.json` | Fixed experiment definitions for `suite` and `sweep` |
| `tests/test_*.mm`, `tests/check.h` | One executable per test file, registered with CTest. `check.h` provides `CHECK`, `CHECK_NEAR`, `checkSummary()`. No test framework. |
| `tools/train.py`, `tools/requirements.txt` | Neural training and model export (M4); pinned torch/numpy |
| `scenes/*.json`, `configs/*.json`, `models/` | Scene definitions, renderer configs, model artifacts |

## 4. Shared scene contract

Every renderer reads the same `Scene`. The physics is implemented once, in `common.metal`.

| Quantity | Definition |
|---|---|
| World | Right-handed, +Y up, meters |
| Radiance | Relative linear units |
| Color | Linear Rec.709/sRGB primaries, 3-channel RGB. The medium is grey ($\sigma$ does not depend on wavelength); only light colors are RGB. |
| Camera | Pinhole defined by `position`, `target`, up = +Y, and vertical FOV `vfov_deg`. A scene holds a named `cameras` array (`front`, `side`, `below`). |
| Pixel filter | `pt`: box filter, uniform jitter inside the pixel. `analytic`/`neural`/`density`: one ray through the pixel center, with a per-pixel dithered march start. |
| Exposure | Not baked into the PFM. It is applied only for display and display-referred metrics: $\mathrm{display}=\mathrm{srgb\_oetf}\big(x\cdot2^{EV}/(1+x\cdot2^{EV})\big)$ per channel (Reinhard). The PNG writer, the viewer display pass and the C++ metrics apply the same mapping. |
| Volume | AABB `bounds_min`/`bounds_max` holding a baked `r16Float` 3D texture of size `grid`, with a full mip chain, trilinear sampling with `address::clamp_to_edge`, and an explicit AABB test that returns 0 outside. This keeps a homogeneous box exactly homogeneous up to its faces. |
| Density | $d\in[0,1]$, clamped at bake time |
| Coefficients | $\sigma_t(x)=\sigma_{t,max}\,d(x)$ [m⁻¹], $\sigma_s=\alpha\,\sigma_t$, where $\alpha$ is the single-scattering albedo |
| Phase | Henyey–Greenstein with parameter $g$: $p(\cos\theta)=\frac{1}{4\pi}\frac{1-g^2}{(1+g^2-2g\cos\theta)^{3/2}}$. $\cos\theta$ is measured between the propagation directions of the incident and scattered light. |
| Sun | Delta directional light. `sun_direction` points toward the sun; `sun_irradiance` is RGB $E_\perp$. The sun disc is not visible to camera rays. |
| Environment | $L_{env}(\omega)=$ `sky_radiance` if $\omega_y>0$, otherwise `ground_radiance`. The ground is a non-reflecting emitter with default 0. It exists so that a white-furnace test is possible. |

Density is `volume.density`: `"cumulus"` or `"constant"` (homogeneous box with `constant_density`, default 1.0, used
for closed-form validation). The cumulus is the article's `Sample()` function, evaluated in cloud-local metres
$p = x - \text{base\_center}$ and baked at voxel centres:

```
p.xz -= shear * p.y                                              // wind shear
p.xz /= 1 + anvil_spread / radius * smoothstep(anvil_height, height, p.y)   // anvil warp
d = max over lobes (smooth, width lobe_blend) of  r_i - |p - c_i|           // welded spheres
d = smooth_min(d, p.y, base_softness)                            // flat base
d += fbm(p / erosion_feature_size, erosion_octaves, seed) * erosion_amplitude * smoothstep(0, base_fade_height, p.y)
density = saturate(d / edge)
```

Smooth min/max are Quilez's polynomial forms; fBm is signed gradient noise in $[-1,1]$ with lacunarity 2 and gain
0.5, seeded by integer hashing. `makeRandomCumulus(seed)` generates training and held-out shapes (4–12 lobes, shear,
anvil and erosion drawn from fixed ranges, always inside the volume). In JSON, `"cloud":{"generator_seed":N}` expands
to that shape at load time.

Scene JSON. The keys are load-bearing; unknown keys are an error; `version` must be ≥ 3:

```json
{"version":3,"name":"cumulus",
 "cameras":[{"name":"front","position":[0,120,-2300],"target":[0,300,0],"vfov_deg":22,"exposure_ev":0.5}],
 "film":{"width":960,"height":540},
 "volume":{"bounds_min":[-500,0,-500],"bounds_max":[500,800,500],"grid":[256,192,256],"density":"cumulus",
   "cloud":{"base_center":[0,90,0],"lobes":[[0,140,0,190],[-230,110,30,150]],"lobe_blend":35,"shear":[0.08,0.02],
            "anvil_height":500,"height":640,"anvil_spread":60,"radius":300,"base_softness":10,"base_fade_height":50,
            "erosion_feature_size":45,"erosion_octaves":5,"erosion_amplitude":40,"edge":12,"seed":1}},
 "medium":{"sigma_t_max":0.05,"albedo":0.99,"hg_g":0.8},
 "light":{"sun_direction":[-0.6124,0.5,-0.6124],"sun_irradiance":[12,11.6,10.8],
          "sky_radiance":[0.18,0.3,0.62],"ground_radiance":[0.08,0.07,0.06]}}
```

`RenderConfig` uses one JSON file per renderer variant in `configs/`. The march keys (`step_scale`, `max_steps`,
`min_transmittance`, `dither`, `empty_skip`) are optional for `density`, `analytic` and `neural`:

```json
{"renderer":"pt","spp":4096,"seed":1,"max_bounces":0,"rr_start_bounce":16}
{"renderer":"analytic","step_scale":1,"max_steps":512,"min_transmittance":0.01,"dither":true,"empty_skip":true,
 "shadow_steps":6,"shadow_distance":0,"ambient_steps":4,"octaves":3,"ms_a":0.4,"ms_b":0.35,"ms_c":0.1,"ambient":true}
{"renderer":"neural","model":"models/cumulus_single.cnrm","inference":"scalar"}
{"renderer":"density"}
```

In the `pt` config, `max_bounces=0` means unlimited. `configs/analytic_ss.json` (`octaves` 1, no ambient, no dither,
no early termination, fine steps) is exact single scattering for validation. The `density` renderer marches
$\sigma_t$ only and shows $1-T$ over the environment. Any change to a scene file requires a bump of its `version`.

## 5. Radiative transfer

Steady-state RTE without emission, in a grey medium, with no polarization or refraction:

$$(\omega\cdot\nabla)L(x,\omega)=-\sigma_t(x)L(x,\omega)+\sigma_s(x)\int_{S^2}p(\omega,\omega')L(x,\omega')\,d\omega'$$

Volume rendering form along a camera ray $x_t=x+t\omega$, clipped to the AABB:

$$L(x,\omega)=\int_0^{t_{max}}T(x,x_t)\,\sigma_s(x_t)\,L_s(x_t,\omega)\,dt+T(x,x_{t_{max}})\,L_{env}(\omega),\qquad T(a,b)=\exp\Big(-\int_a^b\sigma_t\Big)$$

The shared march evaluates the in-scattered radiance $L_s(x,\omega)$ at each step. In a grey medium the RTE is linear
in the sources, so

$$L_s(x,\omega)=f_{sun}(x,\omega)\,E_{sun}+f_{sky}(x,\omega)\,L_{sky}+f_{gnd}(x,\omega)\,L_{gnd}$$

where the $f$ terms are scalars. The analytic method approximates $L_s$ with closed-form terms; the neural method
predicts $(f_{sun},f_{sky},f_{gnd})$. This is the article's "one line the network replaces".

Assumptions: there is no medium outside the AABB, no atmosphere or aerosol, no ground reflection, and the
sun is a delta light. HG is used instead of Mie, so fogbow and glory are not modeled.

## 6. Reference path tracer

Each dispatch uses one thread per pixel and computes one sample. Accumulation is progressive.

- **RNG**: PCG32 seeded from `hash(seed, pixel_index, sample_index)`. Results do not depend on scheduling and are bit-reproducible on the same GPU and OS.
- **Free flight**: clip to the AABB, then delta tracking with the global majorant $\bar\sigma=\sigma_{t,max}$. This is valid because $d\le1$ and trilinear interpolation never exceeds the voxel maximum. A majorant grid is deferred until measurements show it is needed.
- **At a real collision**:
  - Multiply the throughput by $\alpha$.
  - Do next-event estimation to the sun, with $T_{sun}$ from ratio tracking.
  - Sample a new direction from HG.
- **On escape**: add throughput $\cdot L_{env}(\omega)$. The sun is never added on escape, so nothing is counted twice and no MIS is needed.
- **Russian roulette**: starts at bounce `rr_start_bounce` with a fixed survival probability of 0.95; the throughput is divided by 0.95.
- **Bias**: a safety cap of 4096 bounces is the only source of bias. The renderer counts and reports the paths that hit it.
- **Accumulation**: a float32 running mean $m\leftarrow m+(x-m)/n$, to limit precision loss at high spp.
- `max_bounces` is used only for validation, for example `max_bounces=1` isolates single scattering.

## 7. Shared march and analytic baseline

`march_view_ray` (common.metal) is the article's loop, shared by `density`, `analytic` and `neural`:

- Step length $\Delta = $ `voxel_min_extent` × `step_scale`. The first segment is $\Delta$ times a per-pixel
  interleaved-gradient-noise value in $(0,1]$ (`dither`), so pixels do not sample at the same depths. Each segment
  is sampled at its midpoint with its own length.
- `empty_skip`: if mip 3 reads zero at a segment start, the surrounding 8-voxel region is empty and the march skips
  whole segments (the samples stay on the same grid, so the image is unchanged; test-verified).
- Each step adds $T\,\alpha\,L_s\,(1-e^{-\sigma_t\Delta})$ (Hillaire 2015) and multiplies $T$ by $e^{-\sigma_t\Delta}$.
- The march stops when $T<$ `min_transmittance` (0.01; bias at most 1% of the remaining radiance) or after
  `max_steps`. It returns $(L+T\,L_{env},\ 1-T)$; the alpha channel is the cloud mask used by the sun sweep.

The analytic shade function, with the article's constants $(a,b,c)=(0.4,0.35,0.1)$ and $N=3$ octaves:

$$L_s\approx\sum_{i=0}^{N-1}b^i\,E_{sun}\,p_{g c^i}(\cos\theta)\,e^{-a^i\tau_{sun}}+\tfrac12\big(L_{sky}e^{-\tau_{up}}+L_{gnd}e^{-\tau_{down}}\big)$$

$\tau_{sun}$ comes from `shadow_steps` (6) midpoint samples toward the sun, $\tau_{up}$ and $\tau_{down}$ from
`ambient_steps` (4) samples straight up and down: 14 sequential texture reads per step, as in the article.

Approximations: the octave terms are not energy-conserving, ambient occlusion is estimated from two directions, and
the march has discretization bias. The sun sweep (protocol §8) measures how the resulting error moves with the sun.

## 8. Neural renderer

The neural renderer uses the same march and integrator as the analytic renderer and replaces only the shade
function, so the two differ in exactly one term.

**Descriptor** (13 floats, `build_descriptor` in `neural_common.metal`, shared by inference and data generation):

- Radii fixed in metres: $r\in\{6,12,24,48,96,192\}$. Each radius reads the density mip at
  $\mathrm{lod}=\log_2(r/\text{voxel})$, so every input is one texture fetch.
- Inputs 0–5: $\sigma_t$ at $x$ for each radius. Inputs 6–11: $\sigma_t$ at $x+r\,\omega_{sun}$. Input 12:
  $\cos(\text{view},\text{sun})$ = `dot(d, sun_direction)`.
- Metric radii are what lets one network serve any bake resolution and cloud size.

**Network**: standardize $(x-\mu)\cdot s$, then 13→16→16→3 with squared-ReLU hidden activations and a squared
output (non-negative by construction), clamped at 1.5× the largest training target. 512 multiply-adds per step.
Outputs are $(f_{sun},f_{sky},f_{gnd})$; the shader combines them with the scene's lights.

**Inference variants** (`RenderConfig.inference`):

- `scalar`: hand-written MSL loops over fp32 weights in a device buffer (the article's HLSL).
- `shader_ml`: Metal 4 Shader ML. `<metal_tensor>` + MetalPerformancePrimitives `matmul2d` with
  `execution_thread`, `tensor_inline` thread tensors for activations and device tensors for weights, inside the march.
  Found constraints on this M1: K must be a multiple of 16 (13 inputs are zero-padded to 16), single-thread N must be
  1, 2, 4 or a multiple of 8 (3 outputs are padded to 4), the destination accumulates (zeroed first), and weight
  tensors are `[K][N]` with N contiguous. Weights are fp16, activations and accumulation fp32
  (`float × half → float`). fp16 activations were tried first and gave 1.08e-2 max relative error; fp32 activations
  give 3.6e-3.
- `ml_encoder`: `MTL4MachineLearningCommandEncoder` running the network from `models/<name>.mtlpackage` (§9). The
  march is split into a wavefront: per chunk of 16 shaded samples per pixel, a compute pass resumes every pixel's march
  (`march_advance`, the same code and sample order as `march_view_ray`) and writes raw descriptors and step weights;
  one ML dispatch evaluates all rows; a compute pass adds $\sum_j w_j\,(f_j\cdot\text{lights})$. Chunks per frame
  $=\lceil$`max_steps`$/16\rceil$, so no pixel is ever truncated and the image equals the scalar path.

**Training data** (`cloudnr dump-training`, on the GPU):

- Camera-style sampling (the article's fix): a random ray from the bounding sphere toward a random point in the
  volume is delta-tracked to its first real collision $x$; $\omega$ is the ray direction. This samples the visible
  surface rather than the unseen interior.
- Target: the mean over `paths` (256) of the sun term $p(\cos\theta)\,T_{sun}$ (ratio tracking) plus a full random
  walk from $x$ along an HG-sampled direction, with unit sun, sky and ground accumulated separately.
- Random-shape mode: shape seeds $[1000,2000)$ train and $[2000,3000)$ validate; per shape a random sun (elevation
  −10…90°, any azimuth) and $\sigma_{t,max}$ log-uniform in $[0.02,0.1]$. `--fixed-shape` keeps the template cloud
  and $\sigma_t$ and varies only the sun (the article's single-cloud weights).
- Records: little-endian float32 `[13 inputs][3 targets]` after a 64-byte `CNRD` v2 header.

**Training** (`tools/train.py`): PyTorch on MPS, Adam (3e-3, cosine decay), batch 4096, 60 epochs, MSE on raw
targets. The article shows that fitting a transformed target (cube root, log) biases the result, because Monte Carlo
noise does not average through a curved transform. The best-validation checkpoint is exported.

**Model artifact** `models/<name>.cnrm` v2: header (magic, version, dimensions, `hg_g`, `albedo`, radii,
standardization, clamp) and fp32 weights; a sidecar JSON records the architecture, hyperparameters, datasets with
SHA-256, seed ranges, library versions, loss curves, validation Pearson correlation per output and the Shader ML
fp16 error; a `.check.bin` holds 10,000 validation inputs with PyTorch outputs for `neural-check`. The loader rejects
a wrong format and any scene whose `hg_g`/`albedo` differ from the model's.

**Measured models**:

| Model | Data | Val. Pearson sun / sky / ground | GPU vs PyTorch (scalar / shader_ml max rel.) |
|---|---|---|---|
| `cumulus_single` | `scenes/cumulus.json`, 300×4096 records with random suns (`--fixed-shape`) | 0.89 / 0.61 / 0.60 | 5.2e-6 / 5.7e-3 |
| `cumulus_net` | 300 random shapes (1.23 M records), validated on 32 held-out shapes | 0.93 / 0.51 / 0.48 | 5.3e-6 / 3.6e-3 |

Like the article, sky/ground correlation is limited by the 256-path coin-flip estimate of sky visibility, and the
single-cloud weights beat the generalist on their own cloud.

## 9. Metal 4 usage

Every API below was confirmed in the SDK 27.2 headers
(`MacOSX27.2.sdk/System/Library/Frameworks/Metal.framework/Headers`).

| API | Purpose |
|---|---|
| `MTLDevice newMTL4CommandQueue` | The single queue |
| `MTL4CommandAllocator`, `MTL4CommandBuffer beginCommandBufferWithAllocator:` / `endCommandBuffer` | One allocator per frame in flight |
| `MTL4CommandQueue commit:count:` / `commit:count:options:` | Submission |
| `MTL4Compiler`, `MTL4LibraryDescriptor.source`, `newLibraryWithDescriptor:`, `MTL4LibraryFunctionDescriptor`, `MTL4ComputePipelineDescriptor`, `newComputePipelineStateWithDescriptor:` | Runtime shader compilation |
| `MTL4ArgumentTable setAddress:atIndex:` / `setTexture:atIndex:` (takes `texture.gpuResourceID`) | Binding buffers (by GPU address) and textures |
| `MTLResidencySet`, `MTL4CommandQueue addResidencySet:` | Residency for all renderer resources |
| `MTL4ComputeCommandEncoder dispatchThreads:threadsPerThreadgroup:` | All kernels |
| `MTL4ComputeCommandEncoder generateMipmapsForTexture:` | Density mip chain after the bake |
| `MTL4ComputeCommandEncoder copyFromTexture:…toBuffer:` | Readback of the result for PFM output, outside timed frames |
| `barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:` (declared on `MTL4CommandEncoder`) | Ordering between dependent dispatches |
| `MTLSharedEvent`, `MTL4CommandQueue signalEvent:value:` / `waitForEvent:value:` | At most 2 frames in flight; the CPU waits only for allocator reuse |
| `MTL4CommitOptions addFeedbackHandler:` → `MTL4CommitFeedback.GPUStartTime` / `GPUEndTime` | GPU time per commit (the benchmark metric) |
| `MTL4CounterHeap` (`newCounterHeapWithDescriptor:`) + `MTLDevice queryTimestampFrequency` | Per-pass timestamps, only if needed |
| `CAMetalLayer`, `MTL4CommandQueue waitForDrawable:` / `signalDrawable:` | Viewer drawables |
| `MTL4RenderPipelineDescriptor`, `MTL4Compiler newRenderPipelineStateWithDescriptor:compilerTaskOptions:error:` | Viewer display pipeline (`display_vs`/`display_fs`, `BGRA8Unorm`) |
| `MTL4CommandBuffer renderCommandEncoderWithDescriptor:`, `MTL4RenderCommandEncoder setArgumentTable:atStages:`, `drawPrimitives:vertexStart:vertexCount:` | Viewer tone-mapping pass with a fragment-stage argument table |
| MSL `<metal_tensor>` `tensor<…, tensor_inline>` + `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` `mpp::tensor_ops::matmul2d<desc, execution_thread>` | Shader ML inference of the MLP inside the march (`inference: shader_ml`); compiles through the runtime `MTL4Compiler` path, verified on the M1 |

Validation runs use `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`, which was verified to work with the Command Line Tools.
Benchmarks run without them.

**`MTL4MachineLearningCommandEncoder`.** The network must arrive as an `.mtlpackage` loaded with
`newLibraryWithURL:`; `tools/export_mtlpackage.py` produces it: `.cnrm` → PyTorch (parity-checked against the
`.check.bin`) → Core ML mlprogram (fp32, input `x` [N,13], output `f` [N,3], flexible N) → Apple's offline
`metal-package-builder` (Xcode Metal toolchain; `sudo xcodebuild -license accept` and
`xcodebuild -downloadComponent MetalToolchain`). Runtime APIs: `MTL4MachineLearningPipelineDescriptor` with
`setInputDimensions:atBufferIndex:` for the batch and `MTL4PipelineOptions.shaderReflection` (otherwise the reflection
is empty), `newMachineLearningPipelineStateWithDescriptor:`, tensors from shared buffers
(`newTensorWithDescriptor:offset:`, `MTLTensorUsageMachineLearning`) bound with `MTL4ArgumentTable setResource:atBufferIndex:`,
`machineLearningCommandEncoder`, `dispatchNetworkWithIntermediatesHeap:` with a placement heap, and
`MTLStageMachineLearning` barriers between the compute and ML passes.

Toolchain and runtime findings (macOS 27.2, Metal toolchain 27.2.47):

- `metal-package-builder` runs from a read-only cryptex mount and looks for `coremlcompiler` in a sibling
  `XcodeDefault.xctoolchain` that does not exist there. The export script runs the unmodified Apple binary from a
  temporary directory that recreates that layout (symlinks to the real `usr/lib`, `System` and Xcode's `coremlcompiler`).
- The builder has no rewrite patterns for the macOS 26 (`ios19`) Core ML opset's binary elementwise ops (`sub`, `mul`,
  `minimum` abort with "Missing pattern to rewrite MIL op"); exporting with the macOS 15 (`ios18`) opset lowers fully.
- ML tensor row strides must be multiples of 64 bytes, so descriptor and output rows are padded to 16 floats.
- Releasing an intermediates heap after it was dispatched makes the GPU validation layer crash in a later commit, so
  `MetalContext::machineLearningHeap` owns one shared heap for its lifetime.
- The encoder dispatches a whole network over a fixed-size batch, so a per-step network inside a march costs
  `max_steps` rows per pixel per frame, including finished pixels; measured cost is in protocol §8 and the README.

Not used: render pipelines other than the viewer display pass, acceleration structures, MetalFX, CoreML at runtime,
MPSGraph.

## 10. Risks and dependencies

- Xcode license and offline toolchain missing: mitigated by runtime compilation with `MTL4Compiler`.
- GPU watchdog killing long dispatches: render 1 spp per dispatch, and split into tiles if one dispatch exceeds about 0.5 s.
- Reference convergence on the M1 may take hours for thick clouds, and the global majorant is inefficient for sparse density.
- The octave approximation is not energy-conserving, so the analytic error includes a model bias by design.
- Neural generalization and target noise; each model is restricted to its trained `hg_g`/`albedo`, and the
  single-cloud model is only meaningful on `scenes/cumulus.json`.
- Torch availability on Python 3.9 (fallback: Homebrew `python@3.12`).
- M1 DVFS and thermal variance in timings: mitigated by warm-up, repeated runs and median statistics (protocol §4).
- Shader validation overhead: validation is kept out of benchmarks.
- Density precision of `r16Float` (11-bit mantissa): relative quantization of about $5\times10^{-4}$.

## 11. Milestones

Each milestone must pass its acceptance check with a clean validation layer (`MTL_DEBUG_LAYER=1
MTL_SHADER_VALIDATION=1`, no command feedback errors) before dependent work is accepted.
Dependencies: M1 → {M2, M3, M4} → M5.

| Milestone | Content | Acceptance |
|---|---|---|
| **M1 Metal 4 Foundation** | Window, device init, basic rendering pass, camera controls, procedural density, GPU error validation (`info`, `render`, `view`, `compare`, `density` renderer) | The application visualizes the GPU density field in a window with interactive orbit camera. Tests: scene/config JSON round-trip; PFM round-trip bit-exact; center-pixel ray equals normalize(target−position); bake bit-identical for the same seed and different for another; values in [0,1]; sampling outside the AABB returns 0. |
| **M2 Analytic baseline** | Beer–Lambert transmittance, ray marching, HG phase, single scattering (§7); octave multiple scattering and the ambient term are optional settings | Matches closed forms for homogeneous media: with $\alpha=0$, pixel $=L_{env}e^{-\sigma_t\ell}$; with the sun perpendicular to a horizontal view ray, $L=\alpha\,E\,p(0)\,e^{-\sigma_t h}(1-e^{-\sigma_t\ell})+T\,L_{env}$ ($h$ the sun-path length inside the box, $\ell$ the view-path length). |
| **M3 Path-traced reference** | Free-flight sampling, multiple scattering, NEE, progressive accumulation (§6) | Convergence analysis (relMSE of independent halves vs spp falls ~1/spp) and reference images. Tests: white furnace ($\alpha=1$, sky = ground = 1) within 3σ; transmittance $L_{env}e^{-\sigma_t\ell}$ within 3σ; mean HG $\cos\theta=g$ within 3σ for $g\in\{-0.5,0,0.8\}$; same seed bit-identical; `max_bounces=1` matches analytic single scattering. |
| **M4 Neural renderer** | Target $L_s$, dataset generation (`dump-training`), training (`train.py`), `.cnrm` export, Metal inference (§8) | Image-quality comparison vs the `pt` reference and GPU inference timing on unseen evaluation scenes. |
| **M5 Benchmarking** | Unified framework (`bench`, `suite`), identical scene/camera configs; GPU time, PSNR, SSIM, relMSE, memory | Reproducible fixed configs, automated metric collection, exportable JSON/CSV/Markdown reports (protocol §6). |
| **R Article replication** | The article's cumulus, march, analytic constants, 13-input network, camera-style data, raw-target training, sun sweep (`sweep`); Shader ML inference | Closed-form march tests; scalar GPU inference within 1e-5 and Shader ML within 1e-2 of PyTorch; sun sweep and held-out suite reports (protocol §8). |

Test coverage required by AGENTS.md maps as follows. Camera transforms, density sampling, config serialization: M1.
Phase function, transmittance and extinction: M2/M3. Deterministic benchmark configs and image metrics: M1 (`compare`) and M5.

## 12. Build and run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure          # tests run under the Metal validation layer
./build/cloudnr info
./build/cloudnr view   --scene scenes/cumulus.json --config configs/analytic.json    # keys 1-4 switch renderer
./build/cloudnr render --scene scenes/cumulus.json --camera front --config configs/neural_single_shaderml.json --png out/nn.png
./build/cloudnr reference --scene scenes/cumulus.json --camera front --spp 1024 --out out/ref
# Single-cloud model (the article's weights): one cloud, random suns
./build/cloudnr dump-training --fixed-shape --template scenes/cumulus.json --out data/single_train.cnrd \
    --shape-begin 1 --shape-end 2 --shapes 300 --records-per-shape 4096
./build/cloudnr dump-training --fixed-shape --template scenes/cumulus.json --out data/single_val.cnrd \
    --shape-begin 7 --shape-end 8 --shapes 32 --records-per-shape 4096
.venv/bin/python tools/train.py --train data/single_train.cnrd --val data/single_val.cnrd --out models/cumulus_single
# Generalist model: random shapes, held-out validation shapes
./build/cloudnr dump-training --out data/train.cnrd --shape-begin 1000 --shape-end 2000 --shapes 300 --records-per-shape 4096
./build/cloudnr dump-training --out data/val.cnrd   --shape-begin 2000 --shape-end 3000 --shapes 32  --records-per-shape 4096
.venv/bin/python tools/train.py --train data/train.cnrd --val data/val.cnrd --out models/cumulus_net
.venv/bin/python tools/export_mtlpackage.py --model models/cumulus_net.cnrm   # ML encoder package (Xcode Metal toolchain)
./build/cloudnr neural-check --model models/cumulus_net.cnrm --check models/cumulus_net.check.bin
./build/cloudnr sweep --experiment experiments/sweep.json      # the article's sun sweep; --quick for a smoke run
./build/cloudnr suite --experiment experiments/eval.json      # add --quick for a smoke run
```

`render`, `view`, `compare`, `reference`, `dump-training` and `neural-check` accept `--validate` (re-executes with
`MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`); `bench` and `suite` are timing tools and warn if validation is on.
`view --autotest PREFIX` drives the camera controls through the input handlers and saves captures, for headless checks.
