# Metal Cloud Renderer: Architecture

Status: implemented (M1–M5). The benchmark and evaluation rules are in
[experiment-protocol.md](experiment-protocol.md). Build and run instructions are in §12.

## 1. Goal and scope

This is a research renderer for one volumetric cloud. Three methods render the same scene:

1. **Path-traced reference** (`pt`): unbiased apart from a documented bounce cap, with progressive accumulation.
2. **Analytic** (`analytic`): a deterministic real-time ray march with an approximate multiple-scattering model.
3. **Neural** (`neural`): the analytic march, with the multiple-scattering term replaced by a small MLP.

The deliverables are the CLI `cloudnr` and a minimal interactive viewer (`cloudnr view`). It is not an engine.
There is no render graph, ECS, plugin system, scripting, or editor.

Decisions already made:

- Neural method: learned in-scattering in the style of Deep Scattering (Kallweit et al. 2017), reduced so it runs in real time (§8).
- Density: procedural only, baked into a 3D texture. No VDB import.
- The CLI and an interactive viewer (AppKit + `CAMetalLayer`, a tone-mapping render pass, orbit camera) are both part of Milestone 1. Benchmarks stay headless.

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
  are computed in C++. It runs in a venv: `python3 -m venv .venv && .venv/bin/pip install torch numpy`, and the
  resolved versions are recorded. If torch does not install on Python 3.9, build the venv from Homebrew `python@3.12` instead.

## 3. Module layout

`cloudnr_core` is a static library of every `src/*.mm` except `main.mm`; `cloudnr` links it. Each module owns its
files; renderers are reached only through `makeRenderer`.

| Path | Responsibility |
|---|---|
| `src/main.mm` | CLI dispatch: `info`, `render`, `view`, `compare`, `reference`, `dump-training`, `neural-check`, `bench`, `suite` |
| `src/metal_context.{h,mm}` | Device, `MTL4CommandQueue`, `MTL4Compiler`, `MTL4CommandAllocator`s, `MTLResidencySet`, `MTLSharedEvent`, command feedback errors, pipeline creation from source |
| `src/scene.{h,mm}` | `Scene` and `RenderConfig` structs. JSON load/save uses `NSJSONSerialization` (native, so no JSON library). |
| `src/shared_types.h` | POD structs shared by C++ and MSL, each with an explicit `static_assert` on its size |
| `src/image_io.{h,mm}` | PFM read/write (linear float RGB), display mapping, PNG through ImageIO `CGImageDestination` |
| `src/metrics.{h,mm}` | relMSE, PSNR, SSIM and error heat maps (protocol §3) |
| `src/density_volume.{h,mm}` | Density texture ownership, bake dispatch and mip generation |
| `src/renderer.h`, `src/renderer_registry.mm` | `Renderer` interface; `makeRenderer` maps `config.renderer` to a factory |
| `src/density_renderer.{h,mm}` | `density` renderer: density visualization (M1) |
| `src/viewer.{h,mm}` | AppKit window, `CAMetalLayer`, display pass, orbit camera |
| `src/shaders/common.metal` | Camera ray, AABB clip, density sample, $\sigma_t$, HG eval and sample, PCG RNG, environment lookup, ratio/delta tracking, random walk, shared march. This is the only copy of the shared physics. |
| `src/shaders/bake_density.metal` | Procedural / constant density bake |
| `src/shaders/density_view.metal` | Density visualization kernel |
| `src/shaders/display.metal` | Viewer tone-mapping pass (`display_vs`, `display_fs`) |
| `src/analytic_renderer.{h,mm}`, `src/shaders/analytic.metal` | `analytic` renderer (§7) |
| `src/path_tracer.{h,mm}`, `src/shaders/path_tracer.metal` | `pt` renderer (§6) |
| `src/reference.{h,mm}` | Two-half references, convergence log (`reference`, protocol §2) |
| `src/neural_types.h`, `src/neural_model.{h,mm}`, `src/neural_renderer.{h,mm}`, `src/training_dump.{h,mm}` | `.cnrm` loader, `neural` renderer, `neural-check`, `dump-training` (§8) |
| `src/shaders/neural_common.metal`, `neural.metal`, `neural_dump.metal` | Descriptor builder and MLP (shared by inference and data generation), render and dataset kernels |
| `src/bench.{h,mm}`, `src/suite.{h,mm}` | Steady-state benchmark records, full comparison suite and reports (protocol §4–§7) |
| `experiments/*.json` | Fixed experiment definitions for `suite` |
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
| Camera | Pinhole defined by `position`, `target`, up = +Y, and vertical FOV `vfov_deg`. A scene holds a named `cameras` array (`side`, `backlit`). |
| Pixel filter | `pt`: box filter, uniform jitter inside the pixel. `analytic`/`neural`/`density`: one ray through the pixel center. This causes a small edge-aliasing difference, which counts toward their error. |
| Exposure | Not baked into the PFM. It is applied only for display and display-referred metrics: $\mathrm{display}=\mathrm{srgb\_oetf}\big(x\cdot2^{EV}/(1+x\cdot2^{EV})\big)$ per channel (Reinhard). The PNG writer, the viewer display pass and the C++ metrics apply the same mapping. |
| Volume | AABB `bounds_min`/`bounds_max` holding a baked `r16Float` 3D texture of size `grid`, with a full mip chain, trilinear sampling with `address::clamp_to_edge`, and an explicit AABB test that returns 0 outside. This keeps a homogeneous box exactly homogeneous up to its faces. |
| Density | $d\in[0,1]$, clamped at bake time |
| Coefficients | $\sigma_t(x)=\sigma_{t,max}\,d(x)$ [m⁻¹], $\sigma_s=\alpha\,\sigma_t$, where $\alpha$ is the single-scattering albedo |
| Phase | Henyey–Greenstein with parameter $g$: $p(\cos\theta)=\frac{1}{4\pi}\frac{1-g^2}{(1+g^2-2g\cos\theta)^{3/2}}$. $\cos\theta$ is measured between the propagation directions of the incident and scattered light. |
| Sun | Delta directional light. `sun_direction` points toward the sun; `sun_irradiance` is RGB $E_\perp$. The sun disc is not visible to camera rays. |
| Environment | $L_{env}(\omega)=$ `sky_radiance` if $\omega_y>0$, otherwise `ground_radiance`. The ground is a non-reflecting emitter with default 0. It exists so that a white-furnace test is possible. |

Density is `volume.density`: `"procedural"` (default) or `"constant"` (homogeneous box with `constant_density`,
default 1.0, used for closed-form validation). Procedural density:

$$d=\mathrm{saturate}\Big(\mathrm{remap}\big(\mathrm{fbm}_{perlin}(q),\,1-c,\,1\big)\cdot h(y_n)\,s(x_n)\,s(z_n)-e\cdot\mathrm{fbm}_{worley}(4q)\Big)$$

Here $c$ is `coverage`, $e$ is `erosion`, $(x_n,y_n,z_n)\in[0,1]^3$ is the position normalized to the AABB, and
$q=4\,(x-b_{min})/\max(\text{extent})$. $h(y)=\mathrm{smoothstep}(0,0.1,y)\,(1-\mathrm{smoothstep}(0.6,1,y))$ and
$s(u)=\mathrm{smoothstep}(0,0.15,u)\,(1-\mathrm{smoothstep}(0.85,1,u))$ fade the density to 0 at all six faces, so the
AABB never clips a cloud. Perlin fBm has 4 octaves and Worley fBm has 3 (lacunarity 2, gain 0.5). `density_seed` hashes
the noise offsets. These constants (documented in `src/shaders/bake_density.metal`) define scene `version` 1.

Scene JSON. The keys are load-bearing; unknown keys are an error:

```json
{"version":2,"name":"eval_medium",
 "cameras":[{"name":"side","position":[1390.109,200,-563.558],"target":[0,200,0],"vfov_deg":40,"exposure_ev":0}],
 "film":{"width":960,"height":540},
 "volume":{"bounds_min":[-400,0,-400],"bounds_max":[400,400,400],"grid":[256,128,256],
           "density":"procedural","density_seed":1,"coverage":0.5,"erosion":0.3,"constant_density":1.0},
 "medium":{"sigma_t_max":0.04,"albedo":0.99,"hg_g":0.8},
 "light":{"sun_direction":[0.3,0.6,0.74],"sun_irradiance":[3,3,3],
          "sky_radiance":[0.2,0.3,0.5],"ground_radiance":[0,0,0]}}
```

`RenderConfig` uses one JSON file per renderer variant in `configs/`:

```json
{"renderer":"pt","spp":4096,"seed":1,"max_bounces":0,"rr_start_bounce":16}
{"renderer":"analytic","primary_steps":128,"shadow_steps":16,"ambient_steps":4,"octaves":4,"ms_a":0.5,"ms_b":0.5,"ms_c":0.5,"ambient":true}
{"renderer":"neural","model":"models/<name>.cnrm","primary_steps":128,"shadow_steps":16}
{"renderer":"density","primary_steps":128}
```

In the `pt` config, `max_bounces=0` means unlimited. In the `analytic` config, `octaves=1` and `ambient=false`
reduce it to single scattering. The `density` renderer marches $\sigma_t$ only and shows $1-T$ over the environment.
Any change to a scene file requires a bump of its `version`.

## 5. Radiative transfer

Steady-state RTE without emission, in a grey medium, with no polarization or refraction:

$$(\omega\cdot\nabla)L(x,\omega)=-\sigma_t(x)L(x,\omega)+\sigma_s(x)\int_{S^2}p(\omega,\omega')L(x,\omega')\,d\omega'$$

Volume rendering form along a camera ray $x_t=x+t\omega$, clipped to the AABB:

$$L(x,\omega)=\int_0^{t_{max}}T(x,x_t)\,\sigma_s(x_t)\,L_s(x_t,\omega)\,dt+T(x,x_{t_{max}})\,L_{env}(\omega),\qquad T(a,b)=\exp\Big(-\int_a^b\sigma_t\Big)$$

All renderers split the in-scattered radiance $L_s$ the same way:

$$L_s=\underbrace{p(\omega_{sun}\to\omega)\,T_{sun}(x)\,E_{sun}}_{L_{ss}}+L_{rest}$$

$L_{rest}$ contains sky in-scattering and every multiply-scattered sun and sky path. In a grey medium the RTE is
linear in the sources, so

$$L_{rest}(x,\omega)=f_{sun}(x,\omega)\,E_{sun}+f_{sky}(x,\omega)\,L_{sky}+f_{gnd}(x,\omega)\,L_{gnd}$$

where the $f$ terms are scalars. The analytic and neural methods approximate only $L_{rest}$.

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

## 7. Analytic baseline

A deterministic ray march takes `primary_steps` uniform steps across the AABB intersection, sampling at the midpoint of each step.

- $L_{ss}$: a sun shadow march of `shadow_steps` uniform steps to the AABB exit gives $\tau_{sun}$.
- $L_{rest}$ uses Wrenninge 2013 octaves, where $i=0$ is $L_{ss}$, plus a hemispherical ambient term:

$$L_{rest}\approx\sum_{i=1}^{N-1}b^i\,E_{sun}\,p_{g c^i}(\cos\theta)\,e^{-a^i\tau_{sun}}+\tfrac12\big(L_{sky}e^{-\tau_{up}}+L_{gnd}e^{-\tau_{down}}\big)$$

  Here $N=$ `octaves` and $(a,b,c)=$ (`ms_a`,`ms_b`,`ms_c`). $\tau_{up}$ and $\tau_{down}$ come from `ambient_steps`-step marches straight up and straight down to the AABB exit.
- Each step is integrated in the energy-conserving form (Hillaire 2015): $\int_{\Delta t}T\sigma_sL_s\approx T\,\alpha\,L_s\,(1-e^{-\sigma_t\Delta t})$, followed by $T\leftarrow Te^{-\sigma_t\Delta t}$.
- The march never terminates early, because that would bias the fixed-cost comparison.

Approximations: the octave terms are not energy-conserving, ambient occlusion is estimated from only two
directions, and the march has discretization bias in both the primary and the shadow rays.

## 8. Neural renderer

The neural renderer uses the same march, $L_{ss}$ and integrator as the analytic renderer, and replaces only
$L_{rest}$ with an MLP. The two approximations therefore differ in exactly one term.

**Descriptor**: 40 floats, built by one function in `neural_common.metal` that both inference and data generation use.

- Local frame: $z$ points toward the sun. $x$ is the view direction projected perpendicular to $z$. If the two are parallel, fall back to world X.
- Levels $\ell=0..4$ with spacing $h_\ell=h_0 2^\ell$, where $h_0$ is the smallest voxel extent. Each level samples mip $\ell$ at 7 points (the center and $\pm x,\pm y,\pm z$). Each input is $\log(1+\sigma_{t,max}\,d\,h_\ell)$. This gives 35 values.
- The remaining inputs are $\cos\theta_{view,sun}$, world +Y in the local frame (3 values), and $T_{sun}$. Total: 40.

**MLP**: 40→32→32→32→3 with ReLU. From the outputs $y$, $f=\mathrm{expm1}(\max(y,0))$ gives $(f_{sun},f_{sky},f_{gnd})$.

- Inference is hand-written MSL. Weights and activations are half precision, and accumulation is float. The weights live in a buffer. The dimensions are injected as `#define` constants from the model header when the shader is compiled at runtime.
- `MTLTensor`, MetalPerformancePrimitives `matmul2d` and `MTL4MachineLearningCommandEncoder` are not used. A 3.4k-MAC MLP does not need them, and Apple7 support for shader tensor ops was not verified. Revisit them only if profiling shows that the MLP dominates frame time.

**Training data** (`cloudnr dump-training`, on the GPU):

- Points are chosen by rejection sampling where $d>0.01$, each with a uniformly random view direction.
- The descriptor comes from the same `common.metal` function used at inference.
- Each target is estimated from 1024 paths. Sample $\omega'\sim$ HG at $x$, then run the full path tracer from the next collision with unit sun, sky and ground sources and three separate accumulators. The direct-sun term at $x$ is excluded by construction, so the target is exactly $L_{rest}$.
- Records are raw little-endian float32 `[40 inputs][3 targets]` after a 64-byte header: magic `CNRD`, `u32` version, record count, seeds, and `hg_g`/`albedo`.

**Training** (`tools/train.py`): PyTorch on MPS if available, otherwise CPU. Adam, MSE on $\log1p(f)$, batch size
4096. Validation data comes from a disjoint seed range (protocol §1), and the checkpoint with the best
validation loss is exported.

**Model artifact** `models/<name>.cnrm`:

- Header: magic `CNRM`, `u32 format_version=1`, `u32 n_in, n_hidden, n_layers, n_out`, `f32 hg_g, albedo`, `u32 levels`.
- Body: fp16 weights in row-major order, each layer followed by its biases.
- Sidecar `<name>.json`: architecture, hyperparameters, dataset seeds and SHA-256, torch version, renderer version, and loss curves.
- The loader rejects a wrong magic, version or dimensions. It also rejects a scene whose `hg_g` or `albedo` differs from the model's, because the model is valid only for those two values. The $\sigma_t$ scale generalizes through the optical-thickness inputs.

Known bias: noise in the 1024-path targets biases a loss computed in log space ($E[\log1p(\hat f)]\ne\log1p(f)$).
It is reported with the results.

**Measured (M4, model `models/cloud_mlp.cnrm`)**: training set 409,600 records from 200 procedural scenes
(density seeds 1000–1199 of the 1000–1999 range, 2048 records each, 1024 paths per target, 12.7 s GPU), validation set
49,152 records from 24 scenes (seeds 2000–2023). 30 epochs on MPS; best validation loss 2.52e-4 (MSE on
$\log1p f$), unchanged with fp16 weights. GPU inference (`neural-check`, 10k validation descriptors) matches the
PyTorch evaluation of the exported fp16 weights within max relative error 3.1e-5. Against the fp32 PyTorch model,
fp16 weight quantization alone gives mean relative error 3.5e-3 and max 0.13 (rel = |a−b|/(|b|+1e-3), the max on
near-zero $f_{sun}$ outputs); a 1e-2 max-error parity against fp32 is therefore **not** met by fp16 weights.

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

Validation runs use `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`, which was verified to work with the Command Line Tools.
Benchmarks run without them.

Not used: render pipelines other than the viewer display pass, acceleration structures, MetalFX, CoreML, MPSGraph,
`MTLTensor`, and `MTL4MachineLearningCommandEncoder`.

## 10. Risks and dependencies

- Xcode license and offline toolchain missing: mitigated by runtime compilation with `MTL4Compiler`.
- GPU watchdog killing long dispatches: render 1 spp per dispatch, and split into tiles if one dispatch exceeds about 0.5 s.
- Reference convergence on the M1 may take hours for thick clouds, and the global majorant is inefficient for sparse density.
- The octave approximation is not energy-conserving, so the analytic error includes a model bias by design.
- Neural generalization and target noise; the model is restricted to fixed `hg_g`/`albedo`.
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
| **M4 Neural renderer** | Target $L_{rest}$, dataset generation (`dump-training`), training (`train.py`), weight conversion to `.cnrm`, Metal inference (§8) | Image-quality comparison vs the `pt` reference and GPU inference timing on unseen evaluation scenes. |
| **M5 Benchmarking** | Unified framework (`bench`, `suite`), identical scene/camera configs; GPU time, PSNR, SSIM, relMSE, memory | Reproducible fixed configs, automated metric collection, exportable JSON/CSV/Markdown reports (protocol §6). |

Test coverage required by AGENTS.md maps as follows. Camera transforms, density sampling, config serialization: M1.
Phase function, transmittance and extinction: M2/M3. Deterministic benchmark configs and image metrics: M1 (`compare`) and M5.

## 12. Build and run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure          # tests run under the Metal validation layer
./build/cloudnr info
./build/cloudnr view   --scene scenes/eval_medium.json --config configs/analytic.json    # keys 1-4 switch renderer
./build/cloudnr render --scene scenes/eval_medium.json --camera backlit --config configs/pt.json --png out/pt.png
./build/cloudnr reference --scene scenes/eval_thin.json --spp 1024 --out out/ref
./build/cloudnr dump-training --out data/train.cnrd --seed-begin 1000 --seed-end 2000 --scenes 200 --records-per-scene 2048
./build/cloudnr dump-training --out data/val.cnrd   --seed-begin 2000 --seed-end 3000 --scenes 24  --records-per-scene 2048
.venv/bin/python tools/train.py --train data/train.cnrd --val data/val.cnrd --out models/cloud_mlp
./build/cloudnr neural-check --model models/cloud_mlp.cnrm --check models/cloud_mlp.check.bin
./build/cloudnr suite --experiment experiments/eval.json      # add --quick for a smoke run
```

`render`, `view`, `compare`, `reference`, `dump-training` and `neural-check` accept `--validate` (re-executes with
`MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`); `bench` and `suite` are timing tools and warn if validation is on.
`view --autotest PREFIX` drives the camera controls through the input handlers and saves captures, for headless checks.
