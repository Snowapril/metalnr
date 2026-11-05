// Two-half path-traced reference with a convergence log (docs/experiment-protocol.md §2).
//
//   cloudnr reference --scene S [--camera C] [--config configs/pt.json] --spp N --out DIR [--validate]
//
// Renders halves A (seed config.seed) and B (seed config.seed + 1) at N spp each, logs
// relMSE(A_n, B_n) at power-of-two checkpoints to DIR/convergence.csv, and writes
// DIR/refA.pfm, refB.pfm, ref.pfm ((A + B) / 2), ref.png and ref.json (also printed to stdout).
#pragma once

#include "image_io.h"
#include "scene.h"

#include <cstdint>
#include <functional>
#include <string>

class MetalContext;
class DensityVolume;

// Contents of ref.json.
struct ReferenceResult {
    std::string scene;
    uint32_t scene_version = 0;
    std::string camera;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t spp_per_half = 0;
    uint32_t seed_a = 0;
    uint32_t seed_b = 0;
    double est_ref_relMSE = 0.0;  // relMSE(A, B) / 4
    bool meets_abs_gate = false;  // est_ref_relMSE <= 1e-4 (protocol §2); the relative gate is the suite's job
    double convergence_slope = 0.0;
    double bounce_cap_hits = 0.0;
    double mean_bounces = 0.0;
    double gpu_ms_total = 0.0;
    double wall_ms = 0.0;
    double compile_ms = 0.0;  // path tracer pipeline compile during this reference (both halves)
    double bake_ms = 0.0;
    uint32_t max_bounces = 0;
    uint32_t rr_start_bounce = 0;
    std::string renderer_version;
};

// One line of JSON (with trailing newline), the exact ref.json content.
std::string referenceResultJson(const ReferenceResult& result);
// Parses a ref.json written by generateReference. Non-numeric (null) errors load as NaN.
bool loadReferenceResult(const std::string& refJsonPath, ReferenceResult* result, std::string* error);

// Result of one progressive path-traced image.
struct ProgressiveResult {
    Image image;           // final spp mean
    double gpu_ms = 0.0;   // sum of per-frame GPU time
    double bounce_cap_hits = 0.0;
    double mean_bounces = 0.0;
};

// Renders `spp` samples of `ptConfig` (seed used as given) at 1 spp per frame through the context frame
// loop. At every power-of-two n and at n == spp the CPU waits for the GPU, reads the running mean and
// calls atCheckpoint(n, image, cumulative_gpu_ms). Not a timed loop.
bool renderProgressive(MetalContext& context, const Scene& scene, const RenderConfig& ptConfig,
                       const DensityVolume& volume, const SceneGPU& sceneGPU, uint32_t spp,
                       const std::function<void(uint32_t, const Image&, double)>& atCheckpoint,
                       ProgressiveResult* out, std::string* error);

// Generates the reference for one scene/camera into outDir (created if missing): halves A/B with seeds
// ptConfig.seed and ptConfig.seed + 1 at sppPerHalf each, convergence.csv, refA/refB/ref.pfm, ref.png,
// ref.json. `volume` must be baked from `scene`.
bool generateReference(MetalContext& context, const Scene& scene, const DensityVolume& volume,
                       const CameraDesc& camera, const RenderConfig& ptConfig, uint32_t sppPerHalf,
                       const std::string& outDir, ReferenceResult* result, std::string* error);

int cmdReference(int argc, char** argv);
