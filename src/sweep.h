// Sun sweep evaluation (the article's figure; docs/experiment-protocol.md): for each sun azimuth around the
// cloud, a path-traced reference and every method, scored over the cloud mask.
//
//   cloudnr sweep --experiment experiments/sweep.json [--out results/sweep_<ts>] [--quick]
//
// Writes DIR/sweep.csv, DIR/report.md, DIR/images/a<angle>.png (reference | method 1 | method 2 ...),
// DIR/refs/a<angle>/ (generateReference output, reused when it matches) and DIR/experiment.json.
#pragma once

#include "image_io.h"
#include "scene.h"

#include <cstdint>
#include <string>
#include <vector>

struct SweepExperiment {
    std::string name;
    std::string scene;                // scene JSON path
    std::string camera;               // camera name in the scene
    std::vector<double> angles_deg;   // sun azimuth relative to the camera: 0 = behind the camera, 180 = behind the cloud
    double elevation_deg = 20.0;
    uint32_t width = 480;             // film override
    uint32_t height = 270;
    std::vector<std::string> methods;  // RenderConfig JSON paths; one must be an analytic config (mask)
    std::string reference_config;      // pt config
    uint32_t reference_spp_per_half = 4096;
    uint32_t warmup = 30;
    uint32_t frames = 300;
    uint32_t runs = 3;
};

// Every key is required; unknown keys are an error. angles_deg: non-empty array of finite numbers;
// elevation_deg in (-90, 90); width, height, spp_per_half, frames, runs >= 1; warmup >= 0.
bool parseSweepExperiment(const std::string& path, SweepExperiment* experiment, std::string* error);

// Unit vector toward the sun: normalize(cos(e) * (-cos(a) * fwd_h + sin(a) * right_h) + sin(e) * Y), with
// fwd_h the camera forward projected onto the horizontal plane and right_h = normalize(cross(fwd_h, Y))
// (the SceneGPU camera-right convention). A vertical camera falls back to fwd_h = +Z.
simd_float3 sweepSunDirection(const CameraDesc& camera, double angleDeg, double elevationDeg);

// Rec.709 luminance of linear RGB.
double luminance709(const float* rgb);

// Luminance statistics of `test` against `reference` over the pixels where mask[i] != 0 (linear radiance,
// no exposure). score = sum(lum(test)) / sum(lum(reference)); rmse = sqrt(mean((lum(test) - lum(ref))^2)).
// Both NaN if the mask is empty, the sizes differ, or (score only) the reference sum is 0.
struct MaskedLuminance {
    double score = 0.0;
    double rmse = 0.0;
    uint64_t pixels = 0;
};
MaskedLuminance maskedLuminance(const Image& test, const Image& reference, const std::vector<uint8_t>& mask);

int cmdSweep(int argc, char** argv);
