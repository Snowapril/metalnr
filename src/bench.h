// Steady-state benchmark of one renderer on one scene/camera (docs/experiment-protocol.md §4-§6).
//
//   cloudnr bench --scene S [--camera C] --config C [--reference ref.pfm [--reference-id ID]]
//                 [--warmup N] [--frames N] [--runs N] [--out record.json] [--image out.pfm]
#pragma once

#import <Foundation/Foundation.h>

#include "image_io.h"
#include "scene.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

class MetalContext;
class DensityVolume;

struct TimingStats {
    double median = NAN;
    double p5 = NAN;
    double p95 = NAN;
};

// Nearest-rank percentiles (rank = ceil(p/100 * n), 1-based) of the finite values; NaN if none.
// The median is the 50th nearest-rank percentile (the lower median for even n).
TimingStats summarizeTimes(std::vector<double> values);

// Per-run statistics plus the headline: the median of the run medians (and likewise of p5/p95).
struct TimingSummary {
    TimingStats headline;
    std::vector<TimingStats> runs;
};
TimingSummary combineRuns(const std::vector<TimingStats>& runs);

struct BenchOptions {
    uint32_t warmup = 30;
    uint32_t frames = 300;
    uint32_t runs = 3;
};

struct BenchResult {
    std::string scene;
    uint32_t scene_version = 0;
    std::string camera;
    uint32_t width = 0;
    uint32_t height = 0;
    std::string config_path;
    RenderConfig config;
    BenchOptions options;
    TimingSummary gpu_ms;
    TimingSummary cpu_prep_ms;
    double gpu_ms_per_spp = NAN;          // pt only
    uint32_t quality_spp = 0;             // pt: spp of the quality image
    uint64_t memory_renderer = 0;
    uint64_t memory_density = 0;
    int64_t memory_device_delta = 0;      // MTLDevice.currentAllocatedSize around renderer setup
    double compile_ms = 0.0;
    double bake_ms = 0.0;
    double model_load_ms = NAN;           // neural only
    bool has_model = false;
    std::string model_path;
    uint32_t model_format_version = 0;
    std::string model_sha256;
    uint32_t model_train_seed_first = 0;
    uint32_t model_train_seed_last = 0;
    std::vector<std::pair<std::string, double>> stats;  // renderer stats after the quality image
    bool has_metrics = false;
    double relMSE = NAN;
    double psnr = NAN;
    double ssim = NAN;
    std::string reference_id;
    bool validation_enabled = false;
    std::string gpu_name;
    std::string os_version;
};

// Bakes nothing: `volume` is the scene's shared density. Times warmup+frames frames per run with no
// CPU wait or readback inside the measured loop, then renders the quality image (pt: resets and
// accumulates config.spp samples; others: one frame) into *image.
bool runBenchmark(MetalContext& context, const Scene& scene, const DensityVolume& volume,
                  const CameraDesc& camera, const std::string& configPath, const RenderConfig& config,
                  const BenchOptions& options, BenchResult* result, Image* image, std::string* error);

// relMSE / PSNR / SSIM of `image` against `reference` with the camera exposure.
void attachMetrics(BenchResult* result, const Image& image, const Image& reference, float exposureEv,
                   const std::string& referenceId);

// The protocol §6 record.
NSDictionary* benchRecordDictionary(const BenchResult& result);
std::string toJsonString(id object);  // pretty-printed, sorted keys, trailing newline
bool writeTextFile(const std::string& path, const std::string& text, std::string* error);

int cmdBench(int argc, char** argv);
