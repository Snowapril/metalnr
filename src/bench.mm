#include "bench.h"

#import <Metal/Metal.h>

#include "density_volume.h"
#include "metal_context.h"
#include "metrics.h"
#include "neural_model.h"
#include "renderer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// JSON has no NaN/inf.
id jsonNumber(double v) { return std::isfinite(v) ? (id)@(v) : (id)[NSNull null]; }

NSString* ns(const std::string& s) { return [NSString stringWithUTF8String:s.c_str()]; }

NSDictionary* statsDictionary(const TimingStats& s) {
    return @{@"median" : jsonNumber(s.median), @"p5" : jsonNumber(s.p5), @"p95" : jsonNumber(s.p95)};
}

NSDictionary* summaryDictionary(const TimingSummary& s) {
    NSMutableDictionary* d = [statsDictionary(s.headline) mutableCopy];
    NSMutableArray* runs = [NSMutableArray array];
    for (const TimingStats& r : s.runs) [runs addObject:statsDictionary(r)];
    d[@"runs"] = runs;
    return d;
}

double statValue(const std::vector<std::pair<std::string, double>>& stats, const std::string& key) {
    for (const auto& [name, value] : stats)
        if (name == key) return value;
    return NAN;
}

// The exact config file content (the full RenderConfig as used).
id configObject(const std::string& path) {
    NSData* data = [NSData dataWithContentsOfFile:ns(path)];
    id object = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    return object ?: [NSNull null];
}

int fail(const std::string& message) {
    std::fprintf(stderr, "cloudnr bench: %s\n", message.c_str());
    return 1;
}

}  // namespace

TimingStats summarizeTimes(std::vector<double> values) {
    values.erase(std::remove_if(values.begin(), values.end(), [](double v) { return !std::isfinite(v); }),
                 values.end());
    TimingStats s;
    if (values.empty()) return s;
    std::sort(values.begin(), values.end());
    auto nearestRank = [&](double p) {
        const size_t rank = static_cast<size_t>(std::ceil(p / 100.0 * double(values.size())));
        return values[std::clamp<size_t>(rank, 1, values.size()) - 1];
    };
    s.median = nearestRank(50.0);
    s.p5 = nearestRank(5.0);
    s.p95 = nearestRank(95.0);
    return s;
}

TimingSummary combineRuns(const std::vector<TimingStats>& runs) {
    TimingSummary out;
    out.runs = runs;
    std::vector<double> medians, p5s, p95s;
    for (const TimingStats& r : runs) {
        medians.push_back(r.median);
        p5s.push_back(r.p5);
        p95s.push_back(r.p95);
    }
    out.headline.median = summarizeTimes(medians).median;
    out.headline.p5 = summarizeTimes(p5s).median;
    out.headline.p95 = summarizeTimes(p95s).median;
    return out;
}

bool runBenchmark(MetalContext& context, const Scene& scene, const DensityVolume& volume,
                  const CameraDesc& camera, const std::string& configPath, const RenderConfig& config,
                  const BenchOptions& options, BenchResult* result, Image* image, std::string* error) {
    BenchResult r;
    r.scene = scene.name;
    r.scene_version = scene.version;
    r.camera = camera.name;
    r.width = scene.width;
    r.height = scene.height;
    r.config_path = configPath;
    r.config = config;
    r.options = options;
    r.bake_ms = volume.bakeGpuMs();
    r.memory_density = volume.memoryBytes();
    r.validation_enabled = context.validationEnabled();
    r.gpu_name = context.device().name.UTF8String;
    r.os_version = NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String;
    if (r.validation_enabled)
        std::fprintf(stderr, "cloudnr bench: WARNING: Metal validation is enabled; timings are not protocol-valid\n");

    if (config.renderer == "neural") {
        auto model = loadNeuralModel(config.model, error);
        if (!model) return false;
        r.has_model = true;
        r.model_path = config.model;
        r.model_format_version = model->format_version;
        r.model_sha256 = model->sha256;
        if (model->has_train_seed_range) {
            r.model_train_seed_first = model->train_seed_first;
            r.model_train_seed_last = model->train_seed_last;
        }
    }

    // Setup costs (excluded from steady state).
    const double compileBefore = context.totalCompileMs();
    const uint64_t allocatedBefore = context.device().currentAllocatedSize;
    auto renderer = makeRenderer(context, scene, config, volume, error);
    if (!renderer) return false;
    r.memory_device_delta = int64_t(context.device().currentAllocatedSize) - int64_t(allocatedBefore);
    r.compile_ms = context.totalCompileMs() - compileBefore;
    r.memory_renderer = renderer->memoryBytes();
    r.model_load_ms = statValue(renderer->stats(), "model_load_ms");

    const SceneGPU sceneGPU = makeSceneGPU(scene, camera);
    const uint64_t errorsBefore = context.gpuErrorCount();
    auto encodeFrame = [&](MetalContext::FrameFeedback feedback) {
        id<MTL4CommandBuffer> commandBuffer = context.beginFrame();
        const auto start = Clock::now();
        id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        renderer->encode(encoder, FrameInputs{sceneGPU, context.currentSlot()});
        [encoder endEncoding];
        context.endFrame(commandBuffer, std::move(feedback));
        return msSince(start);
    };

    std::vector<TimingStats> gpuRuns, cpuRuns;
    for (uint32_t run = 0; run < options.runs; ++run) {
        renderer->resetAccumulation();
        // Written by the feedback handlers; waitIdle() waits for them before the vector is read.
        auto gpuSamples = std::make_shared<std::vector<double>>(options.frames, NAN);
        std::vector<double> cpuSamples(options.frames, NAN);
        for (uint32_t f = 0; f < options.warmup + options.frames; ++f) {
            const bool measured = f >= options.warmup;
            const uint32_t index = measured ? f - options.warmup : 0;
            MetalContext::FrameFeedback feedback = nullptr;
            if (measured)
                feedback = [gpuSamples, index](double ms, NSError* e) { (*gpuSamples)[index] = e ? NAN : ms; };
            const double cpuMs = encodeFrame(std::move(feedback));
            if (measured) cpuSamples[index] = cpuMs;
        }
        context.waitIdle();
        gpuRuns.push_back(summarizeTimes(*gpuSamples));
        cpuRuns.push_back(summarizeTimes(cpuSamples));
    }
    if (context.gpuErrorCount() != errorsBefore) {
        if (error) *error = "GPU error during timed frames";
        return false;
    }
    r.gpu_ms = combineRuns(gpuRuns);
    r.cpu_prep_ms = combineRuns(cpuRuns);

    // Quality image, outside timing.
    renderer->resetAccumulation();
    const uint32_t qualityFrames = config.renderer == "pt" ? config.spp : 1;
    for (uint32_t f = 0; f < qualityFrames; ++f) encodeFrame(nullptr);
    context.waitIdle();
    if (context.gpuErrorCount() != errorsBefore) {
        if (error) *error = "GPU error while rendering the quality image";
        return false;
    }
    if (!readOutput(*renderer, image, error)) return false;
    if (config.renderer == "pt") {
        r.gpu_ms_per_spp = r.gpu_ms.headline.median;
        r.quality_spp = renderer->accumulatedSamples();
    }
    r.stats = renderer->stats();
    *result = std::move(r);
    return true;
}

void attachMetrics(BenchResult* result, const Image& image, const Image& reference, float exposureEv,
                   const std::string& referenceId) {
    result->has_metrics = true;
    result->relMSE = relMSE(image, reference, exposureEv);
    result->psnr = psnrDisplay(image, reference, exposureEv);
    result->ssim = ssimDisplay(image, reference, exposureEv);
    result->reference_id = referenceId;
}

NSDictionary* benchRecordDictionary(const BenchResult& r) {
    const bool pt = r.config.renderer == "pt";
    NSMutableDictionary* d = [NSMutableDictionary dictionary];
    d[@"renderer_version"] = @CLOUDNR_VERSION;
    d[@"renderer"] = ns(r.config.renderer);
    d[@"config"] = configObject(r.config_path);
    d[@"config_path"] = ns(r.config_path);
    d[@"scene"] = ns(r.scene);
    d[@"scene_version"] = @(r.scene_version);
    d[@"camera"] = ns(r.camera);
    d[@"width"] = @(r.width);
    d[@"height"] = @(r.height);
    d[@"gpu_name"] = ns(r.gpu_name);
    d[@"os_version"] = ns(r.os_version);
    if (pt) {
        d[@"spp_or_quality"] = @{@"spp" : @(r.quality_spp)};
        d[@"seed"] = @(r.config.seed);
        d[@"gpu_ms_per_spp"] = jsonNumber(r.gpu_ms_per_spp);
    } else {
        const RenderConfig& c = r.config;
        NSMutableDictionary* q = [@{@"step_scale" : @(c.step_scale), @"max_steps" : @(c.max_steps),
                                    @"min_transmittance" : @(c.min_transmittance), @"dither" : @(c.dither),
                                    @"empty_skip" : @(c.empty_skip)} mutableCopy];
        if (c.renderer == "analytic") {
            q[@"shadow_steps"] = @(c.shadow_steps);
            q[@"shadow_distance"] = @(c.shadow_distance);
            q[@"ambient_steps"] = @(c.ambient_steps);
            q[@"octaves"] = @(c.octaves);
            q[@"ms_a"] = @(c.ms_a);
            q[@"ms_b"] = @(c.ms_b);
            q[@"ms_c"] = @(c.ms_c);
            q[@"ambient"] = @(c.ambient);
        } else if (c.renderer == "neural") {
            q[@"inference"] = ns(c.inference);
        }
        d[@"spp_or_quality"] = q;
        d[@"seed"] = [NSNull null];
    }
    d[@"gpu_ms"] = summaryDictionary(r.gpu_ms);
    d[@"cpu_prep_ms"] = summaryDictionary(r.cpu_prep_ms);
    d[@"timing"] = @{@"warmup" : @(r.options.warmup), @"frames" : @(r.options.frames), @"runs" : @(r.options.runs)};
    d[@"memory_bytes"] = @{@"renderer" : @(r.memory_renderer), @"density" : @(r.memory_density),
                           @"device_delta" : @(r.memory_device_delta)};
    d[@"excluded_costs_ms"] = @{@"compile" : jsonNumber(r.compile_ms), @"bake" : jsonNumber(r.bake_ms),
                                @"model_load" : jsonNumber(r.model_load_ms)};
    if (r.has_model) {
        d[@"model"] = @{@"path" : ns(r.model_path), @"format_version" : @(r.model_format_version),
                        @"sha256" : ns(r.model_sha256),
                        @"train_seed_range" : @[ @(r.model_train_seed_first), @(r.model_train_seed_last) ]};
    } else {
        d[@"model"] = [NSNull null];
    }
    d[@"metrics"] = r.has_metrics ? (id)@{@"relMSE" : jsonNumber(r.relMSE), @"psnr" : jsonNumber(r.psnr),
                                          @"ssim" : jsonNumber(r.ssim)}
                                  : (id)[NSNull null];
    d[@"reference_id"] = r.has_metrics ? (id)ns(r.reference_id) : (id)[NSNull null];
    if (pt) {
        d[@"bias"] = @{@"bounce_cap_hits" : jsonNumber(statValue(r.stats, "bounce_cap_hits")),
                       @"mean_bounces" : jsonNumber(statValue(r.stats, "mean_bounces"))};
    } else if (r.config.renderer == "neural") {
        d[@"bias"] = @{@"notes" : r.config.inference == "scalar"
                                      ? @"march discretization; network approximation of in-scattering; Monte Carlo "
                                        @"noise in training targets"
                                      : @"march discretization; network approximation of in-scattering; Monte Carlo "
                                        @"noise in training targets; fp16 activations and weights in matmul2d"};
    } else {
        d[@"bias"] = @{@"notes" : @"march discretization; octave multiple-scattering approximation; two-direction "
                                  @"ambient term"};
    }
    d[@"validation_enabled"] = @(r.validation_enabled);
    return d;
}

std::string toJsonString(id object) {
    NSData* data = [NSJSONSerialization dataWithJSONObject:object
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:nil];
    if (!data) return "null\n";
    return std::string(static_cast<const char*>(data.bytes), data.length) + "\n";
}

bool writeTextFile(const std::string& path, const std::string& text, std::string* error) {
    FILE* file = std::fopen(path.c_str(), "wb");
    const bool ok = file && std::fwrite(text.data(), 1, text.size(), file) == text.size();
    if (file && std::fclose(file) != 0) {
        if (error) *error = "cannot write " + path;
        return false;
    }
    if (!ok && error) *error = "cannot write " + path;
    return ok;
}

int cmdBench(int argc, char** argv) {
    const std::set<std::string> valueKeys = {"--scene", "--camera", "--config", "--reference", "--reference-id",
                                             "--warmup", "--frames", "--runs", "--out", "--image"};
    std::map<std::string, std::string> values;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (!valueKeys.count(arg)) return fail("unknown argument '" + arg + "'");
        if (i + 1 >= argc) return fail("missing value for " + arg);
        values[arg] = argv[++i];
    }
    if (!values.count("--scene") || !values.count("--config"))
        return fail("usage: cloudnr bench --scene S [--camera C] --config C [--reference ref.pfm "
                    "[--reference-id ID]] [--warmup N] [--frames N] [--runs N] [--out record.json] [--image out.pfm]");
    BenchOptions options;
    auto readCount = [&](const char* key, uint32_t* out, bool allowZero) {
        if (!values.count(key)) return true;
        const long v = std::strtol(values[key].c_str(), nullptr, 10);
        if (v < (allowZero ? 0 : 1) || v > 1000000) return false;
        *out = uint32_t(v);
        return true;
    };
    if (!readCount("--warmup", &options.warmup, true) || !readCount("--frames", &options.frames, false) ||
        !readCount("--runs", &options.runs, false))
        return fail("--warmup/--frames/--runs must be integers (frames, runs >= 1)");

    std::string error;
    auto scene = loadScene(values["--scene"], &error);
    if (!scene) return fail(error);
    auto config = loadRenderConfig(values["--config"], &error);
    if (!config) return fail(error);
    const std::string cameraName = values.count("--camera") ? values["--camera"] : "";
    const CameraDesc* camera = scene->camera(cameraName);
    if (!camera) return fail("camera '" + cameraName + "' not found in scene");

    auto context = MetalContext::create(&error);
    if (!context) return fail(error);
    auto volume = DensityVolume::bake(*context, *scene, &error);
    if (!volume) return fail(error);
    BenchResult result;
    Image image;
    if (!runBenchmark(*context, *scene, *volume, *camera, values["--config"], *config, options, &result, &image,
                      &error))
        return fail(error);
    if (values.count("--reference")) {
        Image reference;
        if (!readPFM(values["--reference"], &reference, &error)) return fail(error);
        attachMetrics(&result, image, reference, camera->exposure_ev,
                      values.count("--reference-id") ? values["--reference-id"] : values["--reference"]);
    }
    if (values.count("--image") && !writePFM(image, values["--image"], &error)) return fail(error);
    const std::string json = toJsonString(benchRecordDictionary(result));
    if (values.count("--out") && !writeTextFile(values["--out"], json, &error)) return fail(error);
    std::fputs(json.c_str(), stdout);
    return 0;
}
