#include "reference.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "density_volume.h"
#include "image_io.h"
#include "metal_context.h"
#include "metrics.h"
#include "path_tracer.h"
#include "renderer.h"
#include "scene.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace {

int fail(const std::string& message) {
    std::fprintf(stderr, "cloudnr reference: %s\n", message.c_str());
    return 1;
}

std::string jsonString(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
            out += buf;
        } else {
            out += c;
        }
    }
    return out + "\"";
}

// JSON has no inf/NaN; non-finite values are written as null.
std::string jsonNumber(double v) {
    if (!std::isfinite(v)) return "null";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.9g", v);
    return buf;
}

double statValue(const Renderer& renderer, const std::string& key) {
    for (const auto& [name, value] : renderer.stats())
        if (name == key) return value;
    return 0.0;
}

bool isCheckpoint(uint32_t n, uint32_t spp) { return (n & (n - 1)) == 0 || n == spp; }

bool writeText(const std::string& path, const std::string& text, std::string* error) {
    FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) {
        *error = "cannot write " + path;
        return false;
    }
    const bool ok = std::fwrite(text.data(), 1, text.size(), file) == text.size();
    if (std::fclose(file) != 0 || !ok) {
        *error = "cannot write " + path;
        return false;
    }
    return true;
}

// Least-squares slope of log(relMSE) vs log(spp) over checkpoints with spp >= 16.
double convergenceSlope(const std::vector<std::pair<uint32_t, double>>& rows) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int count = 0;
    for (const auto& [spp, err] : rows) {
        if (spp < 16 || !(err > 0.0) || !std::isfinite(err)) continue;
        const double x = std::log(double(spp));
        const double y = std::log(err);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
        ++count;
    }
    const double denom = count * sxx - sx * sx;
    if (count < 2 || denom <= 0.0) return std::nan("");
    return (count * sxy - sx * sy) / denom;
}

bool makeDirectory(const std::string& path, std::string* error) {
    NSError* dirError = nil;
    if ([[NSFileManager defaultManager] createDirectoryAtPath:[NSString stringWithUTF8String:path.c_str()]
                                  withIntermediateDirectories:YES
                                                   attributes:nil
                                                        error:&dirError])
        return true;
    *error = "cannot create " + path + ": " + dirError.localizedDescription.UTF8String;
    return false;
}

double jsonDouble(NSDictionary* dict, NSString* key) {
    id value = dict[key];
    return [value isKindOfClass:[NSNumber class]] ? [value doubleValue] : std::nan("");
}

}  // namespace

bool renderProgressive(MetalContext& context, const Scene& scene, const RenderConfig& ptConfig,
                       const DensityVolume& volume, const SceneGPU& sceneGPU, uint32_t spp,
                       const std::function<void(uint32_t, const Image&, double)>& atCheckpoint,
                       ProgressiveResult* out, std::string* error) {
    auto renderer = makePathTracerRenderer(context, scene, ptConfig, volume, error);
    if (!renderer) return false;

    std::mutex gpuMutex;
    double gpuMs = 0.0;
    const uint64_t errorsBefore = context.gpuErrorCount();
    for (uint32_t n = 1; n <= spp; ++n) {
        id<MTL4CommandBuffer> commandBuffer = context.beginFrame();
        id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        renderer->encode(encoder, FrameInputs{sceneGPU, context.currentSlot()});
        [encoder endEncoding];
        context.endFrame(commandBuffer, [&gpuMutex, &gpuMs](double ms, NSError* e) {
            if (e) return;
            std::lock_guard<std::mutex> lock(gpuMutex);
            gpuMs += ms;
        });
        if (!isCheckpoint(n, spp)) continue;

        context.waitIdle();
        if (context.gpuErrorCount() != errorsBefore) {
            if (error) *error = "GPU error while rendering seed " + std::to_string(ptConfig.seed);
            return false;
        }
        Image image;
        if (!readOutput(*renderer, &image, error)) return false;
        double gpuSoFar;
        {
            std::lock_guard<std::mutex> lock(gpuMutex);
            gpuSoFar = gpuMs;
        }
        atCheckpoint(n, image, gpuSoFar);
        if (n == spp) {
            out->image = std::move(image);
            out->gpu_ms = gpuSoFar;
        }
    }
    out->bounce_cap_hits = statValue(*renderer, "bounce_cap_hits");
    out->mean_bounces = statValue(*renderer, "mean_bounces");
    return true;
}

std::string referenceResultJson(const ReferenceResult& r) {
    char seeds[48];
    std::snprintf(seeds, sizeof(seeds), "[%u,%u]", r.seed_a, r.seed_b);
    return "{\"scene\":" + jsonString(r.scene) + ",\"scene_version\":" + std::to_string(r.scene_version) +
           ",\"camera\":" + jsonString(r.camera) + ",\"width\":" + std::to_string(r.width) +
           ",\"height\":" + std::to_string(r.height) + ",\"spp_per_half\":" + std::to_string(r.spp_per_half) +
           ",\"seeds\":" + seeds + ",\"est_ref_relMSE\":" + jsonNumber(r.est_ref_relMSE) +
           ",\"meets_abs_gate\":" + (r.meets_abs_gate ? "true" : "false") +
           ",\"convergence_slope\":" + jsonNumber(r.convergence_slope) +
           ",\"bounce_cap_hits\":" + jsonNumber(r.bounce_cap_hits) + ",\"mean_bounces\":" + jsonNumber(r.mean_bounces) +
           ",\"gpu_ms_total\":" + jsonNumber(r.gpu_ms_total) + ",\"wall_ms\":" + jsonNumber(r.wall_ms) +
           ",\"compile_ms\":" + jsonNumber(r.compile_ms) + ",\"bake_ms\":" + jsonNumber(r.bake_ms) +
           ",\"max_bounces\":" + std::to_string(r.max_bounces) +
           ",\"rr_start_bounce\":" + std::to_string(r.rr_start_bounce) +
           ",\"renderer_version\":" + jsonString(r.renderer_version) + "}\n";
}

bool loadReferenceResult(const std::string& refJsonPath, ReferenceResult* r, std::string* error) {
    NSData* data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:refJsonPath.c_str()]];
    NSDictionary* d = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    if (![d isKindOfClass:[NSDictionary class]]) {
        if (error) *error = "cannot read " + refJsonPath;
        return false;
    }
    NSArray* seeds = d[@"seeds"];
    if (![d[@"scene"] isKindOfClass:[NSString class]] || ![d[@"camera"] isKindOfClass:[NSString class]] ||
        ![seeds isKindOfClass:[NSArray class]] || seeds.count != 2) {
        if (error) *error = "malformed " + refJsonPath;
        return false;
    }
    r->scene = [d[@"scene"] UTF8String];
    r->camera = [d[@"camera"] UTF8String];
    r->scene_version = [d[@"scene_version"] unsignedIntValue];
    r->width = [d[@"width"] unsignedIntValue];
    r->height = [d[@"height"] unsignedIntValue];
    r->spp_per_half = [d[@"spp_per_half"] unsignedIntValue];
    r->seed_a = [seeds[0] unsignedIntValue];
    r->seed_b = [seeds[1] unsignedIntValue];
    r->est_ref_relMSE = jsonDouble(d, @"est_ref_relMSE");
    r->meets_abs_gate = [d[@"meets_abs_gate"] boolValue];
    r->convergence_slope = jsonDouble(d, @"convergence_slope");
    r->bounce_cap_hits = jsonDouble(d, @"bounce_cap_hits");
    r->mean_bounces = jsonDouble(d, @"mean_bounces");
    r->gpu_ms_total = jsonDouble(d, @"gpu_ms_total");
    r->wall_ms = jsonDouble(d, @"wall_ms");
    r->compile_ms = jsonDouble(d, @"compile_ms");
    r->bake_ms = jsonDouble(d, @"bake_ms");
    r->max_bounces = [d[@"max_bounces"] unsignedIntValue];
    r->rr_start_bounce = [d[@"rr_start_bounce"] unsignedIntValue];
    r->renderer_version = [d[@"renderer_version"] isKindOfClass:[NSString class]] ? [d[@"renderer_version"] UTF8String] : "";
    return true;
}

bool generateReference(MetalContext& context, const Scene& scene, const DensityVolume& volume,
                       const CameraDesc& camera, const RenderConfig& ptConfig, uint32_t spp,
                       const std::string& outDir, ReferenceResult* result, std::string* error) {
    if (!makeDirectory(outDir, error)) return false;
    const float exposure = camera.exposure_ev;
    const SceneGPU sceneGPU = makeSceneGPU(scene, camera);
    RenderConfig configA = ptConfig;
    RenderConfig configB = ptConfig;
    configB.seed = ptConfig.seed + 1;
    const double compileBefore = context.totalCompileMs();

    const auto wallStart = std::chrono::steady_clock::now();
    std::map<uint32_t, std::pair<Image, double>> checkpointsA;  // spp -> (image, cumulative GPU ms)
    ProgressiveResult halfA;
    if (!renderProgressive(
            context, scene, configA, volume, sceneGPU, spp,
            [&](uint32_t n, const Image& image, double gpuMs) { checkpointsA[n] = {image, gpuMs}; }, &halfA, error))
        return false;

    std::string csv = "spp_per_half,relMSE_AB,est_ref_relMSE,gpu_ms_total\n";
    std::vector<std::pair<uint32_t, double>> rows;
    ProgressiveResult halfB;
    if (!renderProgressive(
            context, scene, configB, volume, sceneGPU, spp,
            [&](uint32_t n, const Image& image, double gpuMs) {
                auto it = checkpointsA.find(n);
                const double err = relMSE(it->second.first, image, exposure);
                rows.emplace_back(n, err);
                csv += std::to_string(n) + "," + jsonNumber(err) + "," + jsonNumber(err / 4.0) + "," +
                       jsonNumber(it->second.second + gpuMs) + "\n";
                checkpointsA.erase(it);  // free host memory as B catches up
            },
            &halfB, error))
        return false;
    const double wallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wallStart).count();

    Image ref;
    ref.width = halfA.image.width;
    ref.height = halfA.image.height;
    ref.rgb.resize(halfA.image.rgb.size());
    for (size_t i = 0; i < ref.rgb.size(); ++i) ref.rgb[i] = 0.5f * (halfA.image.rgb[i] + halfB.image.rgb[i]);

    ReferenceResult r;
    r.scene = scene.name;
    r.scene_version = scene.version;
    r.camera = camera.name;
    r.width = scene.width;
    r.height = scene.height;
    r.spp_per_half = spp;
    r.seed_a = configA.seed;
    r.seed_b = configB.seed;
    r.est_ref_relMSE = relMSE(halfA.image, halfB.image, exposure) / 4.0;
    // Only the absolute gate (protocol §2); the relative gate against compared methods is applied by the suite.
    r.meets_abs_gate = r.est_ref_relMSE <= 1e-4;
    r.convergence_slope = convergenceSlope(rows);
    r.bounce_cap_hits = halfA.bounce_cap_hits + halfB.bounce_cap_hits;
    r.mean_bounces = 0.5 * (halfA.mean_bounces + halfB.mean_bounces);
    r.gpu_ms_total = halfA.gpu_ms + halfB.gpu_ms;
    r.wall_ms = wallMs;
    r.compile_ms = context.totalCompileMs() - compileBefore;
    r.bake_ms = volume.bakeGpuMs();
    r.max_bounces = ptConfig.max_bounces;
    r.rr_start_bounce = ptConfig.rr_start_bounce;
    r.renderer_version = CLOUDNR_VERSION;

    if (!writeText(outDir + "/convergence.csv", csv, error)) return false;
    if (!writePFM(halfA.image, outDir + "/refA.pfm", error)) return false;
    if (!writePFM(halfB.image, outDir + "/refB.pfm", error)) return false;
    if (!writePFM(ref, outDir + "/ref.pfm", error)) return false;
    if (!writePNG(ref, exposure, outDir + "/ref.png", error)) return false;
    if (!writeText(outDir + "/ref.json", referenceResultJson(r), error)) return false;
    *result = r;
    return true;
}

int cmdReference(int argc, char** argv) {
    const std::set<std::string> valueKeys = {"--scene", "--camera", "--config", "--spp", "--out"};
    std::map<std::string, std::string> values;
    bool validate = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--validate") {
            validate = true;
        } else if (valueKeys.count(arg)) {
            if (i + 1 >= argc) return fail("missing value for " + arg);
            values[arg] = argv[++i];
        } else {
            return fail("unknown argument '" + arg + "'");
        }
    }
    if (!values.count("--scene") || !values.count("--spp") || !values.count("--out"))
        return fail("usage: cloudnr reference --scene S [--camera C] [--config configs/pt.json] --spp N --out DIR "
                    "[--validate]");
    if (validate) ensureValidationEnvironment(argc, argv);

    const long parsedSpp = std::strtol(values["--spp"].c_str(), nullptr, 10);
    if (parsedSpp <= 0 || parsedSpp > 0x7fffffffL) return fail("--spp must be a positive integer");
    const uint32_t spp = static_cast<uint32_t>(parsedSpp);
    const std::string outDir = values["--out"];
    const std::string configPath = values.count("--config") ? values["--config"] : "configs/pt.json";

    std::string error;
    auto scene = loadScene(values["--scene"], &error);
    if (!scene) return fail(error);
    auto config = loadRenderConfig(configPath, &error);
    if (!config) return fail(error);
    if (config->renderer != "pt") return fail("config '" + configPath + "' is not a pt config");
    const std::string cameraName = values.count("--camera") ? values["--camera"] : "";
    const CameraDesc* camera = scene->camera(cameraName);
    if (!camera) return fail("camera '" + cameraName + "' not found in scene");

    auto context = MetalContext::create(&error);
    if (!context) return fail(error);
    auto volume = DensityVolume::bake(*context, *scene, &error);
    if (!volume) return fail(error);
    ReferenceResult result;
    if (!generateReference(*context, *scene, *volume, *camera, *config, spp, outDir, &result, &error))
        return fail(error);
    std::fputs(referenceResultJson(result).c_str(), stdout);
    return context->gpuErrorCount() > 0 ? 1 : 0;
}
