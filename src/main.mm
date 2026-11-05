// cloudnr command line: info, render, view, compare.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "bench.h"
#include "density_volume.h"
#include "image_io.h"
#include "metal_context.h"
#include "metrics.h"
#include "neural_renderer.h"
#include "reference.h"
#include "renderer.h"
#include "scene.h"
#include "training_dump.h"
#include "suite.h"
#include "viewer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace {

// Parses `--key value` pairs and bare `--flag`s after the subcommand (argv[2..]).
struct Args {
    std::map<std::string, std::string> values;
    std::set<std::string> flags;

    bool has(const std::string& key) const { return values.count(key) != 0; }
    std::string get(const std::string& key, const std::string& fallback = "") const {
        auto it = values.find(key);
        return it == values.end() ? fallback : it->second;
    }
};

bool parseArgs(int argc, char** argv, const std::set<std::string>& valueKeys,
               const std::set<std::string>& flagKeys, Args* args, std::string* error) {
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (flagKeys.count(arg)) {
            args->flags.insert(arg);
        } else if (valueKeys.count(arg)) {
            if (i + 1 >= argc) {
                *error = "missing value for " + arg;
                return false;
            }
            args->values[arg] = argv[++i];
        } else {
            *error = "unknown argument '" + arg + "'";
            return false;
        }
    }
    return true;
}

int fail(const std::string& message) {
    std::fprintf(stderr, "cloudnr: %s\n", message.c_str());
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

// Linear interpolation between closest ranks; q in [0,1].
double percentile(std::vector<double> values, double q) {
    if (values.empty()) return std::nan("");
    std::sort(values.begin(), values.end());
    const double pos = q * double(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (values[hi] - values[lo]) * (pos - double(lo));
}

std::string jsonStats(const std::vector<double>& values) {
    return "{\"median\":" + jsonNumber(percentile(values, 0.5)) + ",\"p5\":" + jsonNumber(percentile(values, 0.05)) +
           ",\"p95\":" + jsonNumber(percentile(values, 0.95)) + "}";
}

bool loadInputs(const Args& args, Scene* scene, RenderConfig* config, std::string* error) {
    if (!args.has("--scene") || !args.has("--config")) {
        *error = "--scene and --config are required";
        return false;
    }
    auto loadedScene = loadScene(args.get("--scene"), error);
    if (!loadedScene) return false;
    auto loadedConfig = loadRenderConfig(args.get("--config"), error);
    if (!loadedConfig) return false;
    *scene = *loadedScene;
    *config = *loadedConfig;
    return true;
}

int cmdInfo(int argc, char** argv) {
    Args args;
    std::string error;
    if (!parseArgs(argc, argv, {}, {}, &args, &error)) return fail(error);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return fail("no Metal device");
    std::printf("device: %s\n", device.name.UTF8String);
    std::printf("MTLGPUFamilyMetal4: %s\n", [device supportsFamily:MTLGPUFamilyMetal4] ? "true" : "false");
    std::printf("MTLGPUFamilyApple7: %s\n", [device supportsFamily:MTLGPUFamilyApple7] ? "true" : "false");
    std::printf("MTLGPUFamilyApple8: %s\n", [device supportsFamily:MTLGPUFamilyApple8] ? "true" : "false");
    std::printf("queryTimestampFrequency: %llu\n", static_cast<unsigned long long>([device queryTimestampFrequency]));
    std::printf("recommendedMaxWorkingSetSize: %llu\n",
                static_cast<unsigned long long>(device.recommendedMaxWorkingSetSize));
    std::printf("maxBufferLength: %llu\n", static_cast<unsigned long long>(device.maxBufferLength));
    return 0;
}

int cmdRender(int argc, char** argv) {
    Args args;
    std::string error;
    if (!parseArgs(argc, argv, {"--scene", "--config", "--camera", "--frames", "--out", "--png"}, {"--validate"},
                   &args, &error))
        return fail(error);
    if (args.flags.count("--validate")) ensureValidationEnvironment(argc, argv);

    Scene scene;
    RenderConfig config;
    if (!loadInputs(args, &scene, &config, &error)) return fail(error);
    const CameraDesc* camera = scene.camera(args.get("--camera"));
    if (!camera) return fail("camera '" + args.get("--camera") + "' not found in scene");

    uint32_t frames = config.renderer == "pt" ? config.spp : 1;
    if (args.has("--frames")) {
        const long parsed = std::strtol(args.get("--frames").c_str(), nullptr, 10);
        if (parsed <= 0) return fail("--frames must be a positive integer");
        frames = static_cast<uint32_t>(parsed);
    }

    auto context = MetalContext::create(&error);
    if (!context) return fail(error);
    auto volume = DensityVolume::bake(*context, scene, &error);
    if (!volume) return fail(error);
    auto renderer = makeRenderer(*context, scene, config, *volume, &error);
    if (!renderer) return fail(error);

    const SceneGPU sceneGPU = makeSceneGPU(scene, *camera);
    std::mutex gpuMutex;
    std::vector<double> gpuMs;
    std::vector<double> cpuMs;
    gpuMs.reserve(frames);
    cpuMs.reserve(frames);
    for (uint32_t f = 0; f < frames; ++f) {
        id<MTL4CommandBuffer> commandBuffer = context->beginFrame();
        const auto start = std::chrono::steady_clock::now();
        id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        renderer->encode(encoder, FrameInputs{sceneGPU, context->currentSlot()});
        [encoder endEncoding];
        context->endFrame(commandBuffer, [&gpuMutex, &gpuMs](double ms, NSError* e) {
            if (e) return;
            std::lock_guard<std::mutex> lock(gpuMutex);
            gpuMs.push_back(ms);
        });
        cpuMs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    context->waitIdle();

    Image image;
    if (!readOutput(*renderer, &image, &error)) return fail(error);
    if (args.has("--out") && !writePFM(image, args.get("--out"), &error)) return fail(error);
    if (args.has("--png") && !writePNG(image, camera->exposure_ev, args.get("--png"), &error)) return fail(error);

    std::string stats = "{";
    for (const auto& [key, value] : renderer->stats()) {
        if (stats.size() > 1) stats += ",";
        stats += jsonString(key) + ":" + jsonNumber(value);
    }
    stats += "}";

    std::vector<double> gpuCopy;
    {
        std::lock_guard<std::mutex> lock(gpuMutex);
        gpuCopy = gpuMs;
    }
    const uint64_t gpuErrors = context->gpuErrorCount();
    std::printf(
        "{\"renderer\":%s,\"scene\":%s,\"camera\":%s,\"width\":%u,\"height\":%u,\"frames\":%u,"
        "\"accumulated_samples\":%u,\"gpu_ms\":%s,\"cpu_encode_ms\":%s,\"compile_ms\":%s,\"bake_ms\":%s,"
        "\"memory_bytes\":{\"renderer\":%llu,\"density\":%llu},\"gpu_errors\":%llu,\"stats\":%s}\n",
        jsonString(renderer->name()).c_str(), jsonString(scene.name).c_str(), jsonString(camera->name).c_str(),
        scene.width, scene.height, frames, renderer->accumulatedSamples(), jsonStats(gpuCopy).c_str(),
        jsonStats(cpuMs).c_str(), jsonNumber(context->totalCompileMs()).c_str(),
        jsonNumber(volume->bakeGpuMs()).c_str(), static_cast<unsigned long long>(renderer->memoryBytes()),
        static_cast<unsigned long long>(volume->memoryBytes()), static_cast<unsigned long long>(gpuErrors),
        stats.c_str());
    return gpuErrors > 0 ? 1 : 0;
}

int cmdView(int argc, char** argv) {
    Args args;
    std::string error;
    if (!parseArgs(argc, argv, {"--scene", "--config", "--camera", "--autotest"}, {"--validate"}, &args, &error))
        return fail(error);
    if (args.flags.count("--validate")) ensureValidationEnvironment(argc, argv);

    Scene scene;
    RenderConfig config;
    if (!loadInputs(args, &scene, &config, &error)) return fail(error);
    auto context = MetalContext::create(&error);
    if (!context) return fail(error);
    const int code = runViewer(*context, scene, args.get("--camera"), config, &error, args.get("--autotest"));
    if (code != 0) fail(error);
    return code;
}

int cmdCompare(int argc, char** argv) {
    Args args;
    std::string error;
    if (!parseArgs(argc, argv, {"--test", "--ref", "--exposure", "--heatmap"}, {}, &args, &error)) return fail(error);
    if (!args.has("--test") || !args.has("--ref")) return fail("--test and --ref are required");
    const float exposure = std::strtof(args.get("--exposure", "0").c_str(), nullptr);

    Image test;
    Image ref;
    if (!readPFM(args.get("--test"), &test, &error)) return fail(error);
    if (!readPFM(args.get("--ref"), &ref, &error)) return fail(error);
    if (test.width != ref.width || test.height != ref.height)
        return fail("image dimensions differ");

    std::printf("{\"relMSE\":%s,\"psnr\":%s,\"ssim\":%s}\n", jsonNumber(relMSE(test, ref, exposure)).c_str(),
                jsonNumber(psnrDisplay(test, ref, exposure)).c_str(),
                jsonNumber(ssimDisplay(test, ref, exposure)).c_str());

    if (args.has("--heatmap")) {
        const std::string prefix = args.get("--heatmap");
        float refMax = 0.0f;
        for (float v : ref.rgb) refMax = std::max(refMax, v);
        // abs_max = 10% of the brightest reference value after exposure; guarded for black references.
        const float absMax = std::max(0.1f * refMax * std::exp2(exposure), 1e-6f);
        if (!writeDisplayPNG(absErrorHeatmap(test, ref, exposure, absMax), prefix + "_abs.png", &error))
            return fail(error);
        if (!writeDisplayPNG(relErrorHeatmap(test, ref, exposure), prefix + "_rel.png", &error)) return fail(error);
    }
    return 0;
}

void printUsage() {
    std::fprintf(stderr,
                 "usage:\n"
                 "  cloudnr info\n"
                 "  cloudnr render --scene S --config C [--camera NAME] [--frames N] [--out X.pfm] [--png X.png] "
                 "[--validate]\n"
                 "  cloudnr view --scene S --config C [--camera NAME] [--autotest PREFIX] [--validate]\n"
                 "  cloudnr compare --test A.pfm --ref R.pfm [--exposure EV] [--heatmap PREFIX]\n");
}

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc < 2) {
            printUsage();
            return 1;
        }
        const std::string command = argv[1];
        if (command == "info") return cmdInfo(argc, argv);
        if (command == "render") return cmdRender(argc, argv);
        if (command == "view") return cmdView(argc, argv);
        if (command == "compare") return cmdCompare(argc, argv);
        if (command == "reference") return cmdReference(argc, argv);
        if (command == "dump-training") return cmdDumpTraining(argc, argv);
        if (command == "neural-check") return cmdNeuralCheck(argc, argv);
        if (command == "bench") return cmdBench(argc, argv);
        if (command == "suite") return cmdSuite(argc, argv);
        printUsage();
        return 1;
    }
}
