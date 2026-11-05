#include "training_dump.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "density_volume.h"
#include "metal_context.h"
#include "neural_model.h"
#include "neural_types.h"
#include "scene.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kPathsPerThread = 32;
constexpr uint32_t kMaxTries = 64;                       // camera rays per record
constexpr uint64_t kThreadCapacity = uint64_t(1) << 17;  // per dispatch
constexpr double kTargetBatchMs = 150.0;                 // well below the ~0.5 s watchdog budget
constexpr uint32_t kFirstTrainingSeed = 1000;            // shape seeds 1-999 are reserved for evaluation
constexpr float kSigmaMin = 0.02f;                       // log-uniform sigma_t_max range (1/m)
constexpr float kSigmaMax = 0.1f;
constexpr double kSunElevationMinDeg = -10.0;
constexpr double kSunElevationMaxDeg = 90.0;
constexpr NSUInteger kParamsOffset = 256;
constexpr NSUInteger kParamsSize = 512;
static_assert(sizeof(SceneGPU) <= kParamsOffset, "SceneGPU must fit before the dump params");
static_assert(kParamsOffset + sizeof(DumpParams) <= kParamsSize, "dump params slot too small");

int fail(const std::string& message) {
    std::fprintf(stderr, "cloudnr: dump-training: %s\n", message.c_str());
    return 1;
}

bool parseUint(const std::string& text, uint32_t* out) {
    if (text.empty() || text[0] == '-') return false;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (*end != '\0' || value > 0xFFFFFFFFull) return false;
    *out = static_cast<uint32_t>(value);
    return true;
}

std::string jsonEscape(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        if (static_cast<unsigned char>(c) < 0x20) continue;
        out += c;
    }
    return out + "\"";
}

// Training shape k, deterministic in (shapeBegin, k). fixedShape keeps the template cloud and sigma_t_max
// (the article's single-cloud weights); otherwise the cloud and sigma_t_max are randomized.
struct DumpShape {
    Scene scene;
    uint32_t shapeSeed = 0;
    uint32_t dumpSeed = 0;
    double sunElevationDeg = 0.0;
};

DumpShape makeDumpShape(const Scene& templ, uint32_t shapeBegin, uint32_t shapeEnd, uint32_t k, bool fixedShape) {
    std::seed_seq seq{shapeBegin, k};
    std::mt19937_64 rng(seq);
    // 53-bit uniform in [0, 1); independent of the standard library's distribution algorithms.
    auto uniform = [&rng] { return double(rng() >> 11) * 0x1.0p-53; };

    DumpShape out;
    out.shapeSeed = shapeBegin + k % (shapeEnd - shapeBegin);
    out.scene = templ;
    Scene& s = out.scene;
    if (!fixedShape) {
        s.name = templ.name + "_shape_" + std::to_string(out.shapeSeed);
        s.volume.density = "cumulus";
        s.volume.cloud = makeRandomCumulus(out.shapeSeed, s.volume);
    }
    const double kPi = 3.14159265358979323846;
    out.sunElevationDeg = kSunElevationMinDeg + (kSunElevationMaxDeg - kSunElevationMinDeg) * uniform();
    const double elevation = out.sunElevationDeg * kPi / 180.0;
    const double azimuth = 2.0 * kPi * uniform();
    s.light.sun_direction = simd_normalize(simd_make_float3(float(std::cos(elevation) * std::cos(azimuth)),
                                                            float(std::sin(elevation)),
                                                            float(std::cos(elevation) * std::sin(azimuth))));
    const double logMin = std::log(double(kSigmaMin)), logMax = std::log(double(kSigmaMax));
    const float sigma = float(std::exp(logMin + (logMax - logMin) * uniform()));
    if (!fixedShape) s.medium.sigma_t_max = sigma;
    // Targets are light coefficients; unit sources keep the scene self-consistent.
    s.light.sun_irradiance = simd_make_float3(1, 1, 1);
    s.light.sky_radiance = simd_make_float3(1, 1, 1);
    s.light.ground_radiance = simd_make_float3(1, 1, 1);
    out.dumpSeed = static_cast<uint32_t>(rng() >> 32);
    return out;
}

// One in-flight dispatch; index i of the batch array owns buffer set i.
struct Batch {
    bool active = false;
    uint64_t frame = 0;
    uint32_t count = 0;
    double gpuMs = 0.0;
    std::string error;
};

}  // namespace

int cmdDumpTraining(int argc, char** argv) {
    std::map<std::string, std::string> values = {{"--paths", "256"}, {"--template", "scenes/cumulus.json"}};
    const std::vector<std::string> keys = {"--out",    "--shape-begin", "--shape-end", "--shapes",
                                           "--records-per-shape", "--paths", "--template"};
    bool validate = false;
    bool fixedShape = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--validate") {
            validate = true;
        } else if (arg == "--fixed-shape") {
            fixedShape = true;
        } else if (std::find(keys.begin(), keys.end(), arg) != keys.end() && i + 1 < argc) {
            values[arg] = argv[++i];
        } else {
            return fail("unknown or incomplete argument '" + arg + "'");
        }
    }
    for (const char* required : {"--out", "--shape-begin", "--shape-end", "--shapes", "--records-per-shape"})
        if (!values.count(required)) return fail(std::string(required) + " is required");

    uint32_t shapeBegin = 0, shapeEnd = 0, shapeCount = 0, recordsPerShape = 0, paths = 0;
    if (!parseUint(values["--shape-begin"], &shapeBegin) || !parseUint(values["--shape-end"], &shapeEnd) ||
        !parseUint(values["--shapes"], &shapeCount) || !parseUint(values["--records-per-shape"], &recordsPerShape) ||
        !parseUint(values["--paths"], &paths))
        return fail("numeric arguments must be non-negative integers");
    if (!fixedShape && shapeBegin < kFirstTrainingSeed)
        return fail("--shape-begin must be >= 1000 (shape seeds 1-999 are reserved for evaluation)");
    if (shapeEnd <= shapeBegin) return fail("--shape-end must be greater than --shape-begin (exclusive end)");
    if (shapeCount == 0 || recordsPerShape == 0 || paths == 0)
        return fail("--shapes, --records-per-shape and --paths must be > 0");
    const uint32_t chunks = (paths + kPathsPerThread - 1) / kPathsPerThread;
    if (chunks > kThreadCapacity) return fail("--paths too large");
    const uint32_t recordCapacity = static_cast<uint32_t>(kThreadCapacity / chunks);

    if (validate) ensureValidationEnvironment(argc, argv);

    std::string error;
    const std::optional<Scene> templ = loadScene(values["--template"], &error);
    if (!templ) return fail(error);
    if (fixedShape && templ->volume.density != "cumulus")
        return fail("--fixed-shape needs a cumulus --template");
    const NeuralDims dims;

    auto context = MetalContext::create(&error);
    if (!context) return fail(error);
    id<MTLDevice> device = context->device();
    id<MTLComputePipelineState> pipeline =
        context->makeComputePipeline(neuralShaderFiles("src/shaders/neural_dump.metal"), "neural_dump",
                                     neuralShaderDefines(dims, NeuralInference::Scalar), &error);
    if (!pipeline) return fail(error);
    id<MTL4ArgumentTable> table = context->makeArgumentTable(4, 1, &error);
    if (!table) return fail(error);

    id<MTLBuffer> params[2];
    id<MTLBuffer> records[2];
    id<MTLBuffer> partials[2];
    std::vector<id<MTLAllocation>> resident;
    for (int i = 0; i < 2; ++i) {
        params[i] = [device newBufferWithLength:kParamsSize options:MTLResourceStorageModeShared];
        records[i] = [device newBufferWithLength:uint64_t(recordCapacity) * CNR_DUMP_RECORD_STRIDE * sizeof(float)
                                         options:MTLResourceStorageModeShared];
        partials[i] = [device newBufferWithLength:kThreadCapacity * CNR_DUMP_PARTIAL_STRIDE * sizeof(float)
                                          options:MTLResourceStorageModeShared];
        if (!params[i] || !records[i] || !partials[i]) return fail("buffer allocation failed");
        resident.insert(resident.end(), {params[i], records[i], partials[i]});
    }
    for (id<MTLAllocation> allocation : resident) context->addResident(allocation);
    context->commitResidency();

    const std::string outPath = values["--out"];
    FILE* file = std::fopen(outPath.c_str(), "wb");
    if (!file) return fail("cannot open " + outPath + " for writing");
    CnrdHeader header{};
    std::memcpy(header.magic, "CNRD", 4);
    header.version = 2;
    header.n_in = dims.n_in;
    header.n_out = dims.n_out;
    header.n_records = 0;
    header.shape_begin = shapeBegin;
    header.shape_end = shapeEnd;
    header.n_shapes = shapeCount;
    header.records_per_shape = recordsPerShape;
    header.paths = paths;
    header.hg_g = templ->medium.hg_g;
    header.albedo = templ->medium.albedo;
    header.sigma_t_min = fixedShape ? templ->medium.sigma_t_max : kSigmaMin;
    header.sigma_t_max = fixedShape ? templ->medium.sigma_t_max : kSigmaMax;
    header.fixed_shape = fixedShape ? 1u : 0u;
    bool ioOk = std::fwrite(&header, sizeof(header), 1, file) == 1;

    const auto wallStart = std::chrono::steady_clock::now();
    uint64_t written = 0, dropped = 0, capHits = 0;
    double gpuMsTotal = 0.0, bakeMsTotal = 0.0;
    uint32_t batchRecords = std::max<uint32_t>(1, std::min<uint32_t>(recordCapacity, 2048 / chunks));
    Batch batches[2];
    uint64_t sequence = 0;
    std::vector<float> row(dims.n_in + dims.n_out);

    // Waits for batch `i`, appends its valid records to the file and adapts the batch size.
    auto harvest = [&](int i) -> bool {
        Batch& b = batches[i];
        context->waitForFrame(b.frame);
        b.active = false;
        if (!b.error.empty()) {
            error = "GPU error: " + b.error;
            return false;
        }
        gpuMsTotal += b.gpuMs;
        const float* rec = static_cast<const float*>(records[i].contents);
        const float* part = static_cast<const float*>(partials[i].contents);
        for (uint32_t r = 0; r < b.count; ++r) {
            const float* descriptor = rec + size_t(r) * CNR_DUMP_RECORD_STRIDE;
            if (descriptor[dims.n_in] != 1.0f) {
                ++dropped;
                continue;
            }
            double sums[3] = {0.0, 0.0, 0.0};
            for (uint32_t c = 0; c < chunks; ++c) {
                const float* p = part + (size_t(r) * chunks + c) * CNR_DUMP_PARTIAL_STRIDE;
                sums[0] += p[0];
                sums[1] += p[1];
                sums[2] += p[2];
                capHits += static_cast<uint64_t>(p[3]);
            }
            std::copy(descriptor, descriptor + dims.n_in, row.begin());
            for (int j = 0; j < 3; ++j) row[dims.n_in + j] = float(sums[j] / double(paths));
            if (!std::all_of(row.begin(), row.end(), [](float v) { return std::isfinite(v); })) {
                error = "non-finite record produced by neural_dump";
                return false;
            }
            ioOk = ioOk && std::fwrite(row.data(), sizeof(float), row.size(), file) == row.size();
            ++written;
        }
        if (b.gpuMs > 0.0) {
            const double perRecord = b.gpuMs / double(b.count);
            const double next = std::min(kTargetBatchMs / perRecord, 4.0 * double(batchRecords));
            batchRecords = static_cast<uint32_t>(std::clamp(next, 1.0, double(recordCapacity)));
        }
        return true;
    };

    for (uint32_t k = 0; k < shapeCount; ++k) {
        const DumpShape shape = makeDumpShape(*templ, shapeBegin, shapeEnd, k, fixedShape);
        const Scene& scene = shape.scene;
        auto volume = DensityVolume::bake(*context, scene, &error);
        if (!volume) {
            std::fclose(file);
            return fail(error);
        }
        bakeMsTotal += volume->bakeGpuMs();
        const CameraDesc camera = scene.cameras.empty() ? CameraDesc{} : scene.cameras[0];
        const SceneGPU sceneGPU = makeSceneGPU(scene, camera);
        const double shapeGpuStart = gpuMsTotal;
        const uint64_t shapeWrittenStart = written;
        const uint64_t shapeDroppedStart = dropped;

        uint32_t offset = 0;
        while (offset < recordsPerShape) {
            const int set = static_cast<int>(sequence % 2);
            if (batches[set].active && !harvest(set)) {
                std::fclose(file);
                return fail(error);
            }
            Batch& b = batches[set];
            b.count = std::min(batchRecords, recordsPerShape - offset);
            b.gpuMs = 0.0;
            b.error.clear();

            DumpParams dump{};
            dump.seed = shape.dumpSeed;
            dump.record_offset = offset;
            dump.record_count = b.count;
            dump.paths = paths;
            dump.paths_per_thread = kPathsPerThread;
            dump.chunks = chunks;
            dump.max_tries = kMaxTries;
            uint8_t* slot = static_cast<uint8_t*>(params[set].contents);
            std::memcpy(slot, &sceneGPU, sizeof(SceneGPU));
            std::memcpy(slot + kParamsOffset, &dump, sizeof(DumpParams));

            [table setAddress:params[set].gpuAddress atIndex:0];
            [table setAddress:params[set].gpuAddress + kParamsOffset atIndex:1];
            [table setAddress:records[set].gpuAddress atIndex:2];
            [table setAddress:partials[set].gpuAddress atIndex:3];
            [table setTexture:volume->texture().gpuResourceID atIndex:0];

            id<MTL4CommandBuffer> commandBuffer = context->beginFrame();
            id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setArgumentTable:table];
            [encoder dispatchThreads:MTLSizeMake(NSUInteger(b.count) * chunks, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [encoder endEncoding];
            b.frame = context->endFrame(commandBuffer, [&b](double ms, NSError* e) {
                b.gpuMs = ms;
                if (e) b.error = e.localizedDescription.UTF8String;
            });
            b.active = true;
            offset += b.count;
            ++sequence;
        }
        // Drain (oldest first) before the volume is released.
        for (int n = 0; n < 2; ++n) {
            const int set = static_cast<int>((sequence + n) % 2);
            if (batches[set].active && !harvest(set)) {
                std::fclose(file);
                return fail(error);
            }
        }
        std::fprintf(stderr,
                     "dump-training: shape %u/%u seed=%u sigma_t_max=%.5f sun_elevation_deg=%.2f records=%llu "
                     "dropped=%llu gpu_ms=%.1f total_records=%llu\n",
                     k + 1, shapeCount, shape.shapeSeed, scene.medium.sigma_t_max, shape.sunElevationDeg,
                     static_cast<unsigned long long>(written - shapeWrittenStart),
                     static_cast<unsigned long long>(dropped - shapeDroppedStart), gpuMsTotal - shapeGpuStart,
                     static_cast<unsigned long long>(written));
    }

    for (id<MTLAllocation> allocation : resident) context->removeResident(allocation);
    context->commitResidency();

    header.n_records = written;
    ioOk = ioOk && std::fseek(file, 0, SEEK_SET) == 0 && std::fwrite(&header, sizeof(header), 1, file) == 1;
    ioOk = (std::fclose(file) == 0) && ioOk;
    if (!ioOk) return fail("write error on " + outPath);

    const double wallS = std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
    const uint64_t gpuErrors = context->gpuErrorCount();
    std::printf("{\"out\":%s,\"version\":2,\"records\":%llu,\"dropped\":%llu,\"bounce_cap_hits\":%llu,"
                "\"shapes\":%u,\"shape_begin\":%u,\"shape_end\":%u,\"fixed_shape_template\":%s,\"records_per_shape\":%u,"
                "\"paths\":%u,\"hg_g\":%.9g,\"albedo\":%.9g,\"sigma_t_range\":[%.9g,%.9g],"
                "\"sun_elevation_deg\":[%.9g,%.9g],\"gpu_ms\":%.3f,\"bake_ms\":%.3f,\"compile_ms\":%.3f,"
                "\"wall_s\":%.3f,\"gpu_errors\":%llu}\n",
                jsonEscape(outPath).c_str(), static_cast<unsigned long long>(written),
                static_cast<unsigned long long>(dropped), static_cast<unsigned long long>(capHits), shapeCount,
                shapeBegin, shapeEnd, fixedShape ? jsonEscape(templ->name).c_str() : "null", recordsPerShape, paths,
                header.hg_g, header.albedo, header.sigma_t_min, header.sigma_t_max, kSunElevationMinDeg,
                kSunElevationMaxDeg, gpuMsTotal, bakeMsTotal, context->totalCompileMs(), wallS,
                static_cast<unsigned long long>(gpuErrors));
    return gpuErrors > 0 ? 1 : 0;
}
