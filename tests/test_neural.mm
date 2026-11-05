// Neural renderer tests (milestone neural inference): .cnrm loader validation, GPU mlp_eval vs a
// CPU reference, descriptor sanity on a constant-density box, and a small neural render.
// Runs under the Metal debug layer and shader validation. No Python.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "check.h"

#include "density_volume.h"
#include "image_io.h"
#include "metal_context.h"
#include "neural_model.h"
#include "neural_renderer.h"
#include "renderer.h"
#include "scene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

void report(const char* what, const std::string& error) { std::fprintf(stderr, "%s: %s\n", what, error.c_str()); }

uint16_t floatToHalfBits(float f) {
    const __fp16 h = static_cast<__fp16>(f);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

float halfBitsToFloat(uint16_t bits) {
    __fp16 h;
    std::memcpy(&h, &bits, sizeof(h));
    return static_cast<float>(h);
}

std::string tempDir() {
    const std::string dir = std::string(NSTemporaryDirectory().UTF8String) + "cloudnr_test_neural_" +
                            std::to_string(getpid());
    [[NSFileManager defaultManager] createDirectoryAtPath:[NSString stringWithUTF8String:dir.c_str()]
                              withIntermediateDirectories:YES
                                               attributes:nil
                                                    error:nil];
    return dir;
}

CnrmHeader validHeader() {
    CnrmHeader h{};
    std::memcpy(h.magic, "CNRM", 4);
    h.format_version = 1;
    h.n_in = 40;
    h.n_hidden = 32;
    h.n_layers = 3;
    h.n_out = 3;
    h.hg_g = 0.8f;
    h.albedo = 0.99f;
    h.levels = 5;
    return h;
}

// He-uniform random weights (fp16) per layer; output biases in [0.2, 0.8] so many outputs are positive.
std::vector<uint16_t> randomWeights(const NeuralDims& dims, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint16_t> weights;
    auto layer = [&](uint32_t in, uint32_t out, bool last) {
        std::uniform_real_distribution<float> w(-std::sqrt(6.0f / float(in)), std::sqrt(6.0f / float(in)));
        std::uniform_real_distribution<float> b = last ? std::uniform_real_distribution<float>(0.2f, 0.8f)
                                                       : std::uniform_real_distribution<float>(-0.1f, 0.1f);
        for (uint32_t i = 0; i < in * out; ++i) weights.push_back(floatToHalfBits(w(rng)));
        for (uint32_t i = 0; i < out; ++i) weights.push_back(floatToHalfBits(b(rng)));
    };
    layer(dims.n_in, dims.n_hidden, false);
    for (uint32_t l = 1; l < dims.n_layers; ++l) layer(dims.n_hidden, dims.n_hidden, false);
    layer(dims.n_hidden, dims.n_out, true);
    return weights;
}

bool writeCnrm(const std::string& path, const CnrmHeader& header, const std::vector<uint16_t>& weights,
               size_t dropBytes = 0) {
    std::vector<uint8_t> bytes(sizeof(header) + weights.size() * sizeof(uint16_t));
    std::memcpy(bytes.data(), &header, sizeof(header));
    std::memcpy(bytes.data() + sizeof(header), weights.data(), weights.size() * sizeof(uint16_t));
    bytes.resize(bytes.size() - dropBytes);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(out);
}

// CPU reference: same fp16 weights as float, float accumulation, ReLU hidden, f = expm1(max(y, 0)).
void cpuMlp(const NeuralDims& dims, const std::vector<uint16_t>& weights, const float* in, float* f) {
    std::vector<float> a(in, in + dims.n_in);
    size_t offset = 0;
    auto layer = [&](uint32_t nIn, uint32_t nOut, bool relu) {
        std::vector<float> out(nOut);
        for (uint32_t o = 0; o < nOut; ++o) {
            float acc = halfBitsToFloat(weights[offset + size_t(nOut) * nIn + o]);
            for (uint32_t i = 0; i < nIn; ++i) acc += halfBitsToFloat(weights[offset + size_t(o) * nIn + i]) * a[i];
            out[o] = relu ? std::max(acc, 0.0f) : acc;
        }
        offset += size_t(nOut) * nIn + nOut;
        a = std::move(out);
    };
    layer(dims.n_in, dims.n_hidden, true);
    for (uint32_t l = 1; l < dims.n_layers; ++l) layer(dims.n_hidden, dims.n_hidden, true);
    layer(dims.n_hidden, dims.n_out, false);
    for (uint32_t o = 0; o < dims.n_out; ++o) f[o] = std::expm1(std::max(a[o], 0.0f));
}

Scene makeConstantScene(uint32_t gridSide, float constantDensity) {
    Scene scene;
    scene.name = "neural_test";
    CameraDesc camera;
    camera.name = "side";
    camera.position = simd_make_float3(0, 200, -1500);
    camera.target = simd_make_float3(0, 200, 0);
    scene.cameras.push_back(camera);
    scene.width = 32;
    scene.height = 18;
    scene.volume.grid = simd_make_uint3(gridSide, gridSide, gridSide);
    scene.volume.density = "constant";
    scene.volume.constant_density = constantDensity;
    scene.medium.sigma_t_max = 0.04f;
    scene.medium.albedo = 0.99f;
    scene.medium.hg_g = 0.8f;
    scene.light.sun_direction = simd_normalize(scene.light.sun_direction);
    return scene;
}

struct ScopedResidency {
    MetalContext& context;
    std::vector<id<MTLAllocation>> allocations;

    ScopedResidency(MetalContext& c, std::vector<id<MTLAllocation>> a) : context(c), allocations(std::move(a)) {
        for (id<MTLAllocation> allocation : allocations) context.addResident(allocation);
        context.commitResidency();
    }
    ~ScopedResidency() {
        for (id<MTLAllocation> allocation : allocations) context.removeResident(allocation);
        context.commitResidency();
    }
};

void testLoader(const std::string& dir) {
    const NeuralDims dims;
    const std::vector<uint16_t> weights = randomWeights(dims, 7);
    CHECK(weights.size() == neuralWeightCount(dims));
    CHECK(neuralWeightCount(dims) == 3523);  // 40*32+32 + 2*(32*32+32) + 32*3+3

    std::string error;
    const std::string good = dir + "/good.cnrm";
    CHECK(writeCnrm(good, validHeader(), weights));
    std::optional<NeuralModel> model = loadNeuralModel(good, &error);
    CHECK(model.has_value());
    if (!model) return report("load good model", error);
    CHECK(model->weights == weights);
    CHECK(model->dims.n_in == 40 && model->dims.n_hidden == 32 && model->dims.n_layers == 3 &&
          model->dims.n_out == 3 && model->dims.levels == 5);
    CHECK(model->sha256.size() == 64);
    CHECK(!model->has_train_seed_range);

    Scene scene = makeConstantScene(16, 1.0f);
    CHECK(validateNeuralModelForScene(*model, scene, &error));
    Scene otherG = scene;
    otherG.medium.hg_g = 0.5f;
    std::string gError;
    CHECK(!validateNeuralModelForScene(*model, otherG, &gError));
    CHECK(!gError.empty());
    Scene otherAlbedo = scene;
    otherAlbedo.medium.albedo = 0.9f;
    CHECK(!validateNeuralModelForScene(*model, otherAlbedo, &error));

    auto rejects = [&](const char* name, const CnrmHeader& header, const std::vector<uint16_t>& w, size_t drop) {
        const std::string path = dir + "/" + name + ".cnrm";
        CHECK(writeCnrm(path, header, w, drop));
        std::string e;
        const bool rejected = !loadNeuralModel(path, &e).has_value();
        CHECK(rejected);
        CHECK(!e.empty());
        if (!rejected) std::fprintf(stderr, "loader accepted invalid model '%s'\n", name);
    };
    CnrmHeader badMagic = validHeader();
    badMagic.magic[0] = 'X';
    rejects("bad_magic", badMagic, weights, 0);
    CnrmHeader badVersion = validHeader();
    badVersion.format_version = 2;
    rejects("bad_version", badVersion, weights, 0);
    // Self-consistent file for a different architecture: only the dimension check can reject it.
    CnrmHeader badDims = validHeader();
    badDims.n_hidden = 16;
    NeuralDims smallDims;
    smallDims.n_hidden = 16;
    rejects("bad_dims", badDims, randomWeights(smallDims, 3), 0);
    CnrmHeader badLevels = validHeader();
    badLevels.levels = 4;
    rejects("bad_levels", badLevels, weights, 0);
    rejects("truncated", validHeader(), weights, 2);
    std::vector<uint16_t> extra = weights;
    extra.push_back(0);
    rejects("extra_bytes", validHeader(), extra, 0);
    std::string missingError;
    CHECK(!loadNeuralModel(dir + "/does_not_exist.cnrm", &missingError).has_value());
}

void testMlpParity(MetalContext& context, const std::string& dir) {
    const NeuralDims dims;
    const std::vector<uint16_t> weights = randomWeights(dims, 11);
    const std::string path = dir + "/parity.cnrm";
    CHECK(writeCnrm(path, validHeader(), weights));
    std::string error;
    std::optional<NeuralModel> model = loadNeuralModel(path, &error);
    CHECK(model.has_value());
    if (!model) return report("load parity model", error);

    constexpr size_t kCount = 256;
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> u(-1.0f, 2.0f);
    std::vector<float> inputs(kCount * dims.n_in);
    for (float& v : inputs) v = u(rng);

    std::vector<float> outputs;
    const bool ok = evalNeuralBatch(context, *model, inputs, &outputs, &error);
    CHECK(ok);
    if (!ok) return report("evalNeuralBatch", error);
    CHECK(outputs.size() == kCount * 3);

    double maxRel = 0.0;
    size_t positive = 0;
    bool finite = true;
    for (size_t i = 0; i < kCount; ++i) {
        float expected[3];
        cpuMlp(dims, weights, inputs.data() + i * dims.n_in, expected);
        for (int c = 0; c < 3; ++c) {
            const double a = outputs[i * 3 + c];
            const double b = expected[c];
            finite = finite && std::isfinite(a);
            maxRel = std::max(maxRel, std::fabs(a - b) / (std::fabs(b) + 1e-3));
            if (b > 1e-3) ++positive;
        }
    }
    CHECK(finite);
    CHECK(maxRel <= 1e-3);
    CHECK(positive > kCount / 4);  // the comparison must exercise non-zero outputs
    if (maxRel > 1e-3) std::fprintf(stderr, "mlp parity: max rel error %.3g\n", maxRel);
}

void testDescriptor(MetalContext& context) {
    constexpr float kDensity = 0.5f;
    const Scene scene = makeConstantScene(64, kDensity);
    std::string error;
    auto volume = DensityVolume::bake(context, scene, &error);
    CHECK(volume != nullptr);
    if (!volume) return report("descriptor bake", error);
    const SceneGPU sceneGPU = makeSceneGPU(scene, scene.cameras[0]);
    const NeuralDims dims;

    id<MTLComputePipelineState> pipeline =
        context.makeComputePipeline(neuralShaderFiles("tests/shaders/test_neural.metal"), "test_build_descriptor",
                                    neuralShaderDefines(dims), &error);
    id<MTL4ArgumentTable> table = context.makeArgumentTable(4, 1, &error);
    CHECK(pipeline != nil && table != nil);
    if (!pipeline || !table) return report("descriptor setup", error);

    // Interior point: the largest offset (16 voxels at level 4) stays inside the 64^3 box.
    const simd_float3 center = (scene.volume.bounds_min + scene.volume.bounds_max) * 0.5f;
    const simd_float3 sun = sceneGPU.sun_direction;
    const simd_float3 dirs3[] = {
        simd_normalize(simd_make_float3(1.0f, 0.2f, 0.3f)),
        simd_normalize(simd_make_float3(-0.4f, -0.8f, 0.1f)),
        sun,   // parallel to the sun: fallback frame
        -sun,  // anti-parallel
    };
    const float taus[] = {0.0f, 0.7f, 2.5f, 10.0f};
    constexpr size_t kQueries = 4;
    simd_float4 queries[kQueries];
    simd_float4 dirs[kQueries];
    for (size_t i = 0; i < kQueries; ++i) {
        queries[i] = simd_make_float4(center, taus[i]);
        dirs[i] = simd_make_float4(dirs3[i], 0.0f);
    }
    std::vector<float> init(kQueries * dims.n_in, -1.0f);
    id<MTLDevice> device = context.device();
    id<MTLBuffer> sceneBuffer = [device newBufferWithBytes:&sceneGPU length:sizeof(sceneGPU)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> queryBuffer = [device newBufferWithBytes:queries length:sizeof(queries)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> dirBuffer = [device newBufferWithBytes:dirs length:sizeof(dirs)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> outBuffer = [device newBufferWithBytes:init.data() length:init.size() * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ScopedResidency residency(context, {sceneBuffer, queryBuffer, dirBuffer, outBuffer});
    [table setAddress:sceneBuffer.gpuAddress atIndex:0];
    [table setAddress:queryBuffer.gpuAddress atIndex:1];
    [table setAddress:dirBuffer.gpuAddress atIndex:2];
    [table setAddress:outBuffer.gpuAddress atIndex:3];
    [table setTexture:volume->texture().gpuResourceID atIndex:0];
    const bool ok = context.runSync(
        [&](id<MTL4CommandBuffer> commandBuffer) {
            id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setArgumentTable:table];
            [encoder dispatchThreads:MTLSizeMake(kQueries, 1, 1) threadsPerThreadgroup:MTLSizeMake(kQueries, 1, 1)];
            [encoder endEncoding];
        },
        nullptr, &error);
    CHECK(ok);
    if (!ok) return report("descriptor dispatch", error);

    const float* out = static_cast<const float*>(outBuffer.contents);
    for (size_t q = 0; q < kQueries; ++q) {
        const float* in = out + q * dims.n_in;
        for (uint32_t l = 0; l < dims.levels; ++l) {
            const float h = sceneGPU.voxel_min_extent * float(1u << l);
            const float expected = std::log(1.0f + scene.medium.sigma_t_max * kDensity * h);
            for (uint32_t j = 0; j < 7; ++j) CHECK_NEAR(in[l * 7 + j], expected, 5e-4);
        }
        const simd_float3 d = dirs3[q];
        CHECK_NEAR(in[35], simd_dot(d, sun), 1e-5);
        const simd_float3 up = simd_make_float3(in[36], in[37], in[38]);
        CHECK_NEAR(simd_length(up), 1.0f, 1e-5);
        CHECK_NEAR(up.z, sun.y, 1e-5);  // local z is the sun direction
        CHECK_NEAR(in[39], std::exp(-taus[q]), 1e-6);
        if (q < 2) {
            const simd_float3 ax = simd_normalize(d - simd_dot(d, sun) * sun);
            const simd_float3 ay = simd_cross(sun, ax);
            CHECK_NEAR(up.x, ax.y, 1e-5);
            CHECK_NEAR(up.y, ay.y, 1e-5);
        }
    }
}

void testNeuralRender(MetalContext& context, const std::string& dir) {
    const NeuralDims dims;
    const std::string path = dir + "/render.cnrm";
    CHECK(writeCnrm(path, validHeader(), randomWeights(dims, 13)));

    const Scene scene = makeConstantScene(32, 0.3f);
    std::string error;
    auto volume = DensityVolume::bake(context, scene, &error);
    CHECK(volume != nullptr);
    if (!volume) return report("render bake", error);

    RenderConfig config;
    config.renderer = "neural";
    config.model = path;
    config.primary_steps = 32;
    config.shadow_steps = 8;
    auto renderer = makeRenderer(context, scene, config, *volume, &error);
    CHECK(renderer != nullptr);
    if (!renderer) return report("makeRenderer neural", error);
    CHECK(std::string(renderer->name()) == "neural");
    CHECK(renderer->memoryBytes() > neuralWeightCount(dims) * sizeof(uint16_t));
    bool hasLoadMs = false;
    for (const auto& [key, value] : renderer->stats()) hasLoadMs = hasLoadMs || (key == "model_load_ms" && value >= 0.0);
    CHECK(hasLoadMs);

    Scene otherG = scene;
    otherG.medium.hg_g = 0.3f;
    std::string mismatchError;
    CHECK(makeRenderer(context, otherG, config, *volume, &mismatchError) == nullptr);
    CHECK(!mismatchError.empty());

    const uint64_t errorsBefore = context.gpuErrorCount();
    const SceneGPU sceneGPU = makeSceneGPU(scene, scene.cameras[0]);
    const bool ok = context.runSync(
        [&](id<MTL4CommandBuffer> commandBuffer) {
            id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            renderer->encode(encoder, FrameInputs{sceneGPU, context.currentSlot()});
            [encoder endEncoding];
        },
        nullptr, &error);
    CHECK(ok);
    if (!ok) return report("neural render", error);
    CHECK(context.gpuErrorCount() == errorsBefore);

    Image image;
    CHECK(readOutput(*renderer, &image, &error));
    bool valid = !image.rgb.empty();
    bool differsFromEnv = false;
    for (uint32_t y = 0; y < image.height; ++y) {
        for (uint32_t x = 0; x < image.width; ++x) {
            const float* p = image.pixel(x, y);
            for (int c = 0; c < 3; ++c) {
                valid = valid && std::isfinite(p[c]) && p[c] >= 0.0f;
                if (std::fabs(p[c] - scene.light.sky_radiance[c]) > 1e-3f &&
                    std::fabs(p[c] - scene.light.ground_radiance[c]) > 1e-3f)
                    differsFromEnv = true;
            }
        }
    }
    CHECK(valid);
    CHECK(differsFromEnv);
}

}  // namespace

int main(int argc, char** argv) {
    ensureValidationEnvironment(argc, argv);
    @autoreleasepool {
        const std::string dir = tempDir();
        testLoader(dir);

        std::string error;
        auto context = MetalContext::create(&error);
        CHECK(context != nullptr);
        if (!context) {
            report("MetalContext::create", error);
            return checkSummary();
        }
        testMlpParity(*context, dir);
        testDescriptor(*context);
        testNeuralRender(*context, dir);
        context->waitIdle();
        CHECK(context->gpuErrorCount() == 0);
        [[NSFileManager defaultManager] removeItemAtPath:[NSString stringWithUTF8String:dir.c_str()] error:nil];
    }
    return checkSummary();
}
