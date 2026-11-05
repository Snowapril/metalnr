// Neural shader tests: .cnrm v2 loader validation and weight layouts, GPU scalar and Shader ML
// evaluators vs a CPU fp32 reference, descriptor on a constant-density box, and small neural renders
// with both inference variants. Runs under the Metal debug layer and shader validation. No Python.
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

// A synthetic model: header fields plus fp32 weights in .cnrm order.
struct SyntheticModel {
    CnrmHeader header{};
    std::vector<float> weights;
};

// Random weights scaled so the hidden activations stay O(1-10) and the output pre-activation stays
// in roughly [0.5, 2] (outputs well away from 0, so relative errors are meaningful).
SyntheticModel makeSyntheticModel(uint32_t seed) {
    std::mt19937 rng(seed);
    auto uniform = [&rng](float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); };
    SyntheticModel m;
    CnrmHeader& h = m.header;
    std::memcpy(h.magic, "CNRM", 4);
    h.format_version = 2;
    h.n_in = CNR_NN_IN;
    h.n_hidden = CNR_NN_HIDDEN;
    h.n_hidden_layers = CNR_NN_HIDDEN_LAYERS;
    h.n_out = CNR_NN_OUT;
    h.hg_g = 0.8f;
    h.albedo = 0.99f;
    for (int i = 0; i < CNR_NN_RADII; ++i) h.radii[i] = kNeuralRadii[i];
    for (int i = 0; i < CNR_NN_IN - 1; ++i) {
        h.mean[i] = uniform(0.0f, 0.05f);
        h.scale[i] = uniform(5.0f, 40.0f);
    }
    h.mean[CNR_NN_IN - 1] = 0.0f;
    h.scale[CNR_NN_IN - 1] = 1.5f;
    for (float& c : h.clamp) c = 1e6f;

    m.weights.resize(CNR_NN_WEIGHT_COUNT);
    float* w = m.weights.data();
    for (int i = 0; i < CNR_NN_HIDDEN * CNR_NN_IN; ++i) w[CNR_NN_W0 + i] = uniform(-0.5f, 0.5f);
    for (int i = 0; i < CNR_NN_HIDDEN; ++i) w[CNR_NN_B0 + i] = uniform(-0.2f, 0.5f);
    for (int i = 0; i < CNR_NN_HIDDEN * CNR_NN_HIDDEN; ++i) w[CNR_NN_W1 + i] = uniform(-0.25f, 0.25f);
    for (int i = 0; i < CNR_NN_HIDDEN; ++i) w[CNR_NN_B1 + i] = uniform(-0.1f, 0.3f);
    for (int i = 0; i < CNR_NN_OUT * CNR_NN_HIDDEN; ++i) w[CNR_NN_W2 + i] = uniform(-0.05f, 0.05f);
    for (int i = 0; i < CNR_NN_OUT; ++i) w[CNR_NN_B2 + i] = uniform(1.0f, 1.5f);
    return m;
}

bool writeModel(const std::string& path, const CnrmHeader& header, const std::vector<float>& weights,
                size_t dropBytes = 0) {
    std::vector<uint8_t> bytes(sizeof(header) + weights.size() * sizeof(float));
    std::memcpy(bytes.data(), &header, sizeof(header));
    std::memcpy(bytes.data() + sizeof(header), weights.data(), weights.size() * sizeof(float));
    bytes.resize(bytes.size() - dropBytes);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(out);
}

bool writeModel(const std::string& path, const SyntheticModel& m) { return writeModel(path, m.header, m.weights); }

// CPU fp32 (double accumulation) reference of the network; clamp optional.
void cpuForward(const SyntheticModel& m, const float* desc, double* f, bool applyClamp) {
    const CnrmHeader& h = m.header;
    const float* w = m.weights.data();
    double x[CNR_NN_IN];
    for (int i = 0; i < CNR_NN_IN; ++i) x[i] = (double(desc[i]) - h.mean[i]) * h.scale[i];
    double a[CNR_NN_HIDDEN];
    for (int o = 0; o < CNR_NN_HIDDEN; ++o) {
        double acc = w[CNR_NN_B0 + o];
        for (int i = 0; i < CNR_NN_IN; ++i) acc += double(w[CNR_NN_W0 + o * CNR_NN_IN + i]) * x[i];
        const double r = std::max(acc, 0.0);
        a[o] = r * r;
    }
    double b[CNR_NN_HIDDEN];
    for (int o = 0; o < CNR_NN_HIDDEN; ++o) {
        double acc = w[CNR_NN_B1 + o];
        for (int i = 0; i < CNR_NN_HIDDEN; ++i) acc += double(w[CNR_NN_W1 + o * CNR_NN_HIDDEN + i]) * a[i];
        const double r = std::max(acc, 0.0);
        b[o] = r * r;
    }
    for (int o = 0; o < CNR_NN_OUT; ++o) {
        double acc = w[CNR_NN_B2 + o];
        for (int i = 0; i < CNR_NN_HIDDEN; ++i) acc += double(w[CNR_NN_W2 + o * CNR_NN_HIDDEN + i]) * b[i];
        const double y = acc * acc;
        f[o] = applyClamp ? std::min(y, double(h.clamp[o])) : y;
    }
}

Scene makeConstantScene(simd_float3 boundsMin, simd_float3 boundsMax, uint32_t gridSide, float constantDensity) {
    Scene scene;
    scene.name = "neural_test";
    CameraDesc camera;
    camera.name = "front";
    camera.position = simd_make_float3(0, 400, -2000);
    camera.target = simd_make_float3(0, 400, 0);
    camera.vfov_deg = 40.0f;
    scene.cameras.push_back(camera);
    scene.width = 32;
    scene.height = 18;
    scene.volume.bounds_min = boundsMin;
    scene.volume.bounds_max = boundsMax;
    scene.volume.grid = simd_make_uint3(gridSide, gridSide, gridSide);
    scene.volume.density = "constant";
    scene.volume.constant_density = constantDensity;
    scene.medium.sigma_t_max = 0.04f;
    scene.medium.albedo = 0.99f;
    scene.medium.hg_g = 0.8f;
    scene.light.sun_direction = simd_normalize(simd_make_float3(0.3f, 0.6f, 0.74f));
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
    CHECK(neuralWeightCount(NeuralDims{}) == CNR_NN_WEIGHT_COUNT);
    CHECK(neuralWeightCount(NeuralDims{}) == 547);  // 13*16+16 + 16*16+16 + 16*3+3

    const SyntheticModel m = makeSyntheticModel(7);
    std::string error;
    const std::string good = dir + "/good.cnrm";
    CHECK(writeModel(good, m));
    std::optional<NeuralModel> model = loadNeuralModel(good, &error);
    CHECK(model.has_value());
    if (!model) return report("load good model", error);
    CHECK(model->format_version == 2);
    CHECK(model->weights == m.weights);
    CHECK(model->dims.n_in == 13 && model->dims.n_hidden == 16 && model->dims.n_hidden_layers == 2 &&
          model->dims.n_out == 3);
    CHECK(model->sha256.size() == 64);
    CHECK(!model->has_train_seed_range);
    CHECK(model->mean[3] == m.header.mean[3] && model->scale[12] == m.header.scale[12]);
    CHECK(model->clamp[1] == m.header.clamp[1]);

    // Shader ML layout: [K][N] transposes with zero padding (K 13 -> 16, N 3 -> 4).
    CHECK(model->weights_ml.size() == CNR_NN_ML_WEIGHT_COUNT);
    bool layoutOk = model->weights_ml.size() == CNR_NN_ML_WEIGHT_COUNT;
    for (int k = 0; layoutOk && k < CNR_NN_IN_PAD; ++k) {
        for (int n = 0; n < CNR_NN_HIDDEN; ++n) {
            const float v = halfBitsToFloat(model->weights_ml[CNR_NN_ML_W0 + k * CNR_NN_HIDDEN + n]);
            const float e = k < CNR_NN_IN ? float(static_cast<__fp16>(m.weights[CNR_NN_W0 + n * CNR_NN_IN + k])) : 0.0f;
            layoutOk = layoutOk && v == e;
        }
    }
    for (int k = 0; layoutOk && k < CNR_NN_HIDDEN; ++k) {
        for (int n = 0; n < CNR_NN_HIDDEN; ++n) {
            const float v = halfBitsToFloat(model->weights_ml[CNR_NN_ML_W1 + k * CNR_NN_HIDDEN + n]);
            layoutOk = layoutOk && v == float(static_cast<__fp16>(m.weights[CNR_NN_W1 + n * CNR_NN_HIDDEN + k]));
        }
        for (int n = 0; n < CNR_NN_OUT_PAD; ++n) {
            const float v = halfBitsToFloat(model->weights_ml[CNR_NN_ML_W2 + k * CNR_NN_OUT_PAD + n]);
            const float e =
                n < CNR_NN_OUT ? float(static_cast<__fp16>(m.weights[CNR_NN_W2 + n * CNR_NN_HIDDEN + k])) : 0.0f;
            layoutOk = layoutOk && v == e;
        }
    }
    CHECK(layoutOk);

    const NeuralStandardization norm = neuralStandardization(*model);
    CHECK(norm.mean[5] == m.header.mean[5] && norm.scale[0] == m.header.scale[0]);
    CHECK(norm.mean[CNR_NN_IN] == 0.0f && norm.clamp[CNR_NN_OUT] == 0.0f);

    Scene scene = makeConstantScene(simd_make_float3(-500, 0, -500), simd_make_float3(500, 800, 500), 16, 1.0f);
    CHECK(validateNeuralModelForScene(*model, scene, &error));
    Scene otherG = scene;
    otherG.medium.hg_g = 0.5f;
    std::string gError;
    CHECK(!validateNeuralModelForScene(*model, otherG, &gError));
    CHECK(!gError.empty());
    Scene otherAlbedo = scene;
    otherAlbedo.medium.albedo = 0.9f;
    CHECK(!validateNeuralModelForScene(*model, otherAlbedo, &error));

    auto rejects = [&](const char* name, const CnrmHeader& header, const std::vector<float>& w, size_t drop) {
        const std::string path = dir + "/" + name + ".cnrm";
        CHECK(writeModel(path, header, w, drop));
        std::string e;
        const bool rejected = !loadNeuralModel(path, &e).has_value();
        CHECK(rejected);
        CHECK(!e.empty());
        if (!rejected) std::fprintf(stderr, "loader accepted invalid model '%s'\n", name);
    };
    CnrmHeader badMagic = m.header;
    badMagic.magic[0] = 'X';
    rejects("bad_magic", badMagic, m.weights, 0);
    CnrmHeader badVersion = m.header;
    badVersion.format_version = 1;
    rejects("bad_version", badVersion, m.weights, 0);
    // Self-consistent file for a different architecture (13 -> 32 -> 32 -> 3): only the dimension check rejects it.
    CnrmHeader badDims = m.header;
    badDims.n_hidden = 32;
    rejects("bad_dims", badDims, std::vector<float>(13 * 32 + 32 + 32 * 32 + 32 + 32 * 3 + 3, 0.1f), 0);
    CnrmHeader badLayers = m.header;
    badLayers.n_hidden_layers = 3;
    rejects("bad_layers", badLayers, m.weights, 0);
    rejects("truncated", m.header, m.weights, 4);
    std::vector<float> extra = m.weights;
    extra.push_back(0.0f);
    rejects("extra_bytes", m.header, extra, 0);
    CnrmHeader badRadii = m.header;
    badRadii.radii[2] = 20.0f;
    rejects("bad_radii", badRadii, m.weights, 0);
    CnrmHeader badMean = m.header;
    badMean.mean[4] = NAN;
    rejects("nan_mean", badMean, m.weights, 0);
    CnrmHeader badScale = m.header;
    badScale.scale[0] = INFINITY;
    rejects("inf_scale", badScale, m.weights, 0);
    CnrmHeader badClamp = m.header;
    badClamp.clamp[2] = 0.0f;
    rejects("zero_clamp", badClamp, m.weights, 0);
    std::vector<float> nanWeights = m.weights;
    nanWeights[CNR_NN_W1 + 3] = NAN;
    rejects("nan_weight", m.header, nanWeights, 0);
    std::string missingError;
    CHECK(!loadNeuralModel(dir + "/does_not_exist.cnrm", &missingError).has_value());
}

void testMlpParity(MetalContext& context, const std::string& dir) {
    constexpr size_t kCount = 512;
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> sigma(0.0f, 0.1f);
    std::uniform_real_distribution<float> cosine(-1.0f, 1.0f);
    std::vector<float> inputs(kCount * CNR_NN_IN);
    for (size_t i = 0; i < kCount; ++i) {
        for (int k = 0; k < CNR_NN_IN - 1; ++k) inputs[i * CNR_NN_IN + k] = sigma(rng);
        inputs[i * CNR_NN_IN + CNR_NN_IN - 1] = cosine(rng);
    }

    // Clamp output 2 at the median of its unclamped values so the clamp is exercised.
    SyntheticModel m = makeSyntheticModel(11);
    std::vector<double> y2(kCount);
    for (size_t i = 0; i < kCount; ++i) {
        double f[CNR_NN_OUT];
        cpuForward(m, inputs.data() + i * CNR_NN_IN, f, false);
        y2[i] = f[2];
    }
    std::nth_element(y2.begin(), y2.begin() + kCount / 2, y2.end());
    m.header.clamp[2] = float(y2[kCount / 2]);

    const std::string path = dir + "/parity.cnrm";
    CHECK(writeModel(path, m));
    std::string error;
    std::optional<NeuralModel> model = loadNeuralModel(path, &error);
    CHECK(model.has_value());
    if (!model) return report("load parity model", error);

    std::vector<double> expected(kCount * CNR_NN_OUT);
    size_t clamped = 0;
    double minExpected = 1e30;
    for (size_t i = 0; i < kCount; ++i) {
        cpuForward(m, inputs.data() + i * CNR_NN_IN, expected.data() + i * CNR_NN_OUT, true);
        if (expected[i * CNR_NN_OUT + 2] == double(m.header.clamp[2])) ++clamped;
        for (int c = 0; c < CNR_NN_OUT; ++c) minExpected = std::min(minExpected, expected[i * CNR_NN_OUT + c]);
    }
    CHECK(clamped > 0);
    CHECK(minExpected > 1e-2);  // outputs well away from 0: the comparison is meaningful

    const std::pair<NeuralInference, double> variants[] = {{NeuralInference::Scalar, 1e-4},
                                                           {NeuralInference::ShaderML, 2e-2}};
    for (const auto& [inference, tolerance] : variants) {
        std::vector<float> outputs;
        const bool ok = evalNeuralBatch(context, *model, inputs, &outputs, inference, &error);
        CHECK(ok);
        if (!ok) {
            report(neuralInferenceName(inference), error);
            continue;
        }
        CHECK(outputs.size() == kCount * CNR_NN_OUT);
        if (outputs.size() != kCount * CNR_NN_OUT) continue;
        double maxRel = 0.0;
        bool finite = true;
        for (size_t i = 0; i < expected.size(); ++i) {
            finite = finite && std::isfinite(outputs[i]);
            maxRel = std::max(maxRel, std::fabs(outputs[i] - expected[i]) / (std::fabs(expected[i]) + 1e-3));
        }
        CHECK(finite);
        CHECK(maxRel <= tolerance);
        std::fprintf(stderr, "mlp parity %s: max rel error %.3g (tolerance %.3g)\n", neuralInferenceName(inference),
                     maxRel, tolerance);
    }
}

void testDescriptor(MetalContext& context) {
    // 1000 m cube, 50^3 voxels of 20 m, density 1: every mip level is exactly 1 inside the box.
    const Scene scene = makeConstantScene(simd_make_float3(0, 0, 0), simd_make_float3(1000, 1000, 1000), 50, 1.0f);
    std::string error;
    auto volume = DensityVolume::bake(context, scene, &error);
    CHECK(volume != nullptr);
    if (!volume) return report("descriptor bake", error);
    const SceneGPU sceneGPU = makeSceneGPU(scene, scene.cameras[0]);
    const NeuralDims dims;

    id<MTLComputePipelineState> pipeline = context.makeComputePipeline(
        neuralShaderFiles("tests/shaders/test_neural.metal"), "test_build_descriptor",
        neuralShaderDefines(dims, NeuralInference::Scalar), &error);
    id<MTL4ArgumentTable> table = context.makeArgumentTable(4, 1, &error);
    CHECK(pipeline != nil && table != nil);
    if (!pipeline || !table) return report("descriptor setup", error);

    const simd_float3 sun = sceneGPU.sun_direction;
    // Query 0: centre; every sample (<= 192 m toward the sun) stays inside the box.
    // Query 1: 100 m below the top; the 192 m sample toward the sun leaves the box (y > 1000).
    const simd_float3 points3[] = {simd_make_float3(500, 500, 500), simd_make_float3(500, 900, 500)};
    const simd_float3 dirs3[] = {simd_normalize(simd_make_float3(1.0f, 0.2f, 0.3f)),
                                 simd_normalize(simd_make_float3(-0.4f, -0.8f, 0.1f))};
    constexpr size_t kQueries = 2;
    simd_float4 points[kQueries];
    simd_float4 dirs[kQueries];
    for (size_t i = 0; i < kQueries; ++i) {
        points[i] = simd_make_float4(points3[i], 1.0f);
        dirs[i] = simd_make_float4(dirs3[i], 0.0f);
    }
    std::vector<float> init(kQueries * CNR_NN_IN, -1.0f);
    id<MTLDevice> device = context.device();
    id<MTLBuffer> sceneBuffer = [device newBufferWithBytes:&sceneGPU length:sizeof(sceneGPU)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> pointBuffer = [device newBufferWithBytes:points length:sizeof(points)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> dirBuffer = [device newBufferWithBytes:dirs length:sizeof(dirs)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> outBuffer = [device newBufferWithBytes:init.data() length:init.size() * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ScopedResidency residency(context, {sceneBuffer, pointBuffer, dirBuffer, outBuffer});
    [table setAddress:sceneBuffer.gpuAddress atIndex:0];
    [table setAddress:pointBuffer.gpuAddress atIndex:1];
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

    const float sigma = scene.medium.sigma_t_max;
    const float* out = static_cast<const float*>(outBuffer.contents);
    for (size_t q = 0; q < kQueries; ++q) {
        const float* in = out + q * CNR_NN_IN;
        for (int r = 0; r < CNR_NN_RADII; ++r) CHECK_NEAR(in[r], sigma * 1.0f, 1e-6);  // local
        for (int r = 0; r < CNR_NN_RADII; ++r) {
            const simd_float3 p = points3[q] + kNeuralRadii[r] * sun;
            const bool inside = p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x <= 1000 && p.y <= 1000 && p.z <= 1000;
            CHECK_NEAR(in[CNR_NN_RADII + r], inside ? sigma : 0.0f, 1e-6);  // toward the sun
        }
        CHECK_NEAR(in[2 * CNR_NN_RADII], simd_dot(dirs3[q], sun), 1e-6);
    }
    // The geometry above must produce both cases.
    CHECK(points3[1].y + kNeuralRadii[CNR_NN_RADII - 1] * sun.y > 1000.0f);
    CHECK(points3[1].y + kNeuralRadii[CNR_NN_RADII - 2] * sun.y < 1000.0f);
    CHECK(out[CNR_NN_IN + 2 * CNR_NN_RADII - 1] == 0.0f);
}

bool renderOnce(MetalContext& context, Renderer& renderer, const Scene& scene, std::vector<float>* rgba,
                std::string* error) {
    const SceneGPU sceneGPU = makeSceneGPU(scene, scene.cameras[0]);
    const bool ok = context.runSync(
        [&](id<MTL4CommandBuffer> commandBuffer) {
            id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            renderer.encode(encoder, FrameInputs{sceneGPU, context.currentSlot()});
            [encoder endEncoding];
        },
        nullptr, error);
    if (!ok) return false;
    id<MTLTexture> texture = renderer.output();
    rgba->assign(size_t(texture.width) * texture.height * 4, 0.0f);
    [texture getBytes:rgba->data()
          bytesPerRow:texture.width * 4 * sizeof(float)
           fromRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
          mipmapLevel:0];
    return true;
}

void testNeuralRender(MetalContext& context, const std::string& dir) {
    const SyntheticModel m = makeSyntheticModel(13);
    const std::string path = dir + "/render.cnrm";
    CHECK(writeModel(path, m));

    const Scene scene = makeConstantScene(simd_make_float3(-500, 0, -500), simd_make_float3(500, 800, 500), 32, 0.3f);
    std::string error;
    auto volume = DensityVolume::bake(context, scene, &error);
    CHECK(volume != nullptr);
    if (!volume) return report("render bake", error);

    RenderConfig config;
    config.renderer = "neural";
    config.model = path;
    config.step_scale = 1.0f;
    config.max_steps = 256;
    config.min_transmittance = 0.0f;
    config.dither = false;
    config.empty_skip = false;

    // ml_encoder belongs to a separate module; an unknown variant is an error.
    RenderConfig encoderConfig = config;
    encoderConfig.inference = "ml_encoder";
    std::string encoderError;
    CHECK(makeNeuralRenderer(context, scene, encoderConfig, *volume, &encoderError) == nullptr);
    CHECK(encoderError == "ml_encoder inference is provided by the ML encoder module");
    RenderConfig unknownConfig = config;
    unknownConfig.inference = "magic";
    std::string unknownError;
    CHECK(makeRenderer(context, scene, unknownConfig, *volume, &unknownError) == nullptr);
    CHECK(!unknownError.empty());

    Scene otherG = scene;
    otherG.medium.hg_g = 0.3f;
    std::string mismatchError;
    CHECK(makeRenderer(context, otherG, config, *volume, &mismatchError) == nullptr);
    CHECK(!mismatchError.empty());

    std::vector<float> images[2];
    const char* variants[2] = {"scalar", "shader_ml"};
    for (int v = 0; v < 2; ++v) {
        RenderConfig c = config;
        c.inference = variants[v];
        auto renderer = makeRenderer(context, scene, c, *volume, &error);
        CHECK(renderer != nullptr);
        if (!renderer) {
            report(variants[v], error);
            continue;
        }
        CHECK(std::string(renderer->name()) == "neural");
        CHECK(renderer->memoryBytes() > CNR_NN_WEIGHT_COUNT * sizeof(float));
        bool hasLoadMs = false;
        for (const auto& [key, value] : renderer->stats())
            hasLoadMs = hasLoadMs || (key == "model_load_ms" && value >= 0.0);
        CHECK(hasLoadMs);

        const uint64_t errorsBefore = context.gpuErrorCount();
        const bool ok = renderOnce(context, *renderer, scene, &images[v], &error);
        CHECK(ok);
        if (!ok) {
            report(variants[v], error);
            continue;
        }
        CHECK(context.gpuErrorCount() == errorsBefore);

        bool valid = !images[v].empty();
        bool differsFromEnv = false;
        bool hasCloud = false;
        for (size_t i = 0; i < images[v].size() / 4; ++i) {
            const float* p = &images[v][i * 4];
            for (int c3 = 0; c3 < 3; ++c3) {
                valid = valid && std::isfinite(p[c3]) && p[c3] >= 0.0f;
                if (std::fabs(p[c3] - scene.light.sky_radiance[c3]) > 1e-3f &&
                    std::fabs(p[c3] - scene.light.ground_radiance[c3]) > 1e-3f)
                    differsFromEnv = true;
            }
            valid = valid && p[3] >= 0.0f && p[3] <= 1.0f;
            hasCloud = hasCloud || p[3] > 0.5f;
        }
        CHECK(valid);
        CHECK(differsFromEnv);
        CHECK(hasCloud);
    }

    // Both variants evaluate the same network: the images agree up to fp16 inference error.
    if (images[0].size() == images[1].size() && !images[0].empty()) {
        double sumRel = 0.0;
        size_t n = 0;
        for (size_t i = 0; i < images[0].size(); ++i) {
            if (i % 4 == 3) continue;
            sumRel += std::fabs(double(images[0][i]) - images[1][i]) / (std::fabs(double(images[0][i])) + 1e-3);
            ++n;
        }
        const double meanRel = sumRel / double(n);
        CHECK(meanRel < 1e-2);
        std::fprintf(stderr, "render scalar vs shader_ml: mean rel difference %.3g\n", meanRel);
    }
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
