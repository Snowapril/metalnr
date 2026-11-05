// Analytic renderer tests (milestone: analytic baseline): closed-form checks in a homogeneous box
// (Beer–Lambert transmittance, single scattering, HG normalization) and a sanity check of the
// multiple-scattering approximation. Runs under the Metal debug layer and shader validation.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "check.h"

#include "density_volume.h"
#include "image_io.h"
#include "metal_context.h"
#include "metrics.h"
#include "renderer.h"
#include "scene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

const std::vector<std::string> kTestKernels = {"tests/shaders/test_analytic.metal"};

void report(const char* what, const std::string& error) {
    std::fprintf(stderr, "%s: %s\n", what, error.c_str());
}

// Homogeneous unit-density box [-1,1]^3 with sigma_t = 1, viewed from (0,0,-5).
Scene makeHomogeneousScene(float albedo, simd_float3 sunDirection) {
    Scene scene;
    scene.name = "analytic_homogeneous";
    CameraDesc camera;
    camera.name = "front";
    camera.position = simd_make_float3(0, 0, -5);
    camera.target = simd_make_float3(0, 0, 0);
    camera.vfov_deg = 30.0f;
    scene.cameras.push_back(camera);
    scene.width = 64;
    scene.height = 64;
    scene.volume.bounds_min = simd_make_float3(-1, -1, -1);
    scene.volume.bounds_max = simd_make_float3(1, 1, 1);
    scene.volume.grid = simd_make_uint3(16, 16, 16);
    scene.volume.density = "constant";
    scene.volume.constant_density = 1.0f;
    scene.medium.sigma_t_max = 1.0f;
    scene.medium.albedo = albedo;
    scene.medium.hg_g = 0.3f;
    scene.light.sun_direction = simd_normalize(sunDirection);
    scene.light.sun_irradiance = simd_make_float3(3.0f, 2.5f, 2.0f);
    scene.light.sky_radiance = simd_make_float3(0.2f, 0.4f, 0.8f);
    scene.light.ground_radiance = simd_make_float3(0.1f, 0.1f, 0.1f);
    return scene;
}

RenderConfig analyticConfig(uint32_t primarySteps, uint32_t shadowSteps, uint32_t octaves, bool ambient) {
    RenderConfig config;
    config.renderer = "analytic";
    config.primary_steps = primarySteps;
    config.shadow_steps = shadowSteps;
    config.octaves = octaves;
    config.ambient = ambient;
    return config;
}

// Adds allocations to the context residency set for the lifetime of this object.
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

id<MTLBuffer> sharedBuffer(MetalContext& context, const void* bytes, size_t length) {
    return [context.device() newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
}

bool dispatch(MetalContext& context, id<MTLComputePipelineState> pipeline, id<MTL4ArgumentTable> table,
              MTLSize threads, std::string* error) {
    return context.runSync(
        [&](id<MTL4CommandBuffer> commandBuffer) {
            id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setArgumentTable:table];
            [encoder dispatchThreads:threads threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [encoder endEncoding];
        },
        nullptr, error);
}

bool renderImage(MetalContext& context, const Scene& scene, const RenderConfig& config,
                 const DensityVolume& volume, Image* image) {
    std::string error;
    auto renderer = makeRenderer(context, scene, config, volume, &error);
    CHECK(renderer != nullptr);
    if (!renderer) {
        report("makeRenderer(analytic)", error);
        return false;
    }
    CHECK(std::string(renderer->name()) == "analytic");
    CHECK(renderer->memoryBytes() > 0);
    const SceneGPU sceneGPU = makeSceneGPU(scene, scene.cameras[0]);
    const bool ok = context.runSync(
        [&](id<MTL4CommandBuffer> commandBuffer) {
            id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            renderer->encode(encoder, FrameInputs{sceneGPU, context.currentSlot()});
            [encoder endEncoding];
        },
        nullptr, &error);
    CHECK(ok);
    if (!ok) {
        report("analytic render", error);
        return false;
    }
    const bool read = readOutput(*renderer, image, &error);
    CHECK(read);
    if (!read) report("readOutput", error);
    return read && image->width == scene.width && image->height == scene.height;
}

// ---- CPU references (double precision) ----

simd_double3 toDouble(simd_float3 v) { return simd_make_double3(v.x, v.y, v.z); }

// Slab test of the ray o + t d, t >= 0, against [bmin, bmax]. Returns false on a miss.
bool slab(simd_double3 o, simd_double3 d, simd_double3 bmin, simd_double3 bmax, double* t0, double* t1) {
    double lo = 0.0;
    double hi = INFINITY;
    for (int a = 0; a < 3; ++a) {
        if (d[a] == 0.0) {
            if (o[a] < bmin[a] || o[a] > bmax[a]) return false;
            continue;
        }
        double ta = (bmin[a] - o[a]) / d[a];
        double tb = (bmax[a] - o[a]) / d[a];
        lo = std::max(lo, std::min(ta, tb));
        hi = std::min(hi, std::max(ta, tb));
    }
    *t0 = lo;
    *t1 = hi;
    return hi > lo;
}

double chord(simd_double3 o, simd_double3 d, simd_double3 bmin, simd_double3 bmax) {
    double t0, t1;
    return slab(o, d, bmin, bmax, &t0, &t1) ? t1 - t0 : 0.0;
}

double hgCpu(double cosTheta, double g) {
    const double denom = 1.0 + g * g - 2.0 * g * cosTheta;
    return (1.0 - g * g) / (4.0 * M_PI * denom * std::sqrt(denom));
}

// camera_ray() of the contract, evaluated in double from the same SceneGPU constants.
simd_double3 cameraDirection(const SceneGPU& s, double px, double py) {
    const double nx = (2.0 * px / double(s.width) - 1.0) * s.tan_half_vfov * s.aspect;
    const double ny = (1.0 - 2.0 * py / double(s.height)) * s.tan_half_vfov;
    return simd_normalize(toDouble(s.cam_forward) + nx * toDouble(s.cam_right) + ny * toDouble(s.cam_up));
}

simd_double3 envCpu(const SceneGPU& s, simd_double3 d) {
    return d.y > 0.0 ? toDouble(s.sky_radiance) : toDouble(s.ground_radiance);
}

// ---- a. Beer–Lambert ----

void testOpticalDepth(MetalContext& context, const Scene& scene, const DensityVolume& volume) {
    const SceneGPU sceneGPU = makeSceneGPU(scene, scene.cameras[0]);
    const simd_double3 bmin = toDouble(scene.volume.bounds_min);
    const simd_double3 bmax = toDouble(scene.volume.bounds_max);
    const double sigma = scene.medium.sigma_t_max * scene.volume.constant_density;

    struct TestRay {
        simd_float3 o;
        simd_float3 d;
        uint32_t steps;
    };
    const TestRay tests[] = {
        {simd_make_float3(-3.0f, 0.1f, 0.2f), simd_make_float3(1, 0, 0), 4},                    // axis, outside
        {simd_make_float3(0, 0, 0), simd_normalize(simd_make_float3(1, 1, 1)), 8},             // diagonal, inside
        {simd_make_float3(0.3f, -0.2f, 0.5f), simd_normalize(simd_make_float3(0.3f, 1, 0.2f)), 16},  // oblique, inside
        {simd_make_float3(2.5f, 1.7f, -3.1f), simd_normalize(simd_make_float3(-2.4f, -1.5f, 3.2f)), 32},  // oblique, outside
        {simd_make_float3(-0.9f, 0.8f, -0.7f), simd_normalize(simd_make_float3(0.6f, -0.7f, 0.4f)), 3},  // oblique, inside
        {simd_make_float3(0.2f, -4.0f, 0.1f), simd_normalize(simd_make_float3(0.05f, 1, -0.1f)), 64},  // steep, outside
        {simd_make_float3(-3.0f, 0.0f, 0.0f), simd_make_float3(-1, 0, 0), 8},                   // pointing away: 0
        {simd_make_float3(-3.0f, 2.0f, 0.0f), simd_normalize(simd_make_float3(1, 0.2f, 0)), 8},  // passes above: 0
    };
    constexpr NSUInteger kRays = sizeof(tests) / sizeof(tests[0]);

    std::vector<simd_float4> rays;
    for (const TestRay& t : tests) {
        rays.push_back(simd_make_float4(t.o, float(t.steps)));
        rays.push_back(simd_make_float4(t.d, 0.0f));
    }
    std::vector<float> init(kRays, -1.0f);

    std::string error;
    id<MTLComputePipelineState> pipeline =
        context.makeComputePipeline(kTestKernels, "test_optical_depth", "", &error);
    id<MTL4ArgumentTable> table = context.makeArgumentTable(3, 1, &error);
    CHECK(pipeline != nil && table != nil);
    if (!pipeline || !table) return report("optical depth setup", error);

    id<MTLBuffer> sceneBuffer = sharedBuffer(context, &sceneGPU, sizeof(sceneGPU));
    id<MTLBuffer> rayBuffer = sharedBuffer(context, rays.data(), rays.size() * sizeof(simd_float4));
    id<MTLBuffer> outBuffer = sharedBuffer(context, init.data(), init.size() * sizeof(float));
    ScopedResidency residency(context, {sceneBuffer, rayBuffer, outBuffer});
    [table setAddress:sceneBuffer.gpuAddress atIndex:0];
    [table setAddress:rayBuffer.gpuAddress atIndex:1];
    [table setAddress:outBuffer.gpuAddress atIndex:2];
    [table setTexture:volume.texture().gpuResourceID atIndex:0];
    const bool ok = dispatch(context, pipeline, table, MTLSizeMake(kRays, 1, 1), &error);
    CHECK(ok);
    if (!ok) return report("optical depth dispatch", error);

    const float* out = static_cast<const float*>(outBuffer.contents);
    for (NSUInteger i = 0; i < kRays; ++i) {
        const double expected = sigma * chord(toDouble(tests[i].o), toDouble(tests[i].d), bmin, bmax);
        if (expected == 0.0) {
            CHECK(out[i] == 0.0f);
        } else {
            CHECK_NEAR(out[i] / expected, 1.0, 1e-4);
        }
    }
}

void testBeerLambertImage(MetalContext& context, const Scene& scene, const DensityVolume& volume) {
    Image image;
    if (!renderImage(context, scene, analyticConfig(64, 8, 1, false), volume, &image)) return;

    const SceneGPU s = makeSceneGPU(scene, scene.cameras[0]);
    const simd_double3 o = toDouble(s.cam_position);
    const simd_double3 bmin = toDouble(scene.volume.bounds_min);
    const simd_double3 bmax = toDouble(scene.volume.bounds_max);
    const double sigma = scene.medium.sigma_t_max * scene.volume.constant_density;

    double maxRel = 0.0;
    uint32_t hits = 0;
    for (uint32_t y = 0; y < image.height; ++y) {
        for (uint32_t x = 0; x < image.width; ++x) {
            const simd_double3 d = cameraDirection(s, x + 0.5, y + 0.5);
            const double len = chord(o, d, bmin, bmax);
            if (len > 0.0) ++hits;
            const simd_double3 expected = envCpu(s, d) * std::exp(-sigma * len);
            const float* p = image.pixel(x, y);
            for (int c = 0; c < 3; ++c) maxRel = std::max(maxRel, std::fabs(p[c] - expected[c]) / expected[c]);
        }
    }
    CHECK(hits > image.width * image.height / 8);  // the box covers a good part of the image
    CHECK(maxRel < 1e-4);
    if (!(maxRel < 1e-4)) std::fprintf(stderr, "Beer-Lambert image: max relative error %.3g\n", maxRel);
}

// ---- b. Single scattering along exact rays ----

void testSingleScatterRays(MetalContext& context, const Scene& scene, const DensityVolume& volume) {
    const SceneGPU s = makeSceneGPU(scene, scene.cameras[0]);
    const double sigma = scene.medium.sigma_t_max * scene.volume.constant_density;
    const double albedo = scene.medium.albedo;
    const double phase = hgCpu(0.0, scene.medium.hg_g);  // sun (0,1,0) is perpendicular to d = (1,0,0)
    const float y0s[] = {-0.5f, 0.0f, 0.6f};
    constexpr NSUInteger kRays = sizeof(y0s) / sizeof(y0s[0]);

    std::vector<simd_float4> rays;
    for (float y0 : y0s) {
        rays.push_back(simd_make_float4(-3.0f, y0, 0.2f, 16.0f));  // primary_steps 16
        rays.push_back(simd_make_float4(1.0f, 0.0f, 0.0f, 8.0f));  // shadow_steps 8
    }
    const std::vector<simd_float4> init(kRays, simd_make_float4(-1, -1, -1, -1));

    std::string error;
    id<MTLComputePipelineState> pipeline =
        context.makeComputePipeline(kTestKernels, "test_march_single_scatter", "", &error);
    id<MTL4ArgumentTable> table = context.makeArgumentTable(3, 1, &error);
    CHECK(pipeline != nil && table != nil);
    if (!pipeline || !table) return report("single scatter setup", error);

    id<MTLBuffer> sceneBuffer = sharedBuffer(context, &s, sizeof(s));
    id<MTLBuffer> rayBuffer = sharedBuffer(context, rays.data(), rays.size() * sizeof(simd_float4));
    id<MTLBuffer> outBuffer = sharedBuffer(context, init.data(), init.size() * sizeof(simd_float4));
    ScopedResidency residency(context, {sceneBuffer, rayBuffer, outBuffer});
    [table setAddress:sceneBuffer.gpuAddress atIndex:0];
    [table setAddress:rayBuffer.gpuAddress atIndex:1];
    [table setAddress:outBuffer.gpuAddress atIndex:2];
    [table setTexture:volume.texture().gpuResourceID atIndex:0];
    const bool ok = dispatch(context, pipeline, table, MTLSizeMake(kRays, 1, 1), &error);
    CHECK(ok);
    if (!ok) return report("single scatter dispatch", error);

    const simd_float4* out = static_cast<const simd_float4*>(outBuffer.contents);
    const simd_double3 sun = toDouble(s.sun_irradiance);
    const simd_double3 env = envCpu(s, simd_make_double3(1, 0, 0));  // d.y == 0 -> ground
    const double chordView = 2.0;
    for (NSUInteger i = 0; i < kRays; ++i) {
        const double tauSun = sigma * (1.0 - double(y0s[i]));
        const simd_double3 expected = albedo * sun * phase * std::exp(-tauSun) * (1.0 - std::exp(-sigma * chordView)) +
                                      std::exp(-sigma * chordView) * env;
        for (int c = 0; c < 3; ++c) CHECK_NEAR(out[i][c] / expected[c], 1.0, 1e-4);
    }
}

// ---- c. Single scattering image with a tilted sun ----

void testSingleScatterImage(MetalContext& context, const Scene& scene, const DensityVolume& volume) {
    Image image;
    if (!renderImage(context, scene, analyticConfig(1024, 64, 1, false), volume, &image)) return;

    const SceneGPU s = makeSceneGPU(scene, scene.cameras[0]);
    const simd_double3 o = toDouble(s.cam_position);
    const simd_double3 bmin = toDouble(scene.volume.bounds_min);
    const simd_double3 bmax = toDouble(scene.volume.bounds_max);
    const simd_double3 sunDir = toDouble(s.sun_direction);
    const simd_double3 sunE = toDouble(s.sun_irradiance);
    const double sigma = scene.medium.sigma_t_max * scene.volume.constant_density;
    const double albedo = scene.medium.albedo;
    const double g = scene.medium.hg_g;
    constexpr int kQuadrature = 20000;

    Image reference;
    reference.width = image.width;
    reference.height = image.height;
    reference.rgb.resize(image.rgb.size());
    for (uint32_t y = 0; y < image.height; ++y) {
        for (uint32_t x = 0; x < image.width; ++x) {
            const simd_double3 d = cameraDirection(s, x + 0.5, y + 0.5);
            const simd_double3 env = envCpu(s, d);
            double t0, t1;
            simd_double3 L = env;
            if (slab(o, d, bmin, bmax, &t0, &t1)) {
                // L = int_t0^t1 e^{-sigma (t - t0)} sigma albedo p E e^{-sigma s_sun(t)} dt + e^{-sigma len} L_env
                const double phase = hgCpu(simd_dot(sunDir, d), g);
                const double dt = (t1 - t0) / kQuadrature;
                double sum = 0.0;
                for (int i = 0; i < kQuadrature; ++i) {
                    const double t = t0 + (i + 0.5) * dt;
                    const simd_double3 p = o + t * d;
                    double u0, u1;
                    const double sSun = slab(p, sunDir, bmin, bmax, &u0, &u1) ? u1 - u0 : 0.0;
                    sum += std::exp(-sigma * (t - t0) - sigma * sSun);
                }
                L = sum * dt * sigma * albedo * phase * sunE + std::exp(-sigma * (t1 - t0)) * env;
            }
            float* r = reference.pixel(x, y);
            for (int c = 0; c < 3; ++c) r[c] = float(L[c]);
        }
    }

    const double error = relMSE(image, reference, 0.0f);
    CHECK(error < 1e-4);
    if (!(error < 1e-4)) std::fprintf(stderr, "single scatter image: relMSE %.3g\n", error);
}

// ---- d. Henyey-Greenstein normalization and mean cosine ----

void testHgIntegral(MetalContext& context) {
    const float gs[] = {-0.5f, 0.0f, 0.3f, 0.8f};
    constexpr NSUInteger kG = sizeof(gs) / sizeof(gs[0]);
    constexpr NSUInteger kThreads = 4096;  // x 64 samples per thread
    constexpr double kSamples = double(kThreads) * 64.0;

    std::string error;
    id<MTLComputePipelineState> pipeline = context.makeComputePipeline(kTestKernels, "test_hg_integral", "", &error);
    id<MTL4ArgumentTable> table = context.makeArgumentTable(2, 1, &error);
    CHECK(pipeline != nil && table != nil);
    if (!pipeline || !table) return report("hg integral setup", error);

    const std::vector<simd_float2> init(kThreads * kG, simd_make_float2(-1, -1));
    id<MTLBuffer> gBuffer = sharedBuffer(context, gs, sizeof(gs));
    id<MTLBuffer> outBuffer = sharedBuffer(context, init.data(), init.size() * sizeof(simd_float2));
    ScopedResidency residency(context, {gBuffer, outBuffer});
    [table setAddress:gBuffer.gpuAddress atIndex:0];
    [table setAddress:outBuffer.gpuAddress atIndex:1];
    const bool ok = dispatch(context, pipeline, table, MTLSizeMake(kThreads, kG, 1), &error);
    CHECK(ok);
    if (!ok) return report("hg integral dispatch", error);

    const simd_float2* out = static_cast<const simd_float2*>(outBuffer.contents);
    const double scale = 2.0 * M_PI * (2.0 / kSamples);  // 2 pi * dcos
    for (NSUInteger gi = 0; gi < kG; ++gi) {
        double sum = 0.0;
        double sumCos = 0.0;
        for (NSUInteger i = 0; i < kThreads; ++i) {
            sum += out[gi * kThreads + i].x;
            sumCos += out[gi * kThreads + i].y;
        }
        CHECK_NEAR(sum * scale, 1.0, 1e-4);
        CHECK_NEAR(sumCos * scale, gs[gi], 1e-4);
    }
}

// ---- e. Multiple-scattering approximation sanity on eval_medium ----

void testMultipleScatteringSanity(MetalContext& context) {
    std::string error;
    const std::string root = CLOUDNR_SOURCE_ROOT;
    auto loaded = loadScene(root + "/scenes/eval_medium.json", &error);
    CHECK(loaded.has_value());
    if (!loaded) return report("load eval_medium", error);
    Scene scene = *loaded;
    scene.width = 192;
    scene.height = 108;

    auto fullConfig = loadRenderConfig(root + "/configs/analytic.json", &error);
    CHECK(fullConfig.has_value());
    if (!fullConfig) return report("load configs/analytic.json", error);
    CHECK(fullConfig->renderer == "analytic");
    CHECK(fullConfig->octaves > 1 && fullConfig->ambient);
    RenderConfig singleConfig = *fullConfig;  // same march, only L_rest removed
    singleConfig.octaves = 1;
    singleConfig.ambient = false;

    auto volume = DensityVolume::bake(context, scene, &error);
    CHECK(volume != nullptr);
    if (!volume) return report("bake eval_medium", error);

    Image full, single;
    if (!renderImage(context, scene, *fullConfig, *volume, &full)) return;
    if (!renderImage(context, scene, singleConfig, *volume, &single)) return;

    bool valid = true;
    bool monotone = true;
    double sumFull = 0.0;
    double sumSingle = 0.0;
    for (size_t i = 0; i < full.rgb.size(); ++i) {
        const float f = full.rgb[i];
        const float ss = single.rgb[i];
        valid = valid && std::isfinite(f) && std::isfinite(ss) && f >= 0.0f && ss >= 0.0f;
        monotone = monotone && f >= ss - 1e-6f * std::fabs(ss) - 1e-7f;
        sumFull += f;
        sumSingle += ss;
    }
    CHECK(valid);
    CHECK(monotone);
    CHECK(sumFull > sumSingle);  // the octaves and ambient term add visible energy
}

}  // namespace

int main(int argc, char** argv) {
    ensureValidationEnvironment(argc, argv);
    @autoreleasepool {
        std::string error;
        auto context = MetalContext::create(&error);
        CHECK(context != nullptr);
        if (!context) {
            report("MetalContext::create", error);
            return checkSummary();
        }

        {
            // a. Beer–Lambert: albedo 0, so only attenuated environment light reaches the camera.
            const Scene absorbing = makeHomogeneousScene(0.0f, simd_make_float3(0, 1, 0));
            auto volume = DensityVolume::bake(*context, absorbing, &error);
            CHECK(volume != nullptr);
            if (volume) {
                testOpticalDepth(*context, absorbing, *volume);
                testBeerLambertImage(*context, absorbing, *volume);
            } else {
                report("bake homogeneous", error);
            }
        }
        {
            // b./c. Single scattering with albedo 0.8 (the baked field does not depend on the medium/light).
            const Scene overhead = makeHomogeneousScene(0.8f, simd_make_float3(0, 1, 0));
            const Scene tilted = makeHomogeneousScene(0.8f, simd_make_float3(0.3f, 1.0f, 0.2f));
            auto volume = DensityVolume::bake(*context, overhead, &error);
            CHECK(volume != nullptr);
            if (volume) {
                testSingleScatterRays(*context, overhead, *volume);
                testSingleScatterImage(*context, tilted, *volume);
            } else {
                report("bake homogeneous", error);
            }
        }
        testHgIntegral(*context);
        testMultipleScatteringSanity(*context);

        context->waitIdle();
        CHECK(context->gpuErrorCount() == 0);
    }
    return checkSummary();
}
