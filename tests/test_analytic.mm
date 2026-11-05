// Analytic renderer and shared march tests: closed-form checks in a homogeneous box (Beer–Lambert
// transmittance, exact single scattering with and without a jittered first segment, HG normalization),
// the bounded bias of early termination, exactness of the empty-space skip on a cumulus, and a sanity check
// of the octaves + ambient approximation. Every scene is built in code. Runs under the Metal debug layer and
// shader validation.
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

// Homogeneous unit-density box [-1,1]^3 (16^3 voxels of 0.125 m) with sigma_t = sigma, viewed from (0,0,-5).
Scene makeHomogeneousScene(float albedo, simd_float3 sunDirection, float sigma = 1.0f) {
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
    scene.medium.sigma_t_max = sigma;
    scene.medium.albedo = albedo;
    scene.medium.hg_g = 0.3f;
    scene.light.sun_direction = simd_normalize(sunDirection);
    scene.light.sun_irradiance = simd_make_float3(3.0f, 2.5f, 2.0f);
    scene.light.sky_radiance = simd_make_float3(0.2f, 0.4f, 0.8f);
    scene.light.ground_radiance = simd_make_float3(0.1f, 0.1f, 0.1f);
    return scene;
}

// A small cumulus in a much larger AABB: most of the volume is empty. 64^3 voxels of 12.5 m.
Scene makeCumulusScene() {
    Scene scene;
    scene.name = "analytic_cumulus";
    CameraDesc camera;
    camera.name = "front";
    camera.position = simd_make_float3(0, 300, -1800);
    camera.target = simd_make_float3(0, 260, 0);
    camera.vfov_deg = 35.0f;
    scene.cameras.push_back(camera);
    scene.width = 96;
    scene.height = 64;
    scene.volume.bounds_min = simd_make_float3(-400, 0, -400);
    scene.volume.bounds_max = simd_make_float3(400, 800, 400);
    scene.volume.grid = simd_make_uint3(64, 64, 64);
    scene.volume.density = "cumulus";
    scene.volume.cloud.base_center = simd_make_float3(0, 100, 0);
    scene.volume.cloud.lobes = {simd_make_float4(0, 150, 0, 150), simd_make_float4(120, 230, 40, 110),
                                simd_make_float4(-100, 200, -30, 100)};
    scene.medium.sigma_t_max = 0.05f;
    scene.medium.albedo = 0.99f;
    scene.medium.hg_g = 0.8f;
    scene.light.sun_direction = simd_normalize(simd_make_float3(0.4f, 0.5f, -0.75f));
    scene.light.sun_irradiance = simd_make_float3(8, 8, 8);
    scene.light.sky_radiance = simd_make_float3(0.35f, 0.5f, 0.8f);
    scene.light.ground_radiance = simd_make_float3(0.12f, 0.11f, 0.1f);
    return scene;
}

// Exact single scattering through the shared march: no dither, no early termination, no empty skip.
RenderConfig exactConfig(float stepScale) {
    RenderConfig config;
    config.renderer = "analytic";
    config.step_scale = stepScale;
    config.max_steps = 8192;
    config.min_transmittance = 0.0f;
    config.dither = false;
    config.empty_skip = false;
    config.shadow_steps = 64;
    config.shadow_distance = 0.0f;
    config.octaves = 1;
    config.ambient = false;
    return config;
}

std::string shippedPath(const char* relative) { return std::string(CLOUDNR_SOURCE_ROOT) + "/" + relative; }

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

// Albedo 0: only attenuated environment light reaches the camera, L = L_env exp(-sigma chord).
void testBeerLambertImage(MetalContext& context, const Scene& scene, const DensityVolume& volume) {
    Image image;
    if (!renderImage(context, scene, exactConfig(1.0f), volume, &image)) return;

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

// ---- b. Exact single scattering along horizontal rays (sun +Y) ----

// Ls is constant along a horizontal ray, so the energy-conserving segments are exact for any segment lengths:
// L = albedo E p(0) e^{-sigma (1 - y0)} (1 - e^{-2 sigma}) + e^{-2 sigma} L_env, alpha = 1 - e^{-2 sigma}.
void testSingleScatterRays(MetalContext& context, const Scene& scene, const DensityVolume& volume) {
    const SceneGPU s = makeSceneGPU(scene, scene.cameras[0]);
    const MarchParams march = makeMarchParams(exactConfig(1.0f));
    const double sigma = scene.medium.sigma_t_max * scene.volume.constant_density;
    const double albedo = scene.medium.albedo;
    const double phase = hgCpu(0.0, scene.medium.hg_g);  // sun (0,1,0) is perpendicular to d = (1,0,0)
    const float y0s[] = {-0.5f, 0.0f, 0.6f};
    const float jitters[] = {1.0f, 0.37f};  // 1 = no dither; 0.37 = a fixed shortened first segment
    constexpr uint32_t kShadowSteps = 64;

    std::vector<simd_float4> rays;
    std::vector<float> rayY0;
    for (float jitter : jitters) {
        for (float y0 : y0s) {
            rays.push_back(simd_make_float4(-3.0f, y0, 0.2f, jitter));
            rays.push_back(simd_make_float4(1.0f, 0.0f, 0.0f, float(kShadowSteps)));
            rayY0.push_back(y0);
        }
    }
    const NSUInteger kRays = rayY0.size();
    const std::vector<simd_float4> init(kRays, simd_make_float4(-1, -1, -1, -1));

    std::string error;
    id<MTLComputePipelineState> pipeline =
        context.makeComputePipeline(kTestKernels, "test_march_single_scatter", "", &error);
    id<MTL4ArgumentTable> table = context.makeArgumentTable(4, 1, &error);
    CHECK(pipeline != nil && table != nil);
    if (!pipeline || !table) return report("single scatter setup", error);

    id<MTLBuffer> sceneBuffer = sharedBuffer(context, &s, sizeof(s));
    id<MTLBuffer> marchBuffer = sharedBuffer(context, &march, sizeof(march));
    id<MTLBuffer> rayBuffer = sharedBuffer(context, rays.data(), rays.size() * sizeof(simd_float4));
    id<MTLBuffer> outBuffer = sharedBuffer(context, init.data(), init.size() * sizeof(simd_float4));
    ScopedResidency residency(context, {sceneBuffer, marchBuffer, rayBuffer, outBuffer});
    [table setAddress:sceneBuffer.gpuAddress atIndex:0];
    [table setAddress:marchBuffer.gpuAddress atIndex:1];
    [table setAddress:rayBuffer.gpuAddress atIndex:2];
    [table setAddress:outBuffer.gpuAddress atIndex:3];
    [table setTexture:volume.texture().gpuResourceID atIndex:0];
    const bool ok = dispatch(context, pipeline, table, MTLSizeMake(kRays, 1, 1), &error);
    CHECK(ok);
    if (!ok) return report("single scatter dispatch", error);

    const simd_float4* out = static_cast<const simd_float4*>(outBuffer.contents);
    const simd_double3 sun = toDouble(s.sun_irradiance);
    const simd_double3 env = envCpu(s, simd_make_double3(1, 0, 0));  // d.y == 0 -> ground
    const double chordView = 2.0;
    for (NSUInteger i = 0; i < kRays; ++i) {
        const double tauSun = sigma * (1.0 - double(rayY0[i]));
        const double transmittance = std::exp(-sigma * chordView);
        const simd_double3 expected =
            albedo * sun * phase * std::exp(-tauSun) * (1.0 - transmittance) + transmittance * env;
        for (int c = 0; c < 3; ++c) CHECK_NEAR(out[i][c] / expected[c], 1.0, 1e-4);
        CHECK_NEAR(out[i][3] / (1.0 - transmittance), 1.0, 1e-4);
    }
}

// make_march_settings: step = voxel_min_extent * step_scale; jitter in (0, 1], 1 without DITHER.
void testMarchSettings(MetalContext& context, const Scene& scene) {
    const SceneGPU s = makeSceneGPU(scene, scene.cameras[0]);
    const NSUInteger pixels = NSUInteger(s.width) * s.height;

    std::string error;
    id<MTLComputePipelineState> pipeline =
        context.makeComputePipeline(kTestKernels, "test_march_settings", "", &error);
    id<MTL4ArgumentTable> table = context.makeArgumentTable(3, 1, &error);
    CHECK(pipeline != nil && table != nil);
    if (!pipeline || !table) return report("march settings setup", error);

    for (bool dither : {false, true}) {
        RenderConfig config = exactConfig(0.5f);
        config.dither = dither;
        const MarchParams march = makeMarchParams(config);
        CHECK(((march.flags & CNR_MARCH_DITHER) != 0) == dither);
        const std::vector<simd_float2> init(pixels, simd_make_float2(-1, -1));
        id<MTLBuffer> sceneBuffer = sharedBuffer(context, &s, sizeof(s));
        id<MTLBuffer> marchBuffer = sharedBuffer(context, &march, sizeof(march));
        id<MTLBuffer> outBuffer = sharedBuffer(context, init.data(), init.size() * sizeof(simd_float2));
        ScopedResidency residency(context, {sceneBuffer, marchBuffer, outBuffer});
        [table setAddress:sceneBuffer.gpuAddress atIndex:0];
        [table setAddress:marchBuffer.gpuAddress atIndex:1];
        [table setAddress:outBuffer.gpuAddress atIndex:2];
        const bool ok = dispatch(context, pipeline, table, MTLSizeMake(pixels, 1, 1), &error);
        CHECK(ok);
        if (!ok) return report("march settings dispatch", error);

        const simd_float2* out = static_cast<const simd_float2*>(outBuffer.contents);
        bool inRange = true;
        bool stepOk = true;
        double mean = 0.0;
        for (NSUInteger i = 0; i < pixels; ++i) {
            inRange = inRange && out[i].x > 0.0f && out[i].x <= 1.0f;
            stepOk = stepOk && std::fabs(out[i].y - 0.5f * s.voxel_min_extent) <= 1e-6f * s.voxel_min_extent;
            mean += out[i].x;
        }
        mean /= double(pixels);
        CHECK(inRange);
        CHECK(stepOk);
        if (dither) {
            CHECK_NEAR(mean, 0.5, 0.05);  // interleaved gradient noise is close to uniform
        } else {
            CHECK(mean == 1.0);
        }
    }
}

// ---- c. Single scattering image with a tilted sun (configs/analytic_ss.json) ----

void testSingleScatterImage(MetalContext& context, const Scene& scene, const DensityVolume& volume) {
    std::string error;
    auto config = loadRenderConfig(shippedPath("configs/analytic_ss.json"), &error);
    CHECK(config.has_value());
    if (!config) return report("load configs/analytic_ss.json", error);
    CHECK(config->renderer == "analytic");
    CHECK(config->octaves == 1 && !config->ambient);
    CHECK(!config->dither && !config->empty_skip && config->min_transmittance == 0.0f);
    CHECK(config->step_scale == 0.25f && config->max_steps == 8192 && config->shadow_steps == 64);
    CHECK(config->shadow_distance == 0.0f);

    Image image;
    if (!renderImage(context, scene, *config, volume, &image)) return;

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

    const double rel = relMSE(image, reference, 0.0f);
    CHECK(rel < 1e-4);
    if (!(rel < 1e-4)) std::fprintf(stderr, "single scatter image: relMSE %.3g\n", rel);
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

// ---- e. Early termination: bounded bias on a thick box ----

// Stopping at T_stop < m drops T_stop (R - (1 - T') L_env) with R <= albedo max Ls (1 - T'), and single-scatter
// Ls <= E p(cos). So |terminated - full| < m * max(albedo E p(cos), L_env) per pixel and channel.
void testEarlyTermination(MetalContext& context, const Scene& scene, const DensityVolume& volume) {
    constexpr float kMinTransmittance = 0.01f;
    const RenderConfig fullConfig = exactConfig(1.0f);
    RenderConfig stopConfig = fullConfig;
    stopConfig.min_transmittance = kMinTransmittance;

    Image full, stopped;
    if (!renderImage(context, scene, fullConfig, volume, &full)) return;
    if (!renderImage(context, scene, stopConfig, volume, &stopped)) return;

    const SceneGPU s = makeSceneGPU(scene, scene.cameras[0]);
    const simd_double3 sunE = toDouble(s.sun_irradiance);
    bool bounded = true;
    uint32_t changed = 0;
    double worst = 0.0;
    for (uint32_t y = 0; y < full.height; ++y) {
        for (uint32_t x = 0; x < full.width; ++x) {
            const simd_double3 d = cameraDirection(s, x + 0.5, y + 0.5);
            const simd_double3 env = envCpu(s, d);
            const double phase = hgCpu(simd_dot(toDouble(s.sun_direction), d), scene.medium.hg_g);
            const float* a = stopped.pixel(x, y);
            const float* b = full.pixel(x, y);
            bool differs = false;
            for (int c = 0; c < 3; ++c) {
                const double bound =
                    kMinTransmittance * std::max(scene.medium.albedo * sunE[c] * phase, env[c]) + 1e-6;
                const double diff = std::fabs(double(a[c]) - double(b[c]));
                bounded = bounded && diff <= bound;
                worst = std::max(worst, diff / bound);
                differs = differs || a[c] != b[c];
            }
            if (differs) ++changed;
        }
    }
    CHECK(bounded);
    if (!bounded) std::fprintf(stderr, "early termination: worst |diff| / bound %.3g\n", worst);
    CHECK(changed > 0);  // the thick box does trigger the termination
}

// ---- f. Empty-space skip is exact ----

void testEmptySkip(MetalContext& context, const Scene& scene, const DensityVolume& volume) {
    RenderConfig skipConfig;  // article march: step 1 voxel, dither, early termination
    skipConfig.renderer = "analytic";
    skipConfig.max_steps = 8192;  // never binding, so skipping cannot reach further than the plain march
    skipConfig.empty_skip = true;
    RenderConfig plainConfig = skipConfig;
    plainConfig.empty_skip = false;

    Image skipped, plain;
    if (!renderImage(context, scene, skipConfig, volume, &skipped)) return;
    if (!renderImage(context, scene, plainConfig, volume, &plain)) return;

    const SceneGPU s = makeSceneGPU(scene, scene.cameras[0]);
    double maxValue = 0.0;
    double maxDiff = 0.0;
    uint32_t cloudPixels = 0;
    for (uint32_t y = 0; y < plain.height; ++y) {
        for (uint32_t x = 0; x < plain.width; ++x) {
            const simd_double3 env = envCpu(s, cameraDirection(s, x + 0.5, y + 0.5));
            const float* a = skipped.pixel(x, y);
            const float* b = plain.pixel(x, y);
            bool cloud = false;
            for (int c = 0; c < 3; ++c) {
                maxValue = std::max(maxValue, double(std::fabs(b[c])));
                maxDiff = std::max(maxDiff, std::fabs(double(a[c]) - double(b[c])));
                cloud = cloud || std::fabs(b[c] - env[c]) > 1e-3 * env[c];
            }
            if (cloud) ++cloudPixels;
        }
    }
    const uint32_t pixels = plain.width * plain.height;
    CHECK(cloudPixels > pixels / 50);  // the cloud is visible ...
    CHECK(cloudPixels < pixels);      // ... and does not fill the frame
    CHECK(maxDiff <= 1e-5 * maxValue);
    if (!(maxDiff <= 1e-5 * maxValue))
        std::fprintf(stderr, "empty skip: max |diff| %.3g (max value %.3g)\n", maxDiff, maxValue);
}

// ---- g. Octaves + ambient only add light (configs/analytic.json) ----

void testMultipleScatteringSanity(MetalContext& context, const Scene& scene, const DensityVolume& volume) {
    std::string error;
    auto fullConfig = loadRenderConfig(shippedPath("configs/analytic.json"), &error);
    CHECK(fullConfig.has_value());
    if (!fullConfig) return report("load configs/analytic.json", error);
    CHECK(fullConfig->renderer == "analytic");
    CHECK(fullConfig->step_scale == 1.0f && fullConfig->max_steps == 512);
    CHECK(fullConfig->min_transmittance == 0.01f && fullConfig->dither && fullConfig->empty_skip);
    CHECK(fullConfig->shadow_steps == 6 && fullConfig->shadow_distance == 0.0f && fullConfig->ambient_steps == 4);
    CHECK(fullConfig->octaves == 3 && fullConfig->ambient);
    CHECK(fullConfig->ms_a == 0.4f && fullConfig->ms_b == 0.35f && fullConfig->ms_c == 0.1f);
    CHECK(makeMarchParams(*fullConfig).flags == (CNR_MARCH_DITHER | CNR_MARCH_EMPTY_SKIP));
    RenderConfig singleConfig = *fullConfig;  // same march (same T sequence), only the extra terms removed
    singleConfig.octaves = 1;
    singleConfig.ambient = false;

    Image full, single;
    if (!renderImage(context, scene, *fullConfig, volume, &full)) return;
    if (!renderImage(context, scene, singleConfig, volume, &single)) return;

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
            // The baked field depends only on the volume, so one bake serves every homogeneous-box scene.
            const Scene absorbing = makeHomogeneousScene(0.0f, simd_make_float3(0, 1, 0));
            const Scene overhead = makeHomogeneousScene(0.8f, simd_make_float3(0, 1, 0));
            const Scene tilted = makeHomogeneousScene(0.8f, simd_make_float3(0.3f, 1.0f, 0.2f));
            const Scene thick = makeHomogeneousScene(0.8f, simd_make_float3(0.3f, 1.0f, 0.2f), 20.0f);
            auto volume = DensityVolume::bake(*context, absorbing, &error);
            CHECK(volume != nullptr);
            if (volume) {
                testOpticalDepth(*context, absorbing, *volume);      // a
                testBeerLambertImage(*context, absorbing, *volume);  // a
                testSingleScatterRays(*context, overhead, *volume);  // b
                testMarchSettings(*context, overhead);
                testSingleScatterImage(*context, tilted, *volume);  // c
                testEarlyTermination(*context, thick, *volume);     // e
            } else {
                report("bake homogeneous", error);
            }
        }
        testHgIntegral(*context);  // d
        {
            const Scene cumulus = makeCumulusScene();
            auto volume = DensityVolume::bake(*context, cumulus, &error);
            CHECK(volume != nullptr);
            if (volume) {
                testEmptySkip(*context, cumulus, *volume);                 // f
                testMultipleScatteringSanity(*context, cumulus, *volume);  // g
            } else {
                report("bake cumulus", error);
            }
        }

        context->waitIdle();
        CHECK(context->gpuErrorCount() == 0);
    }
    return checkSummary();
}
