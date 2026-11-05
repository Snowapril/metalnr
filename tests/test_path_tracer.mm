// Milestone 2 path tracer tests: transmittance, white furnace, HG sampling, determinism,
// single scattering vs quadrature, 1/spp convergence, bounce-cap counting.
// Runs under the Metal debug layer and shader validation. Statistical checks use 3 sigma of the
// sample mean, computed from the sample variance; seeds are fixed, so results are reproducible.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "check.h"

#include "density_volume.h"
#include "image_io.h"
#include "metal_context.h"
#include "metrics.h"
#include "path_tracer.h"
#include "renderer.h"
#include "scene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

const std::vector<std::string> kTestKernels = {"tests/shaders/test_path_tracer.metal"};

// Mirrors of the structs in tests/shaders/test_path_tracer.metal.
struct WalkTestParams {
    simd_float4 origin;
    simd_float4 dir;
    uint32_t seed;
    uint32_t max_bounces;
    uint32_t rr_start_bounce;
    uint32_t bounce_cap;
};
static_assert(sizeof(WalkTestParams) == 48, "WalkTestParams layout");

struct HgTestParams {
    simd_float4 d;
    float g;
    uint32_t seed;
    uint32_t pad[2];
};
static_assert(sizeof(HgTestParams) == 32, "HgTestParams layout");

constexpr NSUInteger kWalks = 1u << 18;
constexpr NSUInteger kHgSamples = 1u << 20;

void report(const char* what, const std::string& error) { std::fprintf(stderr, "%s: %s\n", what, error.c_str()); }

// Unit box [-1,1]^3 of constant density 1 (sigma_t = sigma_t_max), sun and environment off.
Scene makeBoxScene(uint32_t size) {
    Scene scene;
    scene.name = "pt_box";
    CameraDesc camera;
    camera.name = "front";
    camera.position = simd_make_float3(0, 0, -4);
    camera.target = simd_make_float3(0, 0, 0);
    camera.vfov_deg = 40.0f;
    scene.cameras.push_back(camera);
    scene.width = size;
    scene.height = size;
    scene.volume.bounds_min = simd_make_float3(-1, -1, -1);
    scene.volume.bounds_max = simd_make_float3(1, 1, 1);
    scene.volume.grid = simd_make_uint3(16, 16, 16);
    scene.volume.density = "constant";
    scene.volume.constant_density = 1.0f;
    scene.medium = MediumDesc{1.0f, 0.0f, 0.0f};
    scene.light.sun_direction = simd_make_float3(0, 1, 0);
    scene.light.sun_irradiance = simd_make_float3(0, 0, 0);
    scene.light.sky_radiance = simd_make_float3(1, 1, 1);
    scene.light.ground_radiance = simd_make_float3(1, 1, 1);
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

id<MTLBuffer> sharedBuffer(MetalContext& context, const void* bytes, size_t length) {
    return [context.device() newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
}

bool dispatch1D(MetalContext& context, id<MTLComputePipelineState> pipeline, id<MTL4ArgumentTable> table,
                NSUInteger threads, std::string* error) {
    return context.runSync(
        [&](id<MTL4CommandBuffer> commandBuffer) {
            id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setArgumentTable:table];
            [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [encoder endEncoding];
        },
        nullptr, error);
}

struct MeanStd {
    double mean = 0.0;
    double stdOfMean = 0.0;  // sqrt(sample variance / N)
};

template <typename Get>
MeanStd meanStd(size_t n, Get get) {
    double sum = 0.0, sumSq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double v = get(i);
        sum += v;
        sumSq += v * v;
    }
    MeanStd r;
    r.mean = sum / double(n);
    const double var = std::max(0.0, (sumSq - double(n) * r.mean * r.mean) / double(n - 1));
    r.stdOfMean = std::sqrt(var / double(n));
    return r;
}

// Slab test in double; t0 clamped to >= 0. Returns false on a miss.
bool chord(simd_double3 o, simd_double3 d, simd_double3 bmin, simd_double3 bmax, double* t0, double* t1) {
    double lo = 0.0, hi = INFINITY;
    for (int a = 0; a < 3; ++a) {
        if (std::fabs(d[a]) < 1e-300) {
            if (o[a] < bmin[a] || o[a] > bmax[a]) return false;
            continue;
        }
        double ta = (bmin[a] - o[a]) / d[a];
        double tb = (bmax[a] - o[a]) / d[a];
        if (ta > tb) std::swap(ta, tb);
        lo = std::max(lo, ta);
        hi = std::min(hi, tb);
    }
    *t0 = lo;
    *t1 = hi;
    return hi > lo;
}

double hgEval(double cosTheta, double g) {
    const double denom = 1.0 + g * g - 2.0 * g * cosTheta;
    return (1.0 - g * g) / (4.0 * M_PI * denom * std::sqrt(denom));
}

simd_double3 toDouble(simd_float3 v) { return simd_make_double3(v.x, v.y, v.z); }

// Runs kWalks independent random_walk estimates of one ray; returns (rgb, hit_cap) per walk.
bool runWalks(MetalContext& context, const DensityVolume& volume, const SceneGPU& sceneGPU,
              const WalkTestParams& params, std::vector<simd_float4>* out) {
    // Compiled once; every walk test uses the same kernel.
    static id<MTLComputePipelineState> pipeline = nil;
    static id<MTL4ArgumentTable> table = nil;
    std::string error;
    if (!pipeline) pipeline = context.makeComputePipeline(kTestKernels, "test_random_walk", "", &error);
    if (pipeline && !table) table = context.makeArgumentTable(3, 1, &error);
    if (!pipeline || !table) {
        report("random walk setup", error);
        return false;
    }
    id<MTLBuffer> sceneBuffer = sharedBuffer(context, &sceneGPU, sizeof(sceneGPU));
    id<MTLBuffer> paramBuffer = sharedBuffer(context, &params, sizeof(params));
    id<MTLBuffer> outBuffer = [context.device() newBufferWithLength:kWalks * sizeof(simd_float4)
                                                            options:MTLResourceStorageModeShared];
    ScopedResidency residency(context, {sceneBuffer, paramBuffer, outBuffer});
    [table setAddress:sceneBuffer.gpuAddress atIndex:0];
    [table setAddress:paramBuffer.gpuAddress atIndex:1];
    [table setAddress:outBuffer.gpuAddress atIndex:2];
    [table setTexture:volume.texture().gpuResourceID atIndex:0];
    if (!dispatch1D(context, pipeline, table, kWalks, &error)) {
        report("random walk dispatch", error);
        return false;
    }
    const simd_float4* values = static_cast<const simd_float4*>(outBuffer.contents);
    out->assign(values, values + kWalks);
    return true;
}

WalkTestParams walkParams(simd_float3 origin, simd_float3 dir, uint32_t seed, uint32_t maxBounces,
                          uint32_t rrStart, uint32_t cap) {
    WalkTestParams p{};
    p.origin = simd_make_float4(origin, 0.0f);
    p.dir = simd_make_float4(simd_normalize(dir), 0.0f);
    p.seed = seed;
    p.max_bounces = maxBounces;
    p.rr_start_bounce = rrStart;
    p.bounce_cap = cap;
    return p;
}

// Renders `frames` progressive frames through the context frame loop and waits.
bool accumulateFrames(MetalContext& context, Renderer& renderer, const SceneGPU& sceneGPU, uint32_t frames) {
    const uint64_t before = context.gpuErrorCount();
    for (uint32_t f = 0; f < frames; ++f) {
        id<MTL4CommandBuffer> commandBuffer = context.beginFrame();
        id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        renderer.encode(encoder, FrameInputs{sceneGPU, context.currentSlot()});
        [encoder endEncoding];
        context.endFrame(commandBuffer, MetalContext::FrameFeedback{});
    }
    context.waitIdle();
    return context.gpuErrorCount() == before;
}

double statValue(const Renderer& renderer, const std::string& key) {
    for (const auto& [name, value] : renderer.stats())
        if (name == key) return value;
    return std::nan("");
}

// (a) albedo 0: E[L] = L_env * exp(-sigma_t * chord) for an axis ray and an oblique ray.
void testTransmittance(MetalContext& context, const DensityVolume& box) {
    Scene scene = makeBoxScene(32);
    const SceneGPU sceneGPU = makeSceneGPU(scene, scene.cameras[0]);
    const simd_float3 obliqueDir = simd_normalize(simd_make_float3(0.3f, 0.4f, 1.0f));
    const struct {
        simd_float3 origin;
        simd_float3 dir;
    } rays[] = {
        {simd_make_float3(0, 0, -4), simd_make_float3(0, 0, 1)},
        {simd_make_float3(0.1f, -0.05f, 0.0f) - 4.0f * obliqueDir, obliqueDir},
    };
    uint32_t seed = 11;
    for (const auto& ray : rays) {
        const WalkTestParams params = walkParams(ray.origin, ray.dir, seed++, 0, 16, kPathTracerBounceCap);
        double t0 = 0, t1 = 0;
        const bool hit = chord(toDouble(params.origin.xyz), toDouble(params.dir.xyz), simd_make_double3(-1, -1, -1),
                               simd_make_double3(1, 1, 1), &t0, &t1);
        CHECK(hit);
        const double expected = std::exp(-1.0 * (t1 - t0));
        std::vector<simd_float4> walks;
        if (!runWalks(context, box, sceneGPU, params, &walks)) {
            CHECK(false);
            continue;
        }
        const MeanStd ms = meanStd(walks.size(), [&](size_t i) { return double(walks[i].x); });
        CHECK_NEAR(ms.mean, expected, 3.0 * ms.stdOfMean);
        CHECK(ms.stdOfMean > 0.0);
    }
}

// (b) albedo 1, sky = ground = 1, no sun: every ray entering the box returns 1 in expectation.
void testWhiteFurnace(MetalContext& context, const DensityVolume& box) {
    const struct {
        simd_float3 origin;
        simd_float3 dir;
    } rays[] = {
        {simd_make_float3(0, 0, -4), simd_make_float3(0, 0, 1)},
        {simd_make_float3(0.2f, 4, -0.1f), simd_make_float3(0, -1, 0)},
        {simd_make_float3(3, 2, -3), simd_make_float3(-3.1f, -2.2f, 3.0f)},
    };
    uint32_t seed = 21;
    for (float g : {0.0f, 0.8f}) {
        Scene scene = makeBoxScene(32);
        scene.medium = MediumDesc{2.0f, 1.0f, g};
        const SceneGPU sceneGPU = makeSceneGPU(scene, scene.cameras[0]);
        for (const auto& ray : rays) {
            const WalkTestParams params = walkParams(ray.origin, ray.dir, seed++, 0, 16, kPathTracerBounceCap);
            std::vector<simd_float4> walks;
            if (!runWalks(context, box, sceneGPU, params, &walks)) {
                CHECK(false);
                continue;
            }
            const MeanStd ms = meanStd(walks.size(), [&](size_t i) { return double(walks[i].x); });
            CHECK_NEAR(ms.mean, 1.0, 3.0 * ms.stdOfMean + 1e-6);
            const bool noCap = std::all_of(walks.begin(), walks.end(), [](simd_float4 v) { return v.w == 0.0f; });
            CHECK(noCap);
        }
    }
}

// (c) HG sampling: E[cos(d, w)] = g and |w| = 1.
void testHgSampling(MetalContext& context) {
    std::string error;
    id<MTLComputePipelineState> pipeline = context.makeComputePipeline(kTestKernels, "test_hg_sample", "", &error);
    id<MTL4ArgumentTable> table = context.makeArgumentTable(2, 1, &error);
    CHECK(pipeline != nil && table != nil);
    if (!pipeline || !table) return report("hg setup", error);

    uint32_t seed = 31;
    for (float g : {-0.5f, 0.0f, 0.8f}) {
        HgTestParams params{};
        params.d = simd_make_float4(simd_normalize(simd_make_float3(0.3f, -0.5f, 0.8f)), 0.0f);
        params.g = g;
        params.seed = seed++;
        id<MTLBuffer> paramBuffer = sharedBuffer(context, &params, sizeof(params));
        id<MTLBuffer> outBuffer = [context.device() newBufferWithLength:kHgSamples * sizeof(simd_float2)
                                                                options:MTLResourceStorageModeShared];
        ScopedResidency residency(context, {paramBuffer, outBuffer});
        [table setAddress:paramBuffer.gpuAddress atIndex:0];
        [table setAddress:outBuffer.gpuAddress atIndex:1];
        const bool ok = dispatch1D(context, pipeline, table, kHgSamples, &error);
        CHECK(ok);
        if (!ok) return report("hg dispatch", error);

        const simd_float2* out = static_cast<const simd_float2*>(outBuffer.contents);
        const MeanStd ms = meanStd(kHgSamples, [&](size_t i) { return double(out[i].x); });
        CHECK_NEAR(ms.mean, g, 3.0 * ms.stdOfMean);
        double maxLengthError = 0.0;
        for (NSUInteger i = 0; i < kHgSamples; ++i)
            maxLengthError = std::max(maxLengthError, std::fabs(double(out[i].y) - 1.0));
        CHECK(maxLengthError <= 1e-5);
    }
}

// (d) Same seed -> bit-identical image; another seed -> different image.
void testDeterminism(MetalContext& context) {
    std::string error;
    auto loaded = loadScene(std::string(CLOUDNR_SOURCE_ROOT) + "/scenes/cumulus.json", &error);
    CHECK(loaded.has_value());
    if (!loaded) return report("load cumulus", error);
    Scene scene = *loaded;
    scene.volume.grid = simd_make_uint3(32, 32, 32);
    scene.width = 64;
    scene.height = 36;
    auto volume = DensityVolume::bake(context, scene, &error);
    CHECK(volume != nullptr);
    if (!volume) return report("determinism bake", error);
    const SceneGPU sceneGPU = makeSceneGPU(scene, scene.cameras[0]);

    auto render = [&](uint32_t seed, Image* image, double* capHits, double* meanBounces) {
        RenderConfig config;
        config.renderer = "pt";
        config.seed = seed;
        std::string e;
        auto renderer = makeRenderer(context, scene, config, *volume, &e);
        if (!renderer) {
            report("makeRenderer pt", e);
            return false;
        }
        if (std::string(renderer->name()) != "pt" || !accumulateFrames(context, *renderer, sceneGPU, 8)) return false;
        if (renderer->accumulatedSamples() != 8) return false;
        *capHits = statValue(*renderer, "bounce_cap_hits");
        *meanBounces = statValue(*renderer, "mean_bounces");
        return readOutput(*renderer, image, &e);
    };
    Image first, second, other;
    double cap = -1, bounces = -1, unused0 = 0, unused1 = 0;
    const bool ok = render(1, &first, &cap, &bounces) && render(1, &second, &unused0, &unused1) &&
                    render(2, &other, &unused0, &unused1);
    CHECK(ok);
    if (!ok) return;
    CHECK(first.width == 64 && first.height == 36);
    CHECK(first.rgb.size() == second.rgb.size() &&
          std::memcmp(first.rgb.data(), second.rgb.data(), first.rgb.size() * sizeof(float)) == 0);
    CHECK(first.rgb.size() == other.rgb.size() &&
          std::memcmp(first.rgb.data(), other.rgb.data(), first.rgb.size() * sizeof(float)) != 0);
    CHECK(std::all_of(first.rgb.begin(), first.rgb.end(), [](float v) { return std::isfinite(v) && v >= 0.0f; }));
    CHECK(cap == 0.0);
    CHECK(bounces > 0.0 && std::isfinite(bounces));
}

// (e) max_bounces = 1 against a CPU quadrature of single scattering plus attenuated environment.
void testSingleScattering(MetalContext& context, const DensityVolume& box) {
    Scene scene = makeBoxScene(32);
    const float sigmaT = 1.0f, albedo = 0.8f, g = 0.3f;
    scene.medium = MediumDesc{sigmaT, albedo, g};
    scene.light.sun_direction = simd_normalize(simd_make_float3(0.3f, 1.0f, 0.2f));
    scene.light.sun_irradiance = simd_make_float3(1, 1, 1);
    scene.light.sky_radiance = simd_make_float3(0.5f, 0.5f, 0.5f);
    scene.light.ground_radiance = simd_make_float3(0.25f, 0.25f, 0.25f);
    const SceneGPU sceneGPU = makeSceneGPU(scene, scene.cameras[0]);

    const simd_float3 dir = simd_normalize(simd_make_float3(0.1f, 0.2f, 1.0f));
    const WalkTestParams params =
        walkParams(simd_make_float3(0.05f, -0.1f, 0.0f) - 4.0f * dir, dir, 41, 1, 16, kPathTracerBounceCap);

    const simd_double3 o = toDouble(params.origin.xyz);
    const simd_double3 d = toDouble(params.dir.xyz);
    const simd_double3 sun = toDouble(sceneGPU.sun_direction);
    const simd_double3 bmin = simd_make_double3(-1, -1, -1), bmax = simd_make_double3(1, 1, 1);
    double t0 = 0, t1 = 0;
    CHECK(chord(o, d, bmin, bmax, &t0, &t1));
    const double phase = hgEval(simd_dot(sun, d), g);
    constexpr int kSteps = 200000;
    const double dt = (t1 - t0) / kSteps;
    double single = 0.0;
    for (int i = 0; i < kSteps; ++i) {
        const double t = t0 + (i + 0.5) * dt;
        double s0 = 0, s1 = 0;
        chord(o + t * d, sun, bmin, bmax, &s0, &s1);
        single += std::exp(-sigmaT * (t - t0)) * albedo * sigmaT * phase * std::exp(-sigmaT * (s1 - s0)) * dt;
    }
    const double env = d.y > 0.0 ? 0.5 : 0.25;
    const double expected = single + std::exp(-sigmaT * (t1 - t0)) * env;

    std::vector<simd_float4> walks;
    if (!runWalks(context, box, sceneGPU, params, &walks)) return CHECK(false);
    const MeanStd ms = meanStd(walks.size(), [&](size_t i) { return double(walks[i].x); });
    CHECK_NEAR(ms.mean, expected, 3.0 * ms.stdOfMean);
}

// (f) relMSE between independent halves falls as 1/spp: ratio between 16 and 256 spp is ~16.
void testConvergence(MetalContext& context, const DensityVolume& box) {
    Scene scene = makeBoxScene(32);
    scene.medium = MediumDesc{1.0f, 0.9f, 0.3f};
    scene.light.sun_direction = simd_normalize(simd_make_float3(0.3f, 1.0f, 0.2f));
    scene.light.sun_irradiance = simd_make_float3(2, 2, 2);
    scene.light.sky_radiance = simd_make_float3(0.2f, 0.3f, 0.5f);
    scene.light.ground_radiance = simd_make_float3(0.05f, 0.05f, 0.05f);
    const SceneGPU sceneGPU = makeSceneGPU(scene, scene.cameras[0]);

    Image a16, a256, b16, b256;
    auto half = [&](uint32_t seed, Image* at16, Image* at256) {
        RenderConfig config;
        config.renderer = "pt";
        config.seed = seed;
        std::string e;
        auto renderer = makeRenderer(context, scene, config, box, &e);
        if (!renderer) {
            report("makeRenderer pt", e);
            return false;
        }
        return accumulateFrames(context, *renderer, sceneGPU, 16) && readOutput(*renderer, at16, &e) &&
               accumulateFrames(context, *renderer, sceneGPU, 240) && readOutput(*renderer, at256, &e) &&
               renderer->accumulatedSamples() == 256;
    };
    const bool ok = half(1, &a16, &a256) && half(2, &b16, &b256);
    CHECK(ok);
    if (!ok) return;
    const double e16 = relMSE(a16, b16, 0.0f);
    const double e256 = relMSE(a256, b256, 0.0f);
    CHECK(e256 > 0.0);
    const double ratio = e16 / e256;
    std::fprintf(stderr, "convergence: relMSE16 %.6g relMSE256 %.6g ratio %.3f\n", e16, e256, ratio);
    CHECK(ratio >= 8.0 && ratio <= 32.0);
}

// (g) pt_render counts paths that hit a small bounce cap (dense, conservative medium).
void testBounceCap(MetalContext& context, const DensityVolume& box) {
    constexpr uint32_t kSize = 16;
    constexpr uint32_t kCap = 8;
    Scene scene = makeBoxScene(kSize);
    scene.medium = MediumDesc{50.0f, 1.0f, 0.0f};
    const SceneGPU sceneGPU = makeSceneGPU(scene, scene.cameras[0]);

    // Walk kernel: same random_walk, per-path flag.
    const WalkTestParams walk =
        walkParams(simd_make_float3(0, 0, -4), simd_make_float3(0, 0, 1), 51, 0, 0, kCap);
    std::vector<simd_float4> walks;
    if (runWalks(context, box, sceneGPU, walk, &walks)) {
        const size_t capped =
            std::count_if(walks.begin(), walks.end(), [](simd_float4 v) { return v.w == 1.0f; });
        CHECK(capped > 0);
    } else {
        CHECK(false);
    }

    // pt_render directly, with bounce_cap = kCap instead of the renderer's 4096.
    std::string error;
    id<MTLComputePipelineState> pipeline =
        context.makeComputePipeline({"src/shaders/path_tracer.metal"}, "pt_render", "", &error);
    id<MTL4ArgumentTable> table = context.makeArgumentTable(3, 2, &error);
    CHECK(pipeline != nil && table != nil);
    if (!pipeline || !table) return report("pt_render setup", error);

    PathTracerParams params{};
    params.seed = 1;
    params.sample_index = 0;
    params.max_bounces = 0;
    params.rr_start_bounce = 0;
    params.bounce_cap = kCap;
    const uint32_t zeros[4] = {};
    id<MTLBuffer> sceneBuffer = sharedBuffer(context, &sceneGPU, sizeof(sceneGPU));
    id<MTLBuffer> paramBuffer = sharedBuffer(context, &params, sizeof(params));
    id<MTLBuffer> counters = sharedBuffer(context, zeros, sizeof(zeros));
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                    width:kSize
                                                                                   height:kSize
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> accum = [context.device() newTextureWithDescriptor:desc];
    ScopedResidency residency(context, {sceneBuffer, paramBuffer, counters, accum});
    [table setAddress:sceneBuffer.gpuAddress atIndex:0];
    [table setAddress:paramBuffer.gpuAddress atIndex:1];
    [table setAddress:counters.gpuAddress atIndex:2];
    [table setTexture:box.texture().gpuResourceID atIndex:0];
    [table setTexture:accum.gpuResourceID atIndex:1];
    const bool ok = context.runSync(
        [&](id<MTL4CommandBuffer> commandBuffer) {
            id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setArgumentTable:table];
            [encoder dispatchThreads:MTLSizeMake(kSize, kSize, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            [encoder endEncoding];
        },
        nullptr, &error);
    CHECK(ok);
    if (!ok) return report("pt_render dispatch", error);

    const uint32_t* words = static_cast<const uint32_t*>(counters.contents);
    const uint64_t capHits = uint64_t(words[0]) | (uint64_t(words[1]) << 32);
    const uint64_t bounces = uint64_t(words[2]) | (uint64_t(words[3]) << 32);
    std::fprintf(stderr, "bounce cap: %llu of %u paths capped, %llu bounces\n",
                 static_cast<unsigned long long>(capHits), kSize * kSize, static_cast<unsigned long long>(bounces));
    CHECK(capHits > 0);
    CHECK(capHits <= uint64_t(kSize) * kSize);
    CHECK(bounces >= capHits * kCap);
    CHECK(bounces <= uint64_t(kSize) * kSize * kCap);
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
        CHECK(context->validationEnabled());
        auto box = DensityVolume::bake(*context, makeBoxScene(32), &error);
        CHECK(box != nullptr);
        if (!box) {
            report("box bake", error);
            return checkSummary();
        }
        testTransmittance(*context, *box);
        testWhiteFurnace(*context, *box);
        testHgSampling(*context);
        testDeterminism(*context);
        testSingleScattering(*context, *box);
        testConvergence(*context, *box);
        testBounceCap(*context, *box);
        context->waitIdle();
        CHECK(context->gpuErrorCount() == 0);
    }
    return checkSummary();
}
