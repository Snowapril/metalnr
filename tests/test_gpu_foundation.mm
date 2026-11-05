// Milestone 1 GPU integration tests: context, runtime compile, camera rays, density bake and
// sampling, density renderer. Runs under the Metal debug layer and shader validation.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "check.h"

#include "density_volume.h"
#include "image_io.h"
#include "metal_context.h"
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

const std::vector<std::string> kTestKernels = {"tests/shaders/test_kernels.metal"};

Scene makeTestScene(const std::string& density, uint32_t seed) {
    Scene scene;
    scene.name = "test";
    CameraDesc camera;
    camera.name = "side";
    camera.position = simd_make_float3(0, 200, -1500);
    camera.target = simd_make_float3(0, 200, 0);
    scene.cameras.push_back(camera);
    scene.width = 64;
    scene.height = 36;
    scene.volume.grid = simd_make_uint3(32, 32, 32);
    scene.volume.density = density;
    scene.volume.density_seed = seed;
    scene.light.sun_direction = simd_normalize(scene.light.sun_direction);
    return scene;
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

bool dispatch1D(MetalContext& context, id<MTLComputePipelineState> pipeline, id<MTL4ArgumentTable> table,
                NSUInteger threads, std::string* error) {
    return context.runSync(
        [&](id<MTL4CommandBuffer> commandBuffer) {
            id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setArgumentTable:table];
            [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [encoder endEncoding];
        },
        nullptr, error);
}

float halfToFloat(uint16_t bits) {
    __fp16 h;
    std::memcpy(&h, &bits, sizeof(h));
    return static_cast<float>(h);
}

void report(const char* what, const std::string& error) {
    std::fprintf(stderr, "%s: %s\n", what, error.c_str());
}

void testContext(MetalContext& context) {
    CHECK(context.device() != nil);
    CHECK(context.queue() != nil);
    CHECK(context.compiler() != nil);
    CHECK([context.device() supportsFamily:MTLGPUFamilyMetal4]);
    CHECK(context.validationEnabled());
}

void testRuntimeCompile(MetalContext& context) {
    std::string error;
    const double before = context.totalCompileMs();
    id<MTLComputePipelineState> pipeline = context.makeComputePipeline(kTestKernels, "test_camera_ray", "", &error);
    CHECK(pipeline != nil);
    if (!pipeline) report("compile test_camera_ray", error);
    CHECK(context.totalCompileMs() > before);

    std::string missingError;
    CHECK(context.makeComputePipeline(kTestKernels, "no_such_kernel", "", &missingError) == nil);
    CHECK(!missingError.empty());
}

void testCameraCenterRay(MetalContext& context) {
    Scene scene = makeTestScene("constant", 1);
    CameraDesc& camera = scene.cameras[0];
    camera.position = simd_make_float3(10, 150, -900);
    camera.target = simd_make_float3(-20, 210, 30);
    const SceneGPU sceneGPU = makeSceneGPU(scene, camera);

    std::string error;
    id<MTLComputePipelineState> pipeline = context.makeComputePipeline(kTestKernels, "test_camera_ray", "", &error);
    id<MTL4ArgumentTable> table = context.makeArgumentTable(2, 1, &error);
    CHECK(pipeline != nil && table != nil);
    if (!pipeline || !table) return report("camera ray setup", error);

    const float zeros[8] = {};
    id<MTLBuffer> sceneBuffer = sharedBuffer(context, &sceneGPU, sizeof(sceneGPU));
    id<MTLBuffer> outBuffer = sharedBuffer(context, zeros, sizeof(zeros));
    ScopedResidency residency(context, {sceneBuffer, outBuffer});
    [table setAddress:sceneBuffer.gpuAddress atIndex:0];
    [table setAddress:outBuffer.gpuAddress atIndex:1];
    const bool ok = dispatch1D(context, pipeline, table, 1, &error);
    CHECK(ok);
    if (!ok) return report("camera ray dispatch", error);

    const float* out = static_cast<const float*>(outBuffer.contents);
    const simd_float3 expected = simd_normalize(camera.target - camera.position);
    CHECK_NEAR(out[0], expected.x, 1e-5);
    CHECK_NEAR(out[1], expected.y, 1e-5);
    CHECK_NEAR(out[2], expected.z, 1e-5);
    CHECK_NEAR(out[4], camera.position.x, 1e-3);
    CHECK_NEAR(out[5], camera.position.y, 1e-3);
    CHECK_NEAR(out[6], camera.position.z, 1e-3);
}

bool bakeLevel0(MetalContext& context, const Scene& scene, std::vector<uint16_t>* voxels) {
    std::string error;
    auto volume = DensityVolume::bake(context, scene, &error);
    if (!volume) {
        report("bake", error);
        return false;
    }
    if (!volume->readLevel(context, 0, voxels, &error)) {
        report("readLevel", error);
        return false;
    }
    return true;
}

void testBakeDeterminism(MetalContext& context) {
    const Scene sceneA = makeTestScene("procedural", 1);
    const Scene sceneB = makeTestScene("procedural", 2);
    std::vector<uint16_t> first, second, other;
    const bool ok = bakeLevel0(context, sceneA, &first) && bakeLevel0(context, sceneA, &second) &&
                    bakeLevel0(context, sceneB, &other);
    CHECK(ok);
    if (!ok) return;

    CHECK(first.size() == size_t(32) * 32 * 32);
    CHECK(first == second);
    CHECK(first != other);

    bool inRange = true;
    float maxValue = 0.0f;
    for (const auto* field : {&first, &other}) {
        for (uint16_t bits : *field) {
            const float v = halfToFloat(bits);
            inRange = inRange && std::isfinite(v) && v >= 0.0f && v <= 1.0f;
            maxValue = std::max(maxValue, v);
        }
    }
    CHECK(inRange);
    CHECK(maxValue > 0.0f);  // the procedural field must not be empty
}

void testConstantVolume(MetalContext& context) {
    Scene scene = makeTestScene("constant", 1);
    scene.volume.constant_density = 0.5f;
    std::string error;
    auto volume = DensityVolume::bake(context, scene, &error);
    CHECK(volume != nullptr);
    if (!volume) return report("constant bake", error);
    CHECK(volume->mipLevels() == 6);

    for (uint32_t level : {0u, 1u, volume->mipLevels() - 1}) {
        std::vector<uint16_t> voxels;
        const bool ok = volume->readLevel(context, level, &voxels, &error);
        CHECK(ok);
        if (!ok) return report("constant readLevel", error);
        const size_t side = std::max<size_t>(1, 32u >> level);
        CHECK(voxels.size() == side * side * side);
        CHECK(std::all_of(voxels.begin(), voxels.end(), [](uint16_t b) { return halfToFloat(b) == 0.5f; }));
    }

    // density_at: 0 outside the AABB, the constant inside (including near faces).
    const SceneGPU sceneGPU = makeSceneGPU(scene, scene.cameras[0]);
    const simd_float3 bmin = scene.volume.bounds_min;
    const simd_float3 bmax = scene.volume.bounds_max;
    const simd_float3 center = (bmin + bmax) * 0.5f;
    const simd_float4 points[] = {
        simd_make_float4(bmin.x - 1.0f, center.y, center.z, 0),
        simd_make_float4(center.x, bmax.y + 1.0f, center.z, 0),
        simd_make_float4(center.x, center.y, bmin.z - 0.01f, 0),
        simd_make_float4(1e4f, 1e4f, 1e4f, 0),
        simd_make_float4(center.x, center.y, center.z, 0),
        simd_make_float4(bmin.x + 0.1f, bmin.y + 0.1f, bmax.z - 0.1f, 0),
    };
    const float expected[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.5f};
    constexpr NSUInteger kPoints = sizeof(points) / sizeof(points[0]);

    id<MTLComputePipelineState> pipeline = context.makeComputePipeline(kTestKernels, "test_density_at", "", &error);
    id<MTL4ArgumentTable> table = context.makeArgumentTable(3, 1, &error);
    CHECK(pipeline != nil && table != nil);
    if (!pipeline || !table) return report("density_at setup", error);

    const float init[kPoints] = {-1, -1, -1, -1, -1, -1};
    id<MTLBuffer> sceneBuffer = sharedBuffer(context, &sceneGPU, sizeof(sceneGPU));
    id<MTLBuffer> pointBuffer = sharedBuffer(context, points, sizeof(points));
    id<MTLBuffer> outBuffer = sharedBuffer(context, init, sizeof(init));
    ScopedResidency residency(context, {sceneBuffer, pointBuffer, outBuffer});
    [table setAddress:sceneBuffer.gpuAddress atIndex:0];
    [table setAddress:pointBuffer.gpuAddress atIndex:1];
    [table setAddress:outBuffer.gpuAddress atIndex:2];
    [table setTexture:volume->texture().gpuResourceID atIndex:0];
    const bool ok = dispatch1D(context, pipeline, table, kPoints, &error);
    CHECK(ok);
    if (!ok) return report("density_at dispatch", error);

    const float* out = static_cast<const float*>(outBuffer.contents);
    for (NSUInteger i = 0; i < kPoints; ++i) {
        if (expected[i] == 0.0f)
            CHECK(out[i] == 0.0f);
        else
            CHECK_NEAR(out[i], expected[i], 1e-3);
    }
}

void testDensityRenderer(MetalContext& context) {
    Scene scene = makeTestScene("procedural", 1);
    RenderConfig config;
    config.renderer = "density";
    config.primary_steps = 64;

    std::string error;
    auto volume = DensityVolume::bake(context, scene, &error);
    CHECK(volume != nullptr);
    if (!volume) return report("renderer bake", error);
    auto renderer = makeRenderer(context, scene, config, *volume, &error);
    CHECK(renderer != nullptr);
    if (!renderer) return report("makeRenderer", error);
    CHECK(std::string(renderer->name()) == "density");
    CHECK(renderer->memoryBytes() > 0);

    RenderConfig unknown;
    unknown.renderer = "no_such_renderer";
    std::string unknownError;
    CHECK(makeRenderer(context, scene, unknown, *volume, &unknownError) == nullptr);
    CHECK(unknownError == "renderer 'no_such_renderer' not available");

    const uint64_t errorsBefore = context.gpuErrorCount();
    const SceneGPU sceneGPU = makeSceneGPU(scene, scene.cameras[0]);
    double gpuMs = -1.0;
    const bool ok = context.runSync(
        [&](id<MTL4CommandBuffer> commandBuffer) {
            id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            renderer->encode(encoder, FrameInputs{sceneGPU, context.currentSlot()});
            [encoder endEncoding];
        },
        &gpuMs, &error);
    CHECK(ok);
    if (!ok) return report("density render", error);
    CHECK(context.gpuErrorCount() == errorsBefore);
    CHECK(gpuMs >= 0.0);

    Image image;
    CHECK(readOutput(*renderer, &image, &error));
    CHECK(image.width == scene.width && image.height == scene.height);

    const LightDesc& light = scene.light;
    float maxEnv = 0.0f;
    for (int c = 0; c < 3; ++c) maxEnv = std::max({maxEnv, light.sky_radiance[c], light.ground_radiance[c]});
    const float upper = std::max(maxEnv, 1.0f) + 1e-5f;

    bool valid = !image.rgb.empty();
    bool cloudVisible = false;
    for (uint32_t y = 0; y < image.height; ++y) {
        for (uint32_t x = 0; x < image.width; ++x) {
            const float* p = image.pixel(x, y);
            // A pixel matching neither sky nor ground radiance saw the medium.
            for (int c = 0; c < 3; ++c) {
                valid = valid && std::isfinite(p[c]) && p[c] >= 0.0f && p[c] <= upper;
                const float sky = light.sky_radiance[c];
                const float ground = light.ground_radiance[c];
                if (std::fabs(p[c] - sky) > 1e-3f && std::fabs(p[c] - ground) > 1e-3f) cloudVisible = true;
            }
        }
    }
    CHECK(valid);
    CHECK(cloudVisible);
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
        testContext(*context);
        testRuntimeCompile(*context);
        testCameraCenterRay(*context);
        testBakeDeterminism(*context);
        testConstantVolume(*context);
        testDensityRenderer(*context);
        context->waitIdle();
        CHECK(context->gpuErrorCount() == 0);
    }
    return checkSummary();
}
