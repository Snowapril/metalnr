// Shared scene definition and renderer configuration (docs/architecture.md §4).
// JSON load/save uses NSJSONSerialization; keys are documented in the architecture doc.
#pragma once

#include "shared_types.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct CameraDesc {
    std::string name;            // e.g. "side", "backlit"
    simd_float3 position{0, 200, -1500};
    simd_float3 target{0, 200, 0};
    float vfov_deg = 40.0f;
    float exposure_ev = 0.0f;
};

struct VolumeDesc {
    simd_float3 bounds_min{-400, 0, -400};
    simd_float3 bounds_max{400, 400, 400};
    simd_uint3 grid{256, 128, 256};
    std::string density = "procedural";  // "procedural" | "constant"
    uint32_t density_seed = 1;
    float coverage = 0.5f;
    float erosion = 0.3f;
    float constant_density = 1.0f;       // only for density == "constant"
};

struct MediumDesc {
    float sigma_t_max = 0.04f;  // 1/m
    float albedo = 0.99f;
    float hg_g = 0.8f;
};

struct LightDesc {
    simd_float3 sun_direction{0.3f, 0.6f, 0.74f};  // toward the sun; normalized on load
    simd_float3 sun_irradiance{3, 3, 3};
    simd_float3 sky_radiance{0.2f, 0.3f, 0.5f};
    simd_float3 ground_radiance{0, 0, 0};
};

struct Scene {
    uint32_t version = 1;
    std::string name;
    std::vector<CameraDesc> cameras;  // JSON "cameras" array; cameras[0] is the default
    uint32_t width = 960;
    uint32_t height = 540;
    VolumeDesc volume;
    MediumDesc medium;
    LightDesc light;

    // nullptr if no camera has that name. Empty name selects cameras[0].
    const CameraDesc* camera(const std::string& name) const;
};

// renderer: "density" | "analytic" | "pt" | "neural". Only the fields used by that
// renderer are serialized; the others keep their defaults.
struct RenderConfig {
    std::string renderer = "density";
    // density, analytic, neural
    uint32_t primary_steps = 128;
    // analytic, neural
    uint32_t shadow_steps = 16;
    // analytic
    uint32_t ambient_steps = 4;
    uint32_t octaves = 4;
    float ms_a = 0.5f;
    float ms_b = 0.5f;
    float ms_c = 0.5f;
    bool ambient = true;
    // pt
    uint32_t spp = 4096;
    uint32_t seed = 1;
    uint32_t max_bounces = 0;  // 0 = unlimited
    uint32_t rr_start_bounce = 16;
    // neural
    std::string model;  // path to .cnrm, relative to the working directory
};

std::optional<Scene> loadScene(const std::string& path, std::string* error);
bool saveScene(const Scene& scene, const std::string& path, std::string* error);
std::optional<RenderConfig> loadRenderConfig(const std::string& path, std::string* error);
bool saveRenderConfig(const RenderConfig& config, const std::string& path, std::string* error);

// Builds the per-frame GPU constants for one camera. mip_levels is the full chain of
// volume.grid; voxel_min_extent = min((bounds_max - bounds_min) / grid).
SceneGPU makeSceneGPU(const Scene& scene, const CameraDesc& camera);

// Bake parameters derived from scene.volume.
DensityBakeParams makeDensityBakeParams(const Scene& scene);
