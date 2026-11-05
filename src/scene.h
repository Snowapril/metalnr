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

// The article's cumulus (docs/architecture.md §4). Lengths in metres, cloud-local frame with the origin at
// base_center (centre of the flat base, +Y up). JSON object "volume.cloud".
struct CloudDesc {
    simd_float3 base_center{0, 100, 0};    // world position of the cloud-local origin
    std::vector<simd_float4> lobes;        // xyz centre, w radius; 1..CNR_MAX_LOBES
    float lobe_blend = 40.0f;
    simd_float2 shear{0.15f, 0.0f};        // horizontal offset per metre of height (x, z)
    float anvil_height = 450.0f;
    float height = 600.0f;
    float anvil_spread = 120.0f;
    float radius = 250.0f;
    float base_softness = 15.0f;
    float base_fade_height = 60.0f;
    float erosion_feature_size = 60.0f;
    uint32_t erosion_octaves = 4;
    float erosion_amplitude = 35.0f;
    float edge = 25.0f;                    // field value that maps to density 1
    uint32_t seed = 1;                     // erosion noise seed
};

struct VolumeDesc {
    simd_float3 bounds_min{-500, 0, -500};
    simd_float3 bounds_max{500, 800, 500};
    simd_uint3 grid{256, 192, 256};
    std::string density = "cumulus";     // "cumulus" | "constant"
    CloudDesc cloud;                     // only for density == "cumulus"
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
    // density, analytic, neural: the shared march (MarchParams)
    float step_scale = 1.0f;            // step = voxel_min_extent * step_scale
    uint32_t max_steps = 512;
    float min_transmittance = 0.01f;    // 0 disables early termination
    bool dither = true;
    bool empty_skip = true;
    // analytic
    uint32_t shadow_steps = 6;
    float shadow_distance = 0.0f;       // m; 0 = to the AABB exit
    uint32_t ambient_steps = 4;
    uint32_t octaves = 3;
    float ms_a = 0.4f;
    float ms_b = 0.35f;
    float ms_c = 0.1f;
    bool ambient = true;
    // pt
    uint32_t spp = 4096;
    uint32_t seed = 1;
    uint32_t max_bounces = 0;  // 0 = unlimited
    uint32_t rr_start_bounce = 16;
    // neural
    std::string model;              // path to .cnrm, relative to the working directory
    std::string inference = "scalar";  // "scalar" (hand-written MSL) | "shader_ml" (matmul2d) | "ml_encoder"
};

// The MarchParams for config (march fields only).
MarchParams makeMarchParams(const RenderConfig& config);

// Deterministic random cumulus for training and held-out shapes: lobe layout, shear, anvil and
// erosion parameters derived from `seed` (std::mt19937_64), sized to fit `volume` bounds.
CloudDesc makeRandomCumulus(uint32_t seed, const VolumeDesc& volume);

std::optional<Scene> loadScene(const std::string& path, std::string* error);
bool saveScene(const Scene& scene, const std::string& path, std::string* error);
std::optional<RenderConfig> loadRenderConfig(const std::string& path, std::string* error);
bool saveRenderConfig(const RenderConfig& config, const std::string& path, std::string* error);

// Builds the per-frame GPU constants for one camera. mip_levels is the full chain of
// volume.grid; voxel_min_extent = min((bounds_max - bounds_min) / grid).
SceneGPU makeSceneGPU(const Scene& scene, const CameraDesc& camera);

// Bake parameters derived from scene.volume.
DensityBakeParams makeDensityBakeParams(const Scene& scene);
