// POD structs shared by host C++ and MSL. The host prepends this file to every
// shader source string (see MetalContext::makeComputePipeline), so it must stay
// valid in both languages. float3 is 16 bytes / 16-aligned on both sides
// (simd_float3 on the host), so field order below avoids implicit padding.
#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
typedef float3 cnr_float3;
typedef float4 cnr_float4;
typedef uint cnr_uint;
#define CNR_STATIC_ASSERT(cond, msg)
#else
#include <simd/simd.h>
#include <cstdint>
typedef simd_float3 cnr_float3;
typedef simd_float4 cnr_float4;
typedef uint32_t cnr_uint;
#define CNR_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#endif

// Density field kinds (DensityBakeParams::density_type).
#define CNR_DENSITY_CUMULUS 0u
#define CNR_DENSITY_CONSTANT 1u

// Maximum number of sphere lobes of a cumulus (DensityBakeParams::lobes).
#define CNR_MAX_LOBES 16

// MarchParams::flags
#define CNR_MARCH_DITHER 1u      // offset each pixel's first sample by up to one step
#define CNR_MARCH_EMPTY_SKIP 2u  // jump over regions a coarse mip proves empty

// AnalyticParams::flags
#define CNR_ANALYTIC_AMBIENT 1u

// Per-frame scene constants, identical for every renderer.
struct SceneGPU {
    cnr_float3 cam_position;    // m
    cnr_float3 cam_forward;     // unit, normalize(target - position)
    cnr_float3 cam_right;       // unit, normalize(cross(forward, +Y))
    cnr_float3 cam_up;          // unit, cross(right, forward)
    cnr_float3 bounds_min;      // m
    cnr_float3 bounds_max;      // m
    cnr_float3 sun_direction;   // unit, points toward the sun
    cnr_float3 sun_irradiance;  // RGB irradiance perpendicular to the sun direction
    cnr_float3 sky_radiance;    // RGB radiance for directions with y > 0
    cnr_float3 ground_radiance; // RGB radiance for directions with y <= 0
    float tan_half_vfov;
    float aspect;               // width / height
    float exposure_ev;          // display only; never applied to linear output
    float sigma_t_max;          // 1/m
    float albedo;
    float hg_g;
    float voxel_min_extent;     // m, min over axes of (bounds extent / grid)
    float _pad0;
    cnr_uint width;
    cnr_uint height;
    cnr_uint mip_levels;
    cnr_uint _pad1;
};
CNR_STATIC_ASSERT(sizeof(SceneGPU) == 208, "SceneGPU layout");

// Cumulus shape (docs/architecture.md §4): a signed field, positive inside, evaluated in cloud-local
// metres with the origin at base_center (the centre of the flat base). density = saturate(field / edge).
struct DensityBakeParams {
    cnr_float3 bounds_min;
    cnr_float3 bounds_max;
    cnr_float3 base_center;     // world position of the cloud-local origin
    cnr_uint grid_x;
    cnr_uint grid_y;
    cnr_uint grid_z;
    cnr_uint density_type;      // CNR_DENSITY_*
    cnr_uint seed;              // erosion noise seed
    cnr_uint lobe_count;        // <= CNR_MAX_LOBES
    cnr_uint erosion_octaves;
    float constant_density;     // used when density_type == CNR_DENSITY_CONSTANT
    float edge;                 // m; field value that maps to density 1
    float lobe_blend;           // m; smooth-max width welding the lobes
    float shear_x;              // horizontal offset per metre of height (wind shear)
    float shear_z;
    float anvil_height;         // m; anvil widening starts here
    float cloud_height;         // m; anvil widening is complete here
    float anvil_spread;         // m; extra radius at cloud_height
    float cloud_radius;         // m; reference radius for the anvil warp
    float base_softness;        // m; smooth-min width of the flat base
    float base_fade_height;     // m; erosion fades in over this height above the base
    float erosion_feature_size; // m
    float erosion_amplitude;    // m; added to the field
    cnr_float4 lobes[CNR_MAX_LOBES];  // xyz centre (cloud-local m), w radius (m)
};
CNR_STATIC_ASSERT(sizeof(DensityBakeParams) == 384, "DensityBakeParams layout");

// Ray-march settings shared by the analytic and neural renderers (the article's march).
struct MarchParams {
    float step_scale;           // step length = voxel_min_extent * step_scale
    float min_transmittance;    // stop once T falls below this (0 disables)
    cnr_uint max_steps;
    cnr_uint flags;             // CNR_MARCH_*
};
CNR_STATIC_ASSERT(sizeof(MarchParams) == 16, "MarchParams layout");

struct DensityViewParams {
    MarchParams march;
};
CNR_STATIC_ASSERT(sizeof(DensityViewParams) == 16, "DensityViewParams layout");

struct AnalyticParams {
    MarchParams march;
    cnr_uint shadow_steps;      // sun shadow march samples
    cnr_uint ambient_steps;     // samples straight up and straight down
    cnr_uint octaves;           // Wrenninge octaves including i = 0; 1 = single scattering only
    cnr_uint flags;             // CNR_ANALYTIC_*
    float ms_a;                 // extinction falloff per octave
    float ms_b;                 // brightness falloff per octave
    float ms_c;                 // phase anisotropy falloff per octave
    float shadow_distance;      // m; sun shadow march length (0 = to the AABB exit)
};
CNR_STATIC_ASSERT(sizeof(AnalyticParams) == 48, "AnalyticParams layout");

struct PathTracerParams {
    cnr_uint seed;
    cnr_uint sample_index;      // 0-based index of the sample written by this dispatch
    cnr_uint max_bounces;       // 0 = unlimited (bounce_cap still applies)
    cnr_uint rr_start_bounce;
    cnr_uint bounce_cap;        // safety cap, 4096
    cnr_uint _pad[3];
};
CNR_STATIC_ASSERT(sizeof(PathTracerParams) == 32, "PathTracerParams layout");
