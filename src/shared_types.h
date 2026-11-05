// POD structs shared by host C++ and MSL. The host prepends this file to every
// shader source string (see MetalContext::makeComputePipeline), so it must stay
// valid in both languages. float3 is 16 bytes / 16-aligned on both sides
// (simd_float3 on the host), so field order below avoids implicit padding.
#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
typedef float3 cnr_float3;
typedef uint cnr_uint;
#define CNR_STATIC_ASSERT(cond, msg)
#else
#include <simd/simd.h>
#include <cstdint>
typedef simd_float3 cnr_float3;
typedef uint32_t cnr_uint;
#define CNR_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#endif

// Density field kinds (DensityBakeParams::density_type).
#define CNR_DENSITY_PROCEDURAL 0u
#define CNR_DENSITY_CONSTANT 1u

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

struct DensityBakeParams {
    cnr_float3 bounds_min;
    cnr_float3 bounds_max;
    cnr_uint grid_x;
    cnr_uint grid_y;
    cnr_uint grid_z;
    cnr_uint density_type;      // CNR_DENSITY_*
    cnr_uint density_seed;
    float coverage;
    float erosion;
    float constant_density;     // used when density_type == CNR_DENSITY_CONSTANT
};
CNR_STATIC_ASSERT(sizeof(DensityBakeParams) == 64, "DensityBakeParams layout");

struct DensityViewParams {
    cnr_uint steps;
    cnr_uint _pad[3];
};
CNR_STATIC_ASSERT(sizeof(DensityViewParams) == 16, "DensityViewParams layout");

struct AnalyticParams {
    cnr_uint primary_steps;
    cnr_uint shadow_steps;
    cnr_uint ambient_steps;
    cnr_uint octaves;           // 1 = single scattering only
    float ms_a;
    float ms_b;
    float ms_c;
    cnr_uint flags;             // CNR_ANALYTIC_*
};
CNR_STATIC_ASSERT(sizeof(AnalyticParams) == 32, "AnalyticParams layout");

struct PathTracerParams {
    cnr_uint seed;
    cnr_uint sample_index;      // 0-based index of the sample written by this dispatch
    cnr_uint max_bounces;       // 0 = unlimited (bounce_cap still applies)
    cnr_uint rr_start_bounce;
    cnr_uint bounce_cap;        // safety cap, 4096
    cnr_uint _pad[3];
};
CNR_STATIC_ASSERT(sizeof(PathTracerParams) == 32, "PathTracerParams layout");
