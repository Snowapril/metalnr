// Test-only kernels for tests/test_analytic.mm. Concatenated after src/shared_types.h and
// src/shaders/common.metal.

// rays[2i] = (origin, steps), rays[2i+1] = (unit direction, unused); out[i] = optical depth.
kernel void test_optical_depth(constant SceneGPU& s [[buffer(0)]],
                               device const float4* rays [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               texture3d<float> density [[texture(0)]],
                               uint gid [[thread_position_in_grid]]) {
    float4 o = rays[2u * gid];
    float4 d = rays[2u * gid + 1u];
    out[gid] = march_optical_depth(density, s, o.xyz, d.xyz, uint(o.w));
}

struct ZeroLRest {
    float3 operator()(float3, float3, float) { return float3(0.0f); }
};

// rays[2i] = (origin, primary_steps), rays[2i+1] = (unit direction, shadow_steps);
// out[i] = march_view_ray with L_rest = 0 (single scattering of the sun only).
kernel void test_march_single_scatter(constant SceneGPU& s [[buffer(0)]],
                                      device const float4* rays [[buffer(1)]],
                                      device float4* out [[buffer(2)]],
                                      texture3d<float> density [[texture(0)]],
                                      uint gid [[thread_position_in_grid]]) {
    float4 o = rays[2u * gid];
    float4 d = rays[2u * gid + 1u];
    Ray r;
    r.o = o.xyz;
    r.d = d.xyz;
    ZeroLRest lrest;
    out[gid] = float4(march_view_ray(density, s, r, uint(o.w), uint(d.w), lrest), 0.0f);
}

// Midpoint quadrature of hg_eval over cos in [-1, 1] with grid.x * 64 samples; thread (x, y)
// sums 64 consecutive samples for g = gs[y]. out = (sum p, sum p*cos), unscaled by the step.
kernel void test_hg_integral(device const float* gs [[buffer(0)]],
                             device float2* out [[buffer(1)]],
                             uint2 gid [[thread_position_in_grid]],
                             uint2 grid [[threads_per_grid]]) {
    const uint kChunk = 64u;
    float g = gs[gid.y];
    float h = 2.0f / float(grid.x * kChunk);
    float sum = 0.0f;
    float sum_cos = 0.0f;
    for (uint j = 0u; j < kChunk; ++j) {
        float c = -1.0f + (float(gid.x * kChunk + j) + 0.5f) * h;
        float p = hg_eval(c, g);
        sum += p;
        sum_cos += p * c;
    }
    out[gid.y * grid.x + gid.x] = float2(sum, sum_cos);
}
