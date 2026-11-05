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

// Exact single scattering of the sun: Ls = p(cos) exp(-tau_sun) E_sun with the AABB-exit optical depth.
struct SunShade {
    texture3d<float> tex;
    constant SceneGPU* s;
    uint shadow_steps;

    float3 operator()(float3 x, float3 d) {
        float tau = march_optical_depth(tex, *s, x, s->sun_direction, shadow_steps);
        return hg_eval(dot(s->sun_direction, d), s->hg_g) * precise::exp(-tau) * s->sun_irradiance;
    }
};

// rays[2i] = (origin, jitter in (0, 1]), rays[2i+1] = (unit direction, shadow_steps);
// out[i] = march_view_ray with SunShade, settings from make_march_settings(p) and the jitter overridden.
kernel void test_march_single_scatter(constant SceneGPU& s [[buffer(0)]],
                                      constant MarchParams& p [[buffer(1)]],
                                      device const float4* rays [[buffer(2)]],
                                      device float4* out [[buffer(3)]],
                                      texture3d<float> density [[texture(0)]],
                                      uint gid [[thread_position_in_grid]]) {
    float4 o = rays[2u * gid];
    float4 d = rays[2u * gid + 1u];
    Ray r;
    r.o = o.xyz;
    r.d = d.xyz;
    MarchSettings m = make_march_settings(s, p, uint2(gid, 0u));
    m.jitter = o.w;
    SunShade shade;
    shade.tex = density;
    shade.s = &s;
    shade.shadow_steps = uint(d.w);
    out[gid] = march_view_ray(density, s, r, m, shade);
}

// out[i] = (jitter, step) of make_march_settings(p) for pixel (i % width, i / width).
kernel void test_march_settings(constant SceneGPU& s [[buffer(0)]],
                                constant MarchParams& p [[buffer(1)]],
                                device float2* out [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    MarchSettings m = make_march_settings(s, p, uint2(gid % s.width, gid / s.width));
    out[gid] = float2(m.jitter, m.step);
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
