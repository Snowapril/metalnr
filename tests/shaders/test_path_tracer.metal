// Path tracer test kernels. Concatenated after src/shared_types.h and src/shaders/common.metal.

// Host mirror in tests/test_path_tracer.mm (48 bytes).
struct WalkTestParams {
    float4 origin;  // xyz
    float4 dir;     // xyz, unit
    uint seed;
    uint max_bounces;
    uint rr_start_bounce;
    uint bounce_cap;
};

// One independent random_walk estimate per thread for the same exact ray.
// out[i] = (combine_lights(L).rgb, hit_cap ? 1 : 0).
kernel void test_random_walk(constant SceneGPU& s [[buffer(0)]],
                             constant WalkTestParams& p [[buffer(1)]],
                             device float4* out [[buffer(2)]],
                             texture3d<float> density [[texture(0)]],
                             uint gid [[thread_position_in_grid]]) {
    Rng rng = rng_init(p.seed, gid, 0u);
    Ray r;
    r.o = p.origin.xyz;
    r.d = p.dir.xyz;
    WalkResult w = random_walk(density, s, r, rng, p.max_bounces, p.rr_start_bounce, p.bounce_cap);
    out[gid] = float4(combine_lights(s, w.L), w.hit_cap ? 1.0f : 0.0f);
}

// Host mirror in tests/test_path_tracer.mm (32 bytes).
struct HgTestParams {
    float4 d;  // xyz, unit
    float g;
    uint seed;
    uint _pad[2];
};

// out[i] = (dot(d, w), length(w)) for w = hg_sample(d, g, u), u from an independent stream per thread.
kernel void test_hg_sample(constant HgTestParams& p [[buffer(0)]],
                           device float2* out [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    Rng rng = rng_init(p.seed, gid, 0u);
    float u0 = rng_float(rng);
    float u1 = rng_float(rng);
    float3 w = hg_sample(p.d.xyz, p.g, float2(u0, u1));
    out[gid] = float2(dot(p.d.xyz, w), length(w));
}
