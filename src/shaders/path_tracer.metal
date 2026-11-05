// Reference path tracer (docs/architecture.md §6): one thread per pixel, one sample per dispatch,
// progressive running mean m += (x - m) / n in the accumulation texture.
//
// counters (uint pairs forming little-endian 64-bit sums, zeroed by the host on reset):
//   [0], [1] = paths that hit the bounce cap; [2], [3] = total real collisions (bounces).

// 64-bit atomic sum from two 32-bit words: every add that wraps the low word carries once.
inline void pt_add_u64(device atomic_uint* lohi, uint v) {
    if (v == 0u) {
        return;
    }
    uint old = atomic_fetch_add_explicit(&lohi[0], v, memory_order_relaxed);
    if (old + v < old) {
        atomic_fetch_add_explicit(&lohi[1], 1u, memory_order_relaxed);
    }
}

kernel void pt_render(constant SceneGPU& s [[buffer(0)]],
                      constant PathTracerParams& p [[buffer(1)]],
                      device atomic_uint* counters [[buffer(2)]],
                      texture3d<float> density [[texture(0)]],
                      texture2d<float, access::read_write> accum [[texture(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const bool active =
        gid.x < s.width && gid.y < s.height && gid.x < accum.get_width() && gid.y < accum.get_height();
    uint cap = 0u;
    uint bounces = 0u;
    if (active) {
        Rng rng = rng_init(p.seed, gid.y * s.width + gid.x, p.sample_index);
        // Box filter: uniform jitter inside the pixel. Draw explicitly in order.
        float u0 = rng_float(rng);
        float u1 = rng_float(rng);
        Ray r = camera_ray(s, float2(gid) + float2(u0, u1));
        WalkResult w = random_walk(density, s, r, rng, p.max_bounces, p.rr_start_bounce, p.bounce_cap);
        float3 x = combine_lights(s, w.L);
        float3 m = x;
        if (p.sample_index > 0u) {
            float3 prev = accum.read(gid).rgb;
            m = prev + (x - prev) / float(p.sample_index + 1u);
        }
        accum.write(float4(m, 1.0f), gid);
        cap = w.hit_cap ? 1u : 0u;
        bounces = w.bounces;
    }
    // One atomic per SIMD group instead of one per thread (32 lanes * cap 4096 fits in 32 bits).
    uint cap_sum = simd_sum(cap);
    uint bounce_sum = simd_sum(bounces);
    if (simd_is_first()) {
        pt_add_u64(counters + 0, cap_sum);
        pt_add_u64(counters + 2, bounce_sum);
    }
}
