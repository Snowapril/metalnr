// Training-target generation for the neural L_rest (docs/architecture.md §8).
// Concatenated after neural_common.metal.
//
// Record r of a scene: rng stream (seed, r, 0) rejection-samples x uniformly in the AABB
// until density > 0.01 (max_tries attempts; otherwise the record is invalid), then draws a
// uniform view direction d. Target f = mean over `paths` of random_walk from x along
// w ~ HG(d, g) with unit sources: the full in-scattered radiance at x toward -d minus the
// direct sun term (no sun NEE at x itself), i.e. exactly L_rest in coefficient form.
// Each record is split over `chunks` threads (path rng stream (seed, r, 1 + chunk)) to
// bound per-thread work; every thread recomputes x and d from the record stream.

kernel void neural_dump(constant SceneGPU& s [[buffer(0)]],
                        constant DumpParams& p [[buffer(1)]],
                        device float* records [[buffer(2)]],
                        device float* partials [[buffer(3)]],
                        texture3d<float> density [[texture(0)]],
                        uint tid [[thread_position_in_grid]]) {
    uint local = tid / p.chunks;
    uint chunk = tid - local * p.chunks;
    if (local >= p.record_count) {
        return;
    }
    uint record = p.record_offset + local;
    device float* rec = records + local * (uint(NN_IN) + 1u);
    device float* part = partials + tid * CNR_DUMP_PARTIAL_STRIDE;

    Rng rng = rng_init(p.seed, record, 0u);
    float3 x = float3(0.0f);
    bool found = false;
    float3 extent = s.bounds_max - s.bounds_min;
    for (uint k = 0u; k < p.max_tries; ++k) {
        float u0 = rng_float(rng);
        float u1 = rng_float(rng);
        float u2 = rng_float(rng);
        float3 c = s.bounds_min + float3(u0, u1, u2) * extent;
        if (density_at(density, s, c) > 0.01f) {
            x = c;
            found = true;
            break;
        }
    }
    if (!found) {
        if (chunk == 0u) {
            for (uint k = 0u; k <= uint(NN_IN); ++k) {
                rec[k] = 0.0f;
            }
        }
        for (uint k = 0u; k < CNR_DUMP_PARTIAL_STRIDE; ++k) {
            part[k] = 0.0f;
        }
        return;
    }

    float v0 = rng_float(rng);
    float v1 = rng_float(rng);
    float cz = 1.0f - 2.0f * v0;
    float sz = sqrt(max(0.0f, 1.0f - cz * cz));
    float phi = 2.0f * kPi * v1;
    float3 d = float3(sz * cos(phi), sz * sin(phi), cz);

    if (chunk == 0u) {
        float tau_sun = march_optical_depth(density, s, x, s.sun_direction, p.shadow_steps);
        float in[NN_IN];
        build_descriptor(density, s, x, d, tau_sun, in);
        for (uint k = 0u; k < uint(NN_IN); ++k) {
            rec[k] = in[k];
        }
        rec[NN_IN] = 1.0f;
    }

    uint begin = chunk * p.paths_per_thread;
    uint end = min(begin + p.paths_per_thread, p.paths);
    Rng prng = rng_init(p.seed, record, 1u + chunk);
    float sun = 0.0f;
    float sky = 0.0f;
    float ground = 0.0f;
    uint cap_hits = 0u;
    for (uint i = begin; i < end; ++i) {
        float u0 = rng_float(prng);
        float u1 = rng_float(prng);
        Ray r;
        r.o = x;
        r.d = hg_sample(d, s.hg_g, float2(u0, u1));
        WalkResult w = random_walk(density, s, r, prng, 0u, 16u, 4096u);
        sun += w.L.sun;
        sky += w.L.sky;
        ground += w.L.ground;
        cap_hits += w.hit_cap ? 1u : 0u;
    }
    part[0] = sun;
    part[1] = sky;
    part[2] = ground;
    part[3] = float(cap_hits);
}
