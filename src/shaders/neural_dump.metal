// Training-target generation for the neural shader (the article's camera-style sampling).
// Concatenated after neural_common.metal.
//
// Record r of a shape: rng stream (seed, r, 0) shoots camera rays from a uniform point on the
// AABB's bounding sphere toward a uniform point inside the AABB and delta-tracks each to its FIRST
// real collision x (at most max_tries rays; otherwise the record is invalid and dropped on the
// host); d = that ray's direction. Target f = mean over `paths` of
//   hg(dot(sun, d), g) * T_sun(x)            (sun single scattering, ratio tracking)
//   + random_walk(Ray(x, w ~ HG(d, g))).L     (everything that arrives at x from w)
// with unit sources: the full in-scattered radiance at x toward -d (pre-albedo), the function the
// network learns. Each record is split over `chunks` threads (path rng stream (seed, r, 1 + chunk));
// every thread recomputes x and d from the record stream.

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
    device float* rec = records + local * CNR_DUMP_RECORD_STRIDE;
    device float* part = partials + tid * CNR_DUMP_PARTIAL_STRIDE;

    float3 extent = s.bounds_max - s.bounds_min;
    float3 center = 0.5f * (s.bounds_min + s.bounds_max);
    float radius = 0.5f * length(extent);
    Rng rng = rng_init(p.seed, record, 0u);
    float3 x = float3(0.0f);
    float3 d = float3(0.0f, 0.0f, 1.0f);
    bool found = false;
    for (uint k = 0u; k < p.max_tries && !found; ++k) {
        float u0 = rng_float(rng);
        float u1 = rng_float(rng);
        float u2 = rng_float(rng);
        float u3 = rng_float(rng);
        float u4 = rng_float(rng);
        float cz = 1.0f - 2.0f * u0;
        float sz = sqrt(max(0.0f, 1.0f - cz * cz));
        float phi = 2.0f * kPi * u1;
        Ray r;
        r.o = center + radius * float3(sz * cos(phi), sz * sin(phi), cz);
        float3 target = s.bounds_min + float3(u2, u3, u4) * extent;
        float3 dir = target - r.o;
        if (length_squared(dir) <= 0.0f) {
            continue;
        }
        r.d = normalize(dir);
        float t0, t1, t;
        if (intersect_aabb(r, s.bounds_min, s.bounds_max, t0, t1) && delta_track(density, s, r, t0, t1, rng, t)) {
            x = r.o + t * r.d;
            d = r.d;
            found = true;
        }
    }
    if (!found) {
        if (chunk == 0u) {
            for (uint k = 0u; k < CNR_DUMP_RECORD_STRIDE; ++k) {
                rec[k] = 0.0f;
            }
        }
        for (uint k = 0u; k < CNR_DUMP_PARTIAL_STRIDE; ++k) {
            part[k] = 0.0f;
        }
        return;
    }

    if (chunk == 0u) {
        float desc[CNR_NN_IN];
        build_descriptor(density, s, x, d, desc);
        for (uint k = 0u; k < uint(CNR_NN_IN); ++k) {
            rec[k] = desc[k];
        }
        rec[CNR_NN_IN] = 1.0f;
    }

    float phase_sun = hg_eval(dot(s.sun_direction, d), s.hg_g);
    uint begin = chunk * p.paths_per_thread;
    uint end = min(begin + p.paths_per_thread, p.paths);
    Rng prng = rng_init(p.seed, record, 1u + chunk);
    float sun = 0.0f;
    float sky = 0.0f;
    float ground = 0.0f;
    uint cap_hits = 0u;
    for (uint i = begin; i < end; ++i) {
        sun += phase_sun * ratio_tracking_transmittance(density, s, x, s.sun_direction, prng);
        // Draw explicitly in order: argument evaluation order is unspecified.
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
