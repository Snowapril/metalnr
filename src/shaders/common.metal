// Shared physics for every cloudnr kernel. Concatenated after src/shared_types.h
// (which provides <metal_stdlib>, `using namespace metal` and the GPU structs).
// This is the only copy of the camera, medium, phase, RNG and tracking code.

constexpr sampler kDensitySampler(filter::linear, mip_filter::linear, address::clamp_to_edge);

constant float kPi = 3.14159265358979323846f;
constant float kInv4Pi = 0.07957747154594766788f;

struct Ray {
    float3 o;
    float3 d;  // unit; tracing direction (opposite to light flow)
};

// 1 - exp(-x) without cancellation for small x (no expm1 in MSL).
inline float one_minus_exp_neg(float x) {
    if (x < 0.1f) {
        return x * (1.0f - x * (0.5f - x * (1.0f / 6.0f - x * (1.0f / 24.0f - x * (1.0f / 120.0f)))));
    }
    return 1.0f - precise::exp(-x);
}

// pcg_hash (Jarzynski & Olano 2020): 32-bit integer hash used for seeding and noise lattices.
inline uint pcg_hash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline Ray camera_ray(constant SceneGPU& s, float2 pixel) {
    float nx = (2.0f * pixel.x / float(s.width) - 1.0f) * s.tan_half_vfov * s.aspect;
    float ny = (1.0f - 2.0f * pixel.y / float(s.height)) * s.tan_half_vfov;
    Ray r;
    r.o = s.cam_position;
    r.d = normalize(s.cam_forward + nx * s.cam_right + ny * s.cam_up);
    return r;
}

inline bool intersect_aabb(Ray r, float3 bmin, float3 bmax, thread float& t0, thread float& t1) {
    // Replace zero components by a tiny signed value so the slab test stays finite under fast math.
    float3 dd = select(r.d, copysign(float3(1e-20f), r.d), abs(r.d) < 1e-20f);
    float3 inv = 1.0f / dd;
    float3 ta = (bmin - r.o) * inv;
    float3 tb = (bmax - r.o) * inv;
    float3 tn = min(ta, tb);
    float3 tf = max(ta, tb);
    t0 = max(max(tn.x, tn.y), max(tn.z, 0.0f));
    t1 = min(min(tf.x, tf.y), tf.z);
    return t1 > t0;
}

inline float density_at(texture3d<float> tex, constant SceneGPU& s, float3 p, float lod = 0.0f) {
    if (any(p < s.bounds_min) || any(p > s.bounds_max)) {
        return 0.0f;
    }
    float3 uvw = (p - s.bounds_min) / (s.bounds_max - s.bounds_min);
    return tex.sample(kDensitySampler, uvw, level(lod)).r;
}

inline float sigma_t_at(texture3d<float> tex, constant SceneGPU& s, float3 p) {
    return s.sigma_t_max * density_at(tex, s, p);
}

inline float hg_eval(float cos_theta, float g) {
    float g2 = g * g;
    float denom = max(1.0f + g2 - 2.0f * g * cos_theta, 1e-12f);
    return kInv4Pi * (1.0f - g2) / (denom * sqrt(denom));
}

inline float3 hg_sample(float3 d, float g, float2 u) {
    float cos_theta;
    if (abs(g) < 1e-3f) {
        cos_theta = 1.0f - 2.0f * u.x;
    } else {
        // Inverse CDF of HG in cos(d, w): u = 0 -> -1, u = 1 -> +1, E[cos] = g.
        float sq = (1.0f - g * g) / (1.0f - g + 2.0f * g * u.x);
        cos_theta = (1.0f + g * g - sq * sq) / (2.0f * g);
    }
    cos_theta = clamp(cos_theta, -1.0f, 1.0f);
    float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = 2.0f * kPi * u.y;
    // Branchless orthonormal basis around d (Duff et al. 2017).
    float sgn = copysign(1.0f, d.z);
    float a = -1.0f / (sgn + d.z);
    float b = d.x * d.y * a;
    float3 b1 = float3(1.0f + sgn * d.x * d.x * a, sgn * b, -sgn * d.x);
    float3 b2 = float3(b, sgn + d.y * d.y * a, -d.y);
    return normalize(sin_theta * cos(phi) * b1 + sin_theta * sin(phi) * b2 + cos_theta * d);
}

struct Rng {
    uint state;
    uint inc;
};

// PCG32 with 32-bit state (PCG-RXS-M-XS 32/32): LCG step + output permutation of the old state.
inline uint rng_next(thread Rng& r) {
    uint old = r.state;
    r.state = old * 747796405u + r.inc;
    uint word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}

inline Rng rng_init(uint seed, uint stream, uint sample_index) {
    uint h = pcg_hash(seed + pcg_hash(stream + pcg_hash(sample_index)));
    Rng r;
    r.state = 0u;
    r.inc = (pcg_hash(h ^ 0x9E3779B9u) << 1u) | 1u;
    rng_next(r);
    r.state += h;
    rng_next(r);
    return r;
}

inline float rng_float(thread Rng& r) {
    return float(rng_next(r) >> 8u) * (1.0f / 16777216.0f);
}

inline float3 env_radiance(constant SceneGPU& s, float3 d) {
    return d.y > 0.0f ? s.sky_radiance : s.ground_radiance;
}

struct LightCoeffs {
    float sun;
    float sky;
    float ground;
};

inline float3 combine_lights(constant SceneGPU& s, LightCoeffs c) {
    return c.sun * s.sun_irradiance + c.sky * s.sky_radiance + c.ground * s.ground_radiance;
}

inline float march_optical_depth(texture3d<float> tex, constant SceneGPU& s, float3 p, float3 d, uint steps) {
    Ray r;
    r.o = p;
    r.d = d;
    float t0, t1;
    if (steps == 0u || !intersect_aabb(r, s.bounds_min, s.bounds_max, t0, t1)) {
        return 0.0f;
    }
    float dt = (t1 - t0) / float(steps);
    float sum = 0.0f;
    for (uint i = 0u; i < steps; ++i) {
        sum += sigma_t_at(tex, s, p + (t0 + (float(i) + 0.5f) * dt) * d);
    }
    return sum * dt;
}

inline float ratio_tracking_transmittance(texture3d<float> tex, constant SceneGPU& s, float3 p, float3 d,
                                          thread Rng& rng) {
    Ray r;
    r.o = p;
    r.d = d;
    float t0, t1;
    if (s.sigma_t_max <= 0.0f || !intersect_aabb(r, s.bounds_min, s.bounds_max, t0, t1)) {
        return 1.0f;
    }
    float T = 1.0f;
    float t = t0;
    for (;;) {
        t -= log(1.0f - rng_float(rng)) / s.sigma_t_max;
        if (t >= t1) {
            break;
        }
        T *= max(0.0f, 1.0f - sigma_t_at(tex, s, p + t * d) / s.sigma_t_max);
        if (T <= 0.0f) {
            break;  // the product stays 0; stopping is exact
        }
    }
    return T;
}

inline bool delta_track(texture3d<float> tex, constant SceneGPU& s, Ray r, float t0, float t1, thread Rng& rng,
                        thread float& t) {
    t = t0;
    if (s.sigma_t_max <= 0.0f) {
        t = t1;
        return false;
    }
    for (;;) {
        t -= log(1.0f - rng_float(rng)) / s.sigma_t_max;
        if (t >= t1) {
            t = t1;
            return false;
        }
        if (rng_float(rng) * s.sigma_t_max < sigma_t_at(tex, s, r.o + t * r.d)) {
            return true;
        }
    }
}

struct WalkResult {
    LightCoeffs L;
    uint bounces;
    bool hit_cap;
};

inline WalkResult random_walk(texture3d<float> tex, constant SceneGPU& s, Ray r, thread Rng& rng,
                              uint max_bounces, uint rr_start_bounce, uint bounce_cap) {
    WalkResult res;
    res.L.sun = 0.0f;
    res.L.sky = 0.0f;
    res.L.ground = 0.0f;
    res.bounces = 0u;
    res.hit_cap = false;
    float beta = 1.0f;
    for (;;) {
        float t0, t1, t;
        if (!intersect_aabb(r, s.bounds_min, s.bounds_max, t0, t1) || !delta_track(tex, s, r, t0, t1, rng, t)) {
            if (r.d.y > 0.0f) {
                res.L.sky += beta;
            } else {
                res.L.ground += beta;
            }
            break;
        }
        r.o = r.o + t * r.d;
        res.bounces++;
        beta *= s.albedo;
        res.L.sun += beta * hg_eval(dot(s.sun_direction, r.d), s.hg_g) *
                     ratio_tracking_transmittance(tex, s, r.o, s.sun_direction, rng);
        if (beta <= 0.0f) {
            break;  // albedo 0: every later contribution is exactly 0
        }
        if (max_bounces != 0u && res.bounces == max_bounces) {
            break;
        }
        if (rr_start_bounce > 0u && res.bounces >= rr_start_bounce) {
            if (rng_float(rng) >= 0.95f) {
                break;
            }
            beta /= 0.95f;
        }
        if (res.bounces == bounce_cap) {
            res.hit_cap = true;
            break;
        }
        // Draw explicitly in order: argument evaluation order is unspecified.
        float u0 = rng_float(rng);
        float u1 = rng_float(rng);
        r.d = hg_sample(r.d, s.hg_g, float2(u0, u1));
    }
    return res;
}

// ---- Shared view march (the article's ray march; analytic and neural renderers) ----

// Coarse mip level probed by the empty-space skip: one texel covers 8^3 finest voxels.
constant float kEmptySkipLod = 3.0f;

struct MarchSettings {
    float step;               // m, nominal segment length
    float min_transmittance;  // stop once T falls below this (0 disables)
    uint max_steps;           // segments, including skipped ones
    uint flags;               // CNR_MARCH_*
    float jitter;             // (0, 1]: length of the first segment in steps
};

// Interleaved gradient noise (Jimenez 2014), in [0, 1).
inline float interleaved_gradient_noise(float2 pixel) {
    return fract(52.9829189f * fract(dot(pixel, float2(0.06711056f, 0.00583715f))));
}

inline MarchSettings make_march_settings(constant SceneGPU& s, MarchParams p, uint2 pixel) {
    MarchSettings m;
    m.step = s.voxel_min_extent * p.step_scale;
    m.min_transmittance = p.min_transmittance;
    m.max_steps = p.max_steps;
    m.flags = p.flags;
    m.jitter = (p.flags & CNR_MARCH_DITHER) != 0u ? 1.0f - interleaved_gradient_noise(float2(pixel)) : 1.0f;
    return m;
}

// Front-to-back march along r through the AABB. Segment k spans [b_k, b_{k+1}] with b_0 = t0 and
// b_k = t0 + (jitter + k - 1) * step (k >= 1), the last one clipped at t1; each segment is sampled at its
// midpoint with its own length dt (exact for a homogeneous medium, dithered otherwise). shade(x, d) returns
// the in-scattered radiance at x toward -d before albedo; each shaded segment adds the energy-conserving
// T * albedo * Ls * (1 - exp(-sigma dt)) (Hillaire 2015). Returns float4(L + T * L_env, 1 - T).
//
// EMPTY_SKIP: a lod-kEmptySkipLod sample <= 0 at the segment start proves the 2^3 coarse texels around it are
// empty, i.e. the field is 0 within 0.5 * 8 voxels; trilinear fine samples are then 0 up to 3.5 voxels away.
// The skip jumps over the whole segments whose midpoints lie within that radius, so the sample positions stay
// on the same lattice and the result equals the unskipped march (a skip counts as one step).
template <typename Shade>
inline float4 march_view_ray(texture3d<float> tex, constant SceneGPU& s, Ray r, MarchSettings m,
                             thread Shade& shade) {
    float3 env = env_radiance(s, r.d);
    float t0, t1;
    if (m.step <= 0.0f || !intersect_aabb(r, s.bounds_min, s.bounds_max, t0, t1)) {
        return float4(env, 0.0f);
    }
    // Segments k .. k+n-1 have midpoints within (n - 0.5) * step of b_k; pick the largest n keeping them within
    // 3.5 voxels (n = 4 at step_scale 1, i.e. the 0.5 * 8 voxel jump). n = 0 (step > 7 voxels) disables the skip.
    uint skip_segments = 0u;
    if ((m.flags & CNR_MARCH_EMPTY_SKIP) != 0u && s.mip_levels > uint(kEmptySkipLod)) {
        skip_segments = uint(floor((0.5f * 8.0f - 0.5f) * s.voxel_min_extent / m.step + 0.5f + 1e-3f));
    }
    float T = 1.0f;
    float3 L = float3(0.0f);
    uint k = 0u;
    for (uint i = 0u; i < m.max_steps; ++i) {
        float a = k == 0u ? t0 : t0 + (m.jitter + float(k - 1u)) * m.step;
        if (a >= t1) {
            break;
        }
        if (skip_segments > 0u && density_at(tex, s, r.o + a * r.d, kEmptySkipLod) <= 0.0f) {
            k += skip_segments;
            continue;
        }
        float b = min(t0 + (m.jitter + float(k)) * m.step, t1);
        float dt = b - a;
        float3 x = r.o + (a + 0.5f * dt) * r.d;
        float sig = sigma_t_at(tex, s, x);
        if (sig > 0.0f) {
            float3 Ls = shade(x, r.d);
            L += T * s.albedo * Ls * one_minus_exp_neg(sig * dt);
            T *= precise::exp(-sig * dt);
            if (T < m.min_transmittance) {
                break;
            }
        }
        ++k;
    }
    return float4(L + T * env, 1.0f - T);
}
