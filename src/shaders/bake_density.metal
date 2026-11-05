// Procedural / constant density bake into mip 0 of the r16Float 3D texture.
//
// Procedural field, fixed for scene version 1:
//   qn = (x - bounds_min) / max(bounds extent)      longest axis spans [0,1], noise stays isotropic
//   q  = kPerlinBaseFrequency * qn                  4 Perlin cycles across the longest axis
//   d  = saturate(remap(fbm_perlin(q), 1 - coverage, 1) * h(yn) * e(xn, zn) - erosion * fbm_worley(4 q))
//   fbm_perlin: 4 octaves of gradient noise, lacunarity 2, gain 0.5, mapped to [0,1]
//   fbm_worley: 3 octaves of Worley F1 distance, lacunarity 2, gain 0.5 (16, 32, 64 cells across)
//   h(yn) = smoothstep(0, 0.1, yn) * (1 - smoothstep(0.6, 1, yn))
//   e(xn, zn) = s(xn) * s(zn),  s(u) = smoothstep(0, 0.15, u) * (1 - smoothstep(0.85, 1, u))
//   xn, yn, zn = (x - bounds_min) / (bounds_max - bounds_min); h and e are 0 on all six AABB faces
// density_seed and the octave index are hashed into integer lattice offsets (integer hashing only,
// so the bake is deterministic and independent of thread order).

constant float kPerlinBaseFrequency = 4.0f;
constant float kWorleyFrequencyScale = 4.0f;
constant uint kPerlinOctaves = 4u;
constant uint kWorleyOctaves = 3u;
constant uint kWorleySaltBase = 16u;

inline uint lattice_hash(int3 c) {
    uint3 u = uint3(c);
    return pcg_hash(u.x + pcg_hash(u.y + pcg_hash(u.z)));
}

inline int3 seed_offset(uint seed, uint salt) {
    uint h = pcg_hash(seed ^ pcg_hash(salt));
    return int3(int(h & 1023u), int((h >> 10u) & 1023u), int((h >> 20u) & 1023u));
}

// Improved Perlin gradient selection (12 cube-edge gradients, 16 entries).
inline float grad_dot(uint h, float3 f) {
    uint k = h & 15u;
    float u = k < 8u ? f.x : f.y;
    float v = k < 4u ? f.y : ((k == 12u || k == 14u) ? f.x : f.z);
    return ((k & 1u) ? -u : u) + ((k & 2u) ? -v : v);
}

inline float perlin(float3 q, int3 off) {
    float3 fl = floor(q);
    int3 c = int3(fl) + off;
    float3 f = q - fl;
    float3 w = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);
    float n000 = grad_dot(lattice_hash(c + int3(0, 0, 0)), f - float3(0, 0, 0));
    float n100 = grad_dot(lattice_hash(c + int3(1, 0, 0)), f - float3(1, 0, 0));
    float n010 = grad_dot(lattice_hash(c + int3(0, 1, 0)), f - float3(0, 1, 0));
    float n110 = grad_dot(lattice_hash(c + int3(1, 1, 0)), f - float3(1, 1, 0));
    float n001 = grad_dot(lattice_hash(c + int3(0, 0, 1)), f - float3(0, 0, 1));
    float n101 = grad_dot(lattice_hash(c + int3(1, 0, 1)), f - float3(1, 0, 1));
    float n011 = grad_dot(lattice_hash(c + int3(0, 1, 1)), f - float3(0, 1, 1));
    float n111 = grad_dot(lattice_hash(c + int3(1, 1, 1)), f - float3(1, 1, 1));
    float nx00 = mix(n000, n100, w.x);
    float nx10 = mix(n010, n110, w.x);
    float nx01 = mix(n001, n101, w.x);
    float nx11 = mix(n011, n111, w.x);
    return mix(mix(nx00, nx10, w.y), mix(nx01, nx11, w.y), w.z);
}

// Worley F1: distance to the nearest of one hashed feature point per cell, in cell units.
inline float worley(float3 q, int3 off) {
    float3 fl = floor(q);
    int3 c = int3(fl) + off;
    float3 f = q - fl;
    float best = 1e9f;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                uint h0 = lattice_hash(c + int3(dx, dy, dz));
                uint h1 = pcg_hash(h0);
                uint h2 = pcg_hash(h1);
                float3 fp = float3(float(h0 >> 8u), float(h1 >> 8u), float(h2 >> 8u)) * (1.0f / 16777216.0f);
                float3 diff = float3(float(dx), float(dy), float(dz)) + fp - f;
                best = min(best, dot(diff, diff));
            }
        }
    }
    return saturate(sqrt(best));
}

inline float fbm_perlin(float3 q, uint seed) {
    float sum = 0.0f, amp = 1.0f, norm = 0.0f;
    for (uint o = 0u; o < kPerlinOctaves; ++o) {
        sum += amp * perlin(q, seed_offset(seed, o));
        norm += amp;
        amp *= 0.5f;
        q *= 2.0f;
    }
    return saturate(0.5f + 0.5f * sum / norm);
}

inline float fbm_worley(float3 q, uint seed) {
    float sum = 0.0f, amp = 1.0f, norm = 0.0f;
    for (uint o = 0u; o < kWorleyOctaves; ++o) {
        sum += amp * worley(q, seed_offset(seed, kWorleySaltBase + o));
        norm += amp;
        amp *= 0.5f;
        q *= 2.0f;
    }
    return sum / norm;
}

inline float remap(float v, float lo, float hi) {
    return (v - lo) / max(hi - lo, 1e-6f);
}

inline float height_profile(float yn) {
    return smoothstep(0.0f, 0.1f, yn) * (1.0f - smoothstep(0.6f, 1.0f, yn));
}

inline float edge_profile(float u) {
    return smoothstep(0.0f, 0.15f, u) * (1.0f - smoothstep(0.85f, 1.0f, u));
}

kernel void bake_density(constant DensityBakeParams& p [[buffer(0)]],
                         texture3d<half, access::write> out [[texture(0)]],
                         uint3 gid [[thread_position_in_grid]]) {
    uint3 grid = uint3(p.grid_x, p.grid_y, p.grid_z);
    if (any(gid >= grid)) {
        return;
    }
    float d;
    if (p.density_type == CNR_DENSITY_CONSTANT) {
        d = saturate(p.constant_density);
    } else {
        float3 ext = p.bounds_max - p.bounds_min;
        float3 x = p.bounds_min + (float3(gid) + 0.5f) / float3(grid) * ext;
        float3 qn = (x - p.bounds_min) / max(ext.x, max(ext.y, ext.z));
        float3 q = kPerlinBaseFrequency * qn;
        float3 un = (x - p.bounds_min) / ext;
        float shape = height_profile(un.y) * edge_profile(un.x) * edge_profile(un.z);
        float base = remap(fbm_perlin(q, p.density_seed), 1.0f - p.coverage, 1.0f) * shape;
        d = saturate(base - p.erosion * fbm_worley(kWorleyFrequencyScale * q, p.density_seed));
    }
    out.write(half4(half(d)), gid);
}
