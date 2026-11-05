// Cumulus / constant density bake into mip 0 of the r16Float 3D texture (scene version 3).
//
// Cumulus field ("Neural Shaders, Explained With Clouds", Sample()), evaluated at each voxel centre in
// cloud-local metres p = x - base_center (+Y up, origin at the centre of the flat base):
//   p.xz -= shear * p.y                                            wind shear
//   t = smoothstep(anvil_height, cloud_height, p.y)
//   p.xz /= 1 + anvil_spread / cloud_radius * t                    anvil warp (widens the top)
//   d = lobes[0].w - |p - lobes[0].xyz|
//   d = smooth_max(d, lobes[i].w - |p - lobes[i].xyz|, lobe_blend) for i = 1 .. lobe_count-1
//   d = smooth_min(d, p.y, base_softness)                          flat base
//   d += fbm(p / erosion_feature_size) * erosion_amplitude * smoothstep(0, base_fade_height, p.y)
//   density = saturate(d / edge)
// smooth_min / smooth_max: polynomial (Quilez), h = max(k - |a - b|, 0) / k, min(a,b) - h^2 k / 4 and
//   max(a,b) + h^2 k / 4. Hence smooth_max <= max + k/4 and smooth_min <= min, so the density is exactly
//   0 for p.y <= 0 (the erosion fade is 0 there) and outside every lobe inflated by
//   (lobe_count - 1) * lobe_blend / 4 + erosion_amplitude (the bound used by the tests and makeRandomCumulus).
// fbm: signed fractal sum of 3D gradient (improved Perlin) noise, kErosionLacunarity = 2, kErosionGain = 0.5,
//   erosion_octaves octaves, divided by the sum of amplitudes and clamped to [-1, 1].
// Lattice gradients come from integer hashing (pcg_hash) of the cell coordinate plus a per-octave lattice
// offset hashed from (seed, octave), so the bake is deterministic and independent of thread order.

constant float kErosionLacunarity = 2.0f;
constant float kErosionGain = 0.5f;

inline uint lattice_hash(int3 c) {
    uint3 u = uint3(c);
    return pcg_hash(u.x + pcg_hash(u.y + pcg_hash(u.z)));
}

inline int3 seed_offset(uint seed, uint octave) {
    uint h = pcg_hash(seed ^ pcg_hash(octave + 0x51ED27u));
    return int3(int(h & 1023u), int((h >> 10u) & 1023u), int((h >> 20u) & 1023u));
}

// Improved Perlin gradient selection (12 cube-edge gradients, 16 entries).
inline float grad_dot(uint h, float3 f) {
    uint k = h & 15u;
    float u = k < 8u ? f.x : f.y;
    float v = k < 4u ? f.y : ((k == 12u || k == 14u) ? f.x : f.z);
    return ((k & 1u) ? -u : u) + ((k & 2u) ? -v : v);
}

inline float gradient_noise(float3 q, int3 off) {
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

// Signed fBm in [-1, 1].
inline float fbm_noise(float3 q, uint octaves, uint seed) {
    float sum = 0.0f, amp = 1.0f, norm = 0.0f;
    for (uint o = 0u; o < octaves; ++o) {
        sum += amp * gradient_noise(q, seed_offset(seed, o));
        norm += amp;
        amp *= kErosionGain;
        q *= kErosionLacunarity;
    }
    return norm > 0.0f ? clamp(sum / norm, -1.0f, 1.0f) : 0.0f;
}

inline float smooth_min(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0f) / k;
    return min(a, b) - h * h * k * 0.25f;
}

inline float smooth_max(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0f) / k;
    return max(a, b) + h * h * k * 0.25f;
}

inline float cumulus_density(constant DensityBakeParams& c, float3 x) {
    float3 p = x - c.base_center;
    p.xz -= float2(c.shear_x, c.shear_z) * p.y;
    float t = smoothstep(c.anvil_height, c.cloud_height, p.y);
    p.xz /= 1.0f + c.anvil_spread / c.cloud_radius * t;
    float d = c.lobes[0].w - length(p - c.lobes[0].xyz);
    uint n = min(c.lobe_count, uint(CNR_MAX_LOBES));
    for (uint i = 1u; i < n; ++i) {
        d = smooth_max(d, c.lobes[i].w - length(p - c.lobes[i].xyz), c.lobe_blend);
    }
    d = smooth_min(d, p.y, c.base_softness);
    float fade = smoothstep(0.0f, c.base_fade_height, p.y);
    d += fbm_noise(p / c.erosion_feature_size, c.erosion_octaves, c.seed) * c.erosion_amplitude * fade;
    return saturate(d / c.edge);
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
        d = p.lobe_count > 0u ? cumulus_density(p, x) : 0.0f;
    }
    out.write(half4(half(d)), gid);
}
