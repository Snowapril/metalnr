// Neural L_rest: descriptor and MLP inference (docs/architecture.md §8). Concatenated after
// src/shared_types.h, src/shaders/common.metal and src/neural_types.h; the host injects
// NN_IN, NN_HIDDEN, NN_HIDDEN_LAYERS, NN_OUT, NN_LEVELS as #defines from the model header.
//
// Function approximated by the MLP:
//   L_rest(x, d) = f_sun E_sun + f_sky L_sky + f_gnd L_gnd
// the in-scattered radiance at medium point x toward -d (d = tracing/view direction),
// EXCLUDING the direct single-scattered sun term p(sun -> d) T_sun E_sun, which
// march_view_ray adds itself. f = (f_sun, f_sky, f_gnd) are scalars (grey medium), the
// radiance for unit sources. Albedo at x is applied by the integrator, not by f.

#if !defined(NN_IN) || !defined(NN_HIDDEN) || !defined(NN_HIDDEN_LAYERS) || !defined(NN_OUT) || !defined(NN_LEVELS)
#error "NN_* dimensions must be injected by the host"
#endif
#if NN_IN != 7 * NN_LEVELS + 5
#error "NN_IN must equal 7 * NN_LEVELS + 5 (descriptor layout)"
#endif
#if NN_OUT != 3 || NN_HIDDEN_LAYERS < 1
#error "unsupported MLP shape"
#endif

// Descriptor (NN_IN floats):
//   [0, 7*NN_LEVELS): for level l, h_l = voxel_min_extent * 2^l, mip lod l, points
//     x + h_l * {0, +X, -X, +Y, -Y, +Z, -Z} in the local frame: log(1 + sigma_t_max * d * h_l)
//   then cos(view, sun), world +Y in the local frame (3), T_sun = exp(-tau_sun).
// Local frame: Z = sun_direction, X = d projected perpendicular to Z (world X projected if
// d is parallel to the sun), Y = cross(Z, X).
inline void build_descriptor(texture3d<float> tex, constant SceneGPU& s, float3 x, float3 d, float tau_sun,
                             thread float* out) {
    float3 az = s.sun_direction;
    float3 ax = d - dot(d, az) * az;
    if (length_squared(ax) < 1e-8f) {
        ax = float3(1.0f, 0.0f, 0.0f) - az.x * az;
        if (length_squared(ax) < 1e-8f) {
            ax = float3(0.0f, 0.0f, 1.0f) - az.z * az;  // sun along world X
        }
    }
    ax = normalize(ax);
    float3 ay = cross(az, ax);

    uint k = 0u;
    for (uint l = 0u; l < uint(NN_LEVELS); ++l) {
        float h = s.voxel_min_extent * float(1u << l);
        float lod = float(l);
        float scale = s.sigma_t_max * h;
        float3 offsets[7] = {float3(0.0f), ax, -ax, ay, -ay, az, -az};
        for (uint j = 0u; j < 7u; ++j) {
            out[k++] = log(1.0f + scale * density_at(tex, s, x + h * offsets[j], lod));
        }
    }
    out[k++] = dot(d, az);
    out[k++] = ax.y;  // world +Y = (0,1,0) expressed in (ax, ay, az)
    out[k++] = ay.y;
    out[k++] = az.y;
    out[k] = precise::exp(-tau_sun);
}

// expm1 for y >= 0 without cancellation near 0 (MSL has no expm1).
inline float nn_expm1(float y) {
    if (y < 0.1f) {
        return y * (1.0f + y * (0.5f + y * (1.0f / 6.0f + y * (1.0f / 24.0f + y * (1.0f / 120.0f)))));
    }
    return precise::exp(y) - 1.0f;
}

// One dense layer: W[NO][NI] row-major, then bias[NO]; half weights, float accumulation.
template <uint NI, uint NO, bool RELU>
inline void nn_layer(device const half* w, thread const float* in, thread float* out) {
    device const half* bias = w + NO * NI;
    for (uint o = 0u; o < NO; ++o) {
        device const half* row = w + o * NI;
        float acc = float(bias[o]);
        for (uint i = 0u; i < NI; ++i) {
            acc = fma(float(row[i]), in[i], acc);
        }
        out[o] = RELU ? max(acc, 0.0f) : acc;
    }
}

// MLP NN_IN -> NN_HIDDEN (x NN_HIDDEN_LAYERS, ReLU) -> NN_OUT; returns f = expm1(max(y, 0)).
inline float3 mlp_eval(device const half* weights, thread const float* in) {
    float a[NN_HIDDEN];
    float b[NN_HIDDEN];
    device const half* w = weights;
    nn_layer<NN_IN, NN_HIDDEN, true>(w, in, a);
    w += NN_HIDDEN * NN_IN + NN_HIDDEN;
    for (uint layer = 1u; layer < uint(NN_HIDDEN_LAYERS); ++layer) {
        nn_layer<NN_HIDDEN, NN_HIDDEN, true>(w, a, b);
        for (uint i = 0u; i < uint(NN_HIDDEN); ++i) {
            a[i] = b[i];
        }
        w += NN_HIDDEN * NN_HIDDEN + NN_HIDDEN;
    }
    float y[NN_OUT];
    nn_layer<NN_HIDDEN, NN_OUT, false>(w, a, y);
    return float3(nn_expm1(max(y[0], 0.0f)), nn_expm1(max(y[1], 0.0f)), nn_expm1(max(y[2], 0.0f)));
}
