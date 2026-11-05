// The article's neural shader: descriptor and MLP inference. Concatenated after src/shared_types.h,
// src/shaders/common.metal and src/neural_types.h. The host injects NN_IN, NN_HIDDEN,
// NN_HIDDEN_LAYERS, NN_OUT (from the model header) and NN_SHADER_ML (0 = hand-written scalar MSL,
// 1 = Metal 4 Shader ML matmul2d) as #defines.
//
// Function approximated by the MLP (the one line of the march the network replaces):
//   Ls(x, d) = f_sun E_sun + f_sky L_sky + f_gnd L_gnd
// the FULL in-scattered radiance at medium point x toward -d (d = view/tracing direction),
// pre-albedo, including the single-scattered sun. Grey medium, so f = (f_sun, f_sky, f_gnd) are
// scalars: the radiance for unit sources. The march applies albedo and extinction.

#if !defined(NN_IN) || !defined(NN_HIDDEN) || !defined(NN_HIDDEN_LAYERS) || !defined(NN_OUT) || !defined(NN_SHADER_ML)
#error "NN_* dimensions and NN_SHADER_ML must be injected by the host"
#endif
#if NN_IN != CNR_NN_IN || NN_HIDDEN != CNR_NN_HIDDEN || NN_HIDDEN_LAYERS != CNR_NN_HIDDEN_LAYERS || NN_OUT != CNR_NN_OUT
#error "unsupported MLP shape (expected 13 -> 16 -> 16 -> 3)"
#endif

constant float kNNRadii[CNR_NN_RADII] = {6.0f, 12.0f, 24.0f, 48.0f, 96.0f, 192.0f};

// Descriptor (13 floats). For each radius r (m): lod = log2(r / voxel_min_extent) clamped to the
// mip chain; [0:6) local sigma_t = sigma_t_max * density_at(x, lod), [6:12) toward the sun
// sigma_t_max * density_at(x + r * sun_direction, lod), [12] = dot(d, sun_direction).
// Shared by inference and training-data generation.
inline void build_descriptor(texture3d<float> tex, constant SceneGPU& s, float3 x, float3 d, thread float* out) {
    float max_lod = float(max(s.mip_levels, 1u) - 1u);
    for (uint i = 0u; i < uint(CNR_NN_RADII); ++i) {
        float r = kNNRadii[i];
        float lod = clamp(log2(r / s.voxel_min_extent), 0.0f, max_lod);
        out[i] = s.sigma_t_max * density_at(tex, s, x, lod);
        out[CNR_NN_RADII + i] = s.sigma_t_max * density_at(tex, s, x + r * s.sun_direction, lod);
    }
    out[2 * CNR_NN_RADII] = dot(d, s.sun_direction);
}

// Squared ReLU (the article's hidden activation).
inline float nn_act(float a) {
    float r = max(a, 0.0f);
    return r * r;
}

// Squared output, clamped to the model's per-output maximum.
inline float3 nn_output(float y0, float y1, float y2, constant NeuralStandardization& n) {
    return float3(min(y0 * y0, n.clamp[0]), min(y1 * y1, n.clamp[1]), min(y2 * y2, n.clamp[2]));
}

// Dense layer y = b + W x, W[NO][NI] row-major (out, in), fp32.
template <int NI, int NO>
inline void nn_dense(device const float* w, device const float* b, thread const float* x, thread float* y) {
#pragma unroll
    for (int o = 0; o < NO; ++o) {
        float acc = b[o];
#pragma unroll
        for (int i = 0; i < NI; ++i) {
            acc = fma(w[o * NI + i], x[i], acc);
        }
        y[o] = acc;
    }
}

// Hand-written MSL inference (the article's baseline): fp32 weights in .cnrm order, fp32 math.
inline float3 mlp_eval_scalar(device const float* w, constant NeuralStandardization& n, thread const float* desc) {
    float x[CNR_NN_IN];
#pragma unroll
    for (int i = 0; i < CNR_NN_IN; ++i) {
        x[i] = (desc[i] - n.mean[i]) * n.scale[i];
    }
    float a[CNR_NN_HIDDEN];
    float b[CNR_NN_HIDDEN];
    nn_dense<CNR_NN_IN, CNR_NN_HIDDEN>(w + CNR_NN_W0, w + CNR_NN_B0, x, a);
#pragma unroll
    for (int i = 0; i < CNR_NN_HIDDEN; ++i) {
        a[i] = nn_act(a[i]);
    }
    nn_dense<CNR_NN_HIDDEN, CNR_NN_HIDDEN>(w + CNR_NN_W1, w + CNR_NN_B1, a, b);
#pragma unroll
    for (int i = 0; i < CNR_NN_HIDDEN; ++i) {
        b[i] = nn_act(b[i]);
    }
    float y[CNR_NN_OUT];
    nn_dense<CNR_NN_HIDDEN, CNR_NN_OUT>(w + CNR_NN_W2, w + CNR_NN_B2, b, y);
    return nn_output(y[0], y[1], y[2], n);
}

#if NN_SHADER_ML
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// y[N] = x[K] W, W stored [K][N] row-major (N contiguous) as device half; one thread runs the whole
// product (execution_thread). float activations x half weights -> float (a supported matmul2d
// combination); fp32 activations keep the input/hidden values unrounded, so only the weights are fp16.
// The destination accumulates, so it is zeroed.
template <int K, int N>
inline void nn_matmul_ml(device half* w, thread float* x, thread float* y) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        y[i] = 0.0f;
    }
    auto tx = tensor<thread float, extents<int32_t, K, 1>, tensor_inline>(x, extents<int32_t, K, 1>());
    auto tw = tensor<device half, extents<int32_t, N, K>, tensor_inline>(w, extents<int32_t, N, K>());
    auto ty = tensor<thread float, extents<int32_t, N, 1>, tensor_inline>(y, extents<int32_t, N, 1>());
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(1, N, K, false, false, false);
    mpp::tensor_ops::matmul2d<desc, execution_thread> op;
    op.run(tx, tw, ty);
}

// Metal 4 Shader ML inference: matmul2d with fp32 activations and fp16 weights ([K][N] padded layout
// at CNR_NN_ML_*), fp32 accumulation; biases (fp32, from the scalar buffer) and activations are applied
// in registers.
inline float3 mlp_eval_shader_ml(device half* w16, device const float* w, constant NeuralStandardization& n,
                                 thread const float* desc) {
    float x[CNR_NN_IN_PAD];
#pragma unroll
    for (int i = 0; i < CNR_NN_IN; ++i) {
        x[i] = (desc[i] - n.mean[i]) * n.scale[i];
    }
#pragma unroll
    for (int i = CNR_NN_IN; i < CNR_NN_IN_PAD; ++i) {
        x[i] = 0.0f;
    }
    float acc[CNR_NN_HIDDEN];
    float h[CNR_NN_HIDDEN];
    nn_matmul_ml<CNR_NN_IN_PAD, CNR_NN_HIDDEN>(w16 + CNR_NN_ML_W0, x, acc);
#pragma unroll
    for (int i = 0; i < CNR_NN_HIDDEN; ++i) {
        h[i] = nn_act(acc[i] + w[CNR_NN_B0 + i]);
    }
    nn_matmul_ml<CNR_NN_HIDDEN, CNR_NN_HIDDEN>(w16 + CNR_NN_ML_W1, h, acc);
#pragma unroll
    for (int i = 0; i < CNR_NN_HIDDEN; ++i) {
        h[i] = nn_act(acc[i] + w[CNR_NN_B1 + i]);
    }
    float y[CNR_NN_OUT_PAD];
    nn_matmul_ml<CNR_NN_HIDDEN, CNR_NN_OUT_PAD>(w16 + CNR_NN_ML_W2, h, y);
    return nn_output(y[0] + w[CNR_NN_B2 + 0], y[1] + w[CNR_NN_B2 + 1], y[2] + w[CNR_NN_B2 + 2], n);
}
#endif

// The variant selected at compile time. w16 is unused (and may be null) by the scalar variant.
inline float3 mlp_eval(device const float* w, device half* w16, constant NeuralStandardization& n,
                       thread const float* desc) {
#if NN_SHADER_ML
    return mlp_eval_shader_ml(w16, w, n, desc);
#else
    return mlp_eval_scalar(w, n, desc);
#endif
}
