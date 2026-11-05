// Neural renderer: the shared march (march_view_ray) with the in-scattered radiance from the MLP.
// Concatenated after neural_common.metal; compiled once per inference variant (NN_SHADER_ML 0/1).
// Bindings: buffer 2 = fp32 weights (.cnrm order, biases for both variants); with NN_SHADER_ML,
// buffer 3 = fp16 [K][N] matrices for matmul2d.

struct NeuralShade {
    texture3d<float> tex;
    constant SceneGPU* s;
    constant NeuralStandardization* norm;
    device const float* w;
    device half* w16;

    float3 operator()(float3 x, float3 d) {
        float desc[CNR_NN_IN];
        build_descriptor(tex, *s, x, d, desc);
        float3 f = mlp_eval(w, w16, *norm, desc);
        LightCoeffs c;
        c.sun = f.x;
        c.sky = f.y;
        c.ground = f.z;
        return combine_lights(*s, c);
    }
};

kernel void neural_render(constant SceneGPU& s [[buffer(0)]],
                          constant NeuralParams& p [[buffer(1)]],
                          device const float* weights [[buffer(2)]],
#if NN_SHADER_ML
                          device half* weights16 [[buffer(3)]],
#endif
                          texture3d<float> density [[texture(0)]],
                          texture2d<float, access::write> out [[texture(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= s.width || gid.y >= s.height || gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    Ray r = camera_ray(s, float2(gid) + 0.5f);
    MarchSettings m = make_march_settings(s, p.march, gid);
    NeuralShade shade;
    shade.tex = density;
    shade.s = &s;
    shade.norm = &p.norm;
    shade.w = weights;
#if NN_SHADER_ML
    shade.w16 = weights16;
#else
    shade.w16 = nullptr;
#endif
    out.write(march_view_ray(density, s, r, m, shade), gid);
}

// Batch inference for parity checks: inputs[count][13] (raw descriptors) -> outputs[count][3] = f.
kernel void neural_eval_batch(constant NeuralEvalParams& p [[buffer(0)]],
                              device const float* inputs [[buffer(1)]],
                              device float* outputs [[buffer(2)]],
                              device const float* weights [[buffer(3)]],
#if NN_SHADER_ML
                              device half* weights16 [[buffer(4)]],
#endif
                              uint i [[thread_position_in_grid]]) {
    if (i >= p.count) {
        return;
    }
    float desc[CNR_NN_IN];
    for (uint k = 0u; k < uint(CNR_NN_IN); ++k) {
        desc[k] = inputs[i * uint(CNR_NN_IN) + k];
    }
#if NN_SHADER_ML
    float3 f = mlp_eval(weights, weights16, p.norm, desc);
#else
    float3 f = mlp_eval(weights, nullptr, p.norm, desc);
#endif
    outputs[i * 3u + 0u] = f.x;
    outputs[i * 3u + 1u] = f.y;
    outputs[i * 3u + 2u] = f.z;
}
