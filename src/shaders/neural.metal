// Neural renderer (docs/architecture.md §8): the analytic integrator march_view_ray with
// L_rest from the MLP. Concatenated after neural_common.metal.

struct NeuralLRest {
    texture3d<float> tex;
    constant SceneGPU* s;
    device const half* weights;

    float3 operator()(float3 x, float3 d, float tau_sun) {
        float in[NN_IN];
        build_descriptor(tex, *s, x, d, tau_sun, in);
        float3 f = mlp_eval(weights, in);
        LightCoeffs c;
        c.sun = f.x;
        c.sky = f.y;
        c.ground = f.z;
        return combine_lights(*s, c);
    }
};

kernel void neural_render(constant SceneGPU& s [[buffer(0)]],
                          constant NeuralParams& p [[buffer(1)]],
                          device const half* weights [[buffer(2)]],
                          texture3d<float> density [[texture(0)]],
                          texture2d<float, access::write> out [[texture(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= s.width || gid.y >= s.height || gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    Ray r = camera_ray(s, float2(gid) + 0.5f);
    NeuralLRest lrest{density, &s, weights};
    float3 L = march_view_ray(density, s, r, p.primary_steps, p.shadow_steps, lrest);
    out.write(float4(L, 1.0f), gid);
}

// Batch inference for GPU/PyTorch parity checks: inputs[count][NN_IN] -> outputs[count][3] = f.
kernel void neural_eval_batch(constant NeuralEvalParams& p [[buffer(0)]],
                              device const half* weights [[buffer(1)]],
                              device const float* inputs [[buffer(2)]],
                              device float* outputs [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= p.count) {
        return;
    }
    float in[NN_IN];
    for (uint k = 0u; k < uint(NN_IN); ++k) {
        in[k] = inputs[i * uint(NN_IN) + k];
    }
    float3 f = mlp_eval(weights, in);
    outputs[i * 3u + 0u] = f.x;
    outputs[i * 3u + 1u] = f.y;
    outputs[i * 3u + 2u] = f.z;
}
