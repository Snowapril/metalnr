// Test kernels for tests/test_neural.mm. Concatenated after neural_common.metal.

// queries[i].xyz = x, queries[i].w = tau_sun; dirs[i].xyz = d (unit). out: NN_IN floats per query.
kernel void test_build_descriptor(constant SceneGPU& s [[buffer(0)]],
                                  device const float4* queries [[buffer(1)]],
                                  device const float4* dirs [[buffer(2)]],
                                  device float* out [[buffer(3)]],
                                  texture3d<float> density [[texture(0)]],
                                  uint i [[thread_position_in_grid]]) {
    float in[NN_IN];
    build_descriptor(density, s, queries[i].xyz, dirs[i].xyz, queries[i].w, in);
    for (uint k = 0u; k < uint(NN_IN); ++k) {
        out[i * uint(NN_IN) + k] = in[k];
    }
}
