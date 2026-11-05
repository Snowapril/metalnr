// Test kernels for tests/test_neural.mm. Concatenated after neural_common.metal.

// points[i].xyz = x, dirs[i].xyz = d (unit). out: CNR_NN_IN floats per query (build_descriptor).
kernel void test_build_descriptor(constant SceneGPU& s [[buffer(0)]],
                                  device const float4* points [[buffer(1)]],
                                  device const float4* dirs [[buffer(2)]],
                                  device float* out [[buffer(3)]],
                                  texture3d<float> density [[texture(0)]],
                                  uint i [[thread_position_in_grid]]) {
    float desc[CNR_NN_IN];
    build_descriptor(density, s, points[i].xyz, dirs[i].xyz, desc);
    for (uint k = 0u; k < uint(CNR_NN_IN); ++k) {
        out[i * uint(CNR_NN_IN) + k] = desc[k];
    }
}
