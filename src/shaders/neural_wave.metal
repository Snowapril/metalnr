// Neural renderer, ml_encoder inference: the shared march split into chunks (march_advance) so the MLP runs
// as one MTL4MachineLearningCommandEncoder dispatch per chunk. Concatenated after neural_common.metal
// (build_descriptor). Per frame: wave_init; C x {wave_gather -> ML dispatch -> wave_accumulate}; wave_finish,
// with C = ceil(max_steps / CNR_WAVE_SAMPLES) so no march is truncated. The result equals the single-pass
// march with the same network (same samples, weights and accumulation order).
// Bindings: buffer 0 SceneGPU, 1 MarchParams, 2 MarchState[pixels], 3 ML input rows, 4 march weights,
// 5 ML output rows; texture 0 density, texture 1 rgba32Float output.

static_assert(sizeof(MarchState) == CNR_WAVE_STATE_BYTES, "MarchState layout (host allocates CNR_WAVE_STATE_BYTES)");

// march_advance sink: stores the raw descriptor and the march weight of shaded sample j and contributes
// nothing yet (wave_accumulate adds w_j * Ls(f_j) once the network ran).
struct WaveGatherSink {
    texture3d<float> tex;
    constant SceneGPU* s;
    device float* rows;     // this pixel's CNR_WAVE_SAMPLES rows of CNR_WAVE_ROW floats
    device float* weights;  // this pixel's CNR_WAVE_SAMPLES weights
    uint j;

    float3 operator()(float3 x, float3 d, float w) {
        float desc[CNR_NN_IN];
        build_descriptor(tex, *s, x, d, desc);
        device float* row = rows + j * CNR_WAVE_ROW;
        for (uint k = 0u; k < uint(CNR_NN_IN); ++k) {
            row[k] = desc[k];
        }
        weights[j] = w;
        ++j;
        return float3(0.0f);
    }
};

inline bool wave_pixel(constant SceneGPU& s, uint2 gid, thread uint& pixel) {
    if (gid.x >= s.width || gid.y >= s.height) {
        return false;
    }
    pixel = gid.y * s.width + gid.x;
    return true;
}

kernel void wave_init(constant SceneGPU& s [[buffer(0)]],
                      device MarchState* state [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint pixel;
    if (!wave_pixel(s, gid, pixel)) {
        return;
    }
    state[pixel] = march_state_init();
}

// Advances each pixel's march by up to CNR_WAVE_SAMPLES shaded samples. Slots past the last shaded sample
// (finished pixels: all of them) get weight 0 and an all-zero input row.
kernel void wave_gather(constant SceneGPU& s [[buffer(0)]],
                        constant MarchParams& p [[buffer(1)]],
                        device MarchState* state [[buffer(2)]],
                        device float* rows [[buffer(3)]],
                        device float* weights [[buffer(4)]],
                        texture3d<float> density [[texture(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint pixel;
    if (!wave_pixel(s, gid, pixel)) {
        return;
    }
    WaveGatherSink sink;
    sink.tex = density;
    sink.s = &s;
    sink.rows = rows + pixel * CNR_WAVE_SAMPLES * CNR_WAVE_ROW;
    sink.weights = weights + pixel * CNR_WAVE_SAMPLES;
    sink.j = 0u;
    MarchState st = state[pixel];
    if (st.done == 0u) {
        Ray r = camera_ray(s, float2(gid) + 0.5f);
        MarchSettings m = make_march_settings(s, p, gid);
        march_advance(density, s, r, m, sink, st, CNR_WAVE_SAMPLES);
        state[pixel] = st;
    }
    for (uint j = sink.j; j < CNR_WAVE_SAMPLES; ++j) {
        device float* row = sink.rows + j * CNR_WAVE_ROW;
        for (uint k = 0u; k < uint(CNR_NN_IN); ++k) {
            row[k] = 0.0f;
        }
        sink.weights[j] = 0.0f;
    }
}

// L += w_j * Ls(f_j) in sample order (the order of the single-pass march). f rows: (f_sun, f_sky, f_gnd, pad).
kernel void wave_accumulate(constant SceneGPU& s [[buffer(0)]],
                            device MarchState* state [[buffer(2)]],
                            device const float* weights [[buffer(4)]],
                            device const float* f [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint pixel;
    if (!wave_pixel(s, gid, pixel)) {
        return;
    }
    device const float* w = weights + pixel * CNR_WAVE_SAMPLES;
    device const float* rows = f + pixel * CNR_WAVE_SAMPLES * CNR_WAVE_ROW;
    float3 L = state[pixel].L;
    for (uint j = 0u; j < CNR_WAVE_SAMPLES; ++j) {
        // w = 0 marks unused slots; a shaded sample with w = 0 contributes 0 as well.
        if (w[j] > 0.0f) {
            device const float* row = rows + j * CNR_WAVE_ROW;
            LightCoeffs c;
            c.sun = row[0];
            c.sky = row[1];
            c.ground = row[2];
            L += w[j] * combine_lights(s, c);
        }
    }
    state[pixel].L = L;
}

kernel void wave_finish(constant SceneGPU& s [[buffer(0)]],
                        device const MarchState* state [[buffer(2)]],
                        texture2d<float, access::write> out [[texture(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint pixel;
    if (!wave_pixel(s, gid, pixel) || gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    Ray r = camera_ray(s, float2(gid) + 0.5f);
    out.write(march_finish(s, r, state[pixel]), gid);
}
