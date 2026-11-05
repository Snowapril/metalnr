// Test-only kernels. Concatenated after src/shared_types.h and src/shaders/common.metal.

// out[0] = ray direction, out[1] = ray origin of the ray through pixel (width/2, height/2),
// which is the image center for even width and height.
kernel void test_camera_ray(constant SceneGPU& s [[buffer(0)]],
                            device float4* out [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    Ray r = camera_ray(s, float2(float(s.width) * 0.5f, float(s.height) * 0.5f));
    out[0] = float4(r.d, 0.0f);
    out[1] = float4(r.o, 0.0f);
}

// out[i] = density_at(points[i]); one thread per point.
kernel void test_density_at(constant SceneGPU& s [[buffer(0)]],
                            device const float4* points [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            texture3d<float> density [[texture(0)]],
                            uint gid [[thread_position_in_grid]]) {
    out[gid] = density_at(density, s, points[gid].xyz);
}
