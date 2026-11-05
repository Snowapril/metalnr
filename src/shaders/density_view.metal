// Density visualization: optical depth along one pixel-center ray, shown as white
// over the environment: out = mix(L_env, 1, 1 - T), T = exp(-tau).

kernel void density_view(constant SceneGPU& s [[buffer(0)]],
                         constant DensityViewParams& v [[buffer(1)]],
                         texture3d<float> density [[texture(0)]],
                         texture2d<float, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= s.width || gid.y >= s.height || gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    Ray r = camera_ray(s, float2(gid) + 0.5f);
    float tau = march_optical_depth(density, s, r.o, r.d, v.steps);
    float T = exp(-tau);
    out.write(float4(mix(env_radiance(s, r.d), float3(1.0f), 1.0f - T), 1.0f), gid);
}
