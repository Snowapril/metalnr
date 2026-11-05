// Density visualization: the shared march (march_view_ray) with zero in-scattering, shown as white
// over the environment. The march returns (T * L_env, alpha = 1 - T); out = (mix(L_env, 1, alpha), alpha).

struct DensityShade {
    float3 operator()(float3, float3) { return float3(0.0f); }
};

kernel void density_view(constant SceneGPU& s [[buffer(0)]],
                         constant DensityViewParams& v [[buffer(1)]],
                         texture3d<float> density [[texture(0)]],
                         texture2d<float, access::write> out [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= s.width || gid.y >= s.height || gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    Ray r = camera_ray(s, float2(gid) + 0.5f);
    MarchSettings m = make_march_settings(s, v.march, gid);
    DensityShade shade;
    float4 res = march_view_ray(density, s, r, m, shade);
    float alpha = res.w;
    out.write(float4(mix(env_radiance(s, r.d), float3(1.0f), alpha), alpha), gid);
}
