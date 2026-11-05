// Analytic real-time baseline (docs/architecture.md §7): deterministic march (march_view_ray) with
// L_rest from Wrenninge 2013 multiple-scattering octaves plus a two-direction ambient estimate.

struct AnalyticLRest {
    texture3d<float> tex;
    constant SceneGPU* s;
    constant AnalyticParams* p;

    // L_rest = sum_{i=1}^{octaves-1} b^i E_sun p_{g c^i}(cos) exp(-a^i tau_sun)
    //        + [ambient] 0.5 (L_sky exp(-tau_up) + L_gnd exp(-tau_down)).
    float3 operator()(float3 x, float3 d, float tau_sun) {
        float cos_theta = dot(s->sun_direction, d);
        float f_sun = 0.0f;
        float a = 1.0f;
        float b = 1.0f;
        float c = 1.0f;
        for (uint i = 1u; i < p->octaves; ++i) {
            a *= p->ms_a;
            b *= p->ms_b;
            c *= p->ms_c;
            f_sun += b * hg_eval(cos_theta, s->hg_g * c) * precise::exp(-a * tau_sun);
        }
        float3 L = f_sun * s->sun_irradiance;
        if ((p->flags & CNR_ANALYTIC_AMBIENT) != 0u) {
            float tau_up = march_optical_depth(tex, *s, x, float3(0.0f, 1.0f, 0.0f), p->ambient_steps);
            float tau_down = march_optical_depth(tex, *s, x, float3(0.0f, -1.0f, 0.0f), p->ambient_steps);
            L += 0.5f * (s->sky_radiance * precise::exp(-tau_up) + s->ground_radiance * precise::exp(-tau_down));
        }
        return L;
    }
};

kernel void analytic_render(constant SceneGPU& s [[buffer(0)]],
                            constant AnalyticParams& p [[buffer(1)]],
                            texture3d<float> density [[texture(0)]],
                            texture2d<float, access::write> out [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= s.width || gid.y >= s.height || gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    Ray r = camera_ray(s, float2(gid) + 0.5f);
    AnalyticLRest lrest;
    lrest.tex = density;
    lrest.s = &s;
    lrest.p = &p;
    float3 L = march_view_ray(density, s, r, p.primary_steps, p.shadow_steps, lrest);
    out.write(float4(L, 1.0f), gid);
}
