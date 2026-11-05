// Analytic renderer ("the usual way"): the shared march_view_ray with an in-scattering estimate made of the
// Wrenninge 2013 multiple-scattering octaves (i = 0 is the exact single-scattered sun) plus a two-direction
// ambient term.

// Optical depth from p along unit d over min(max_distance, AABB exit) (max_distance 0 = AABB exit), with
// `steps` uniform midpoint samples.
inline float march_optical_depth_bounded(texture3d<float> tex, constant SceneGPU& s, float3 p, float3 d, uint steps,
                                         float max_distance) {
    Ray r;
    r.o = p;
    r.d = d;
    float t0, t1;
    if (steps == 0u || !intersect_aabb(r, s.bounds_min, s.bounds_max, t0, t1)) {
        return 0.0f;
    }
    if (max_distance > 0.0f) {
        t1 = min(t1, max_distance);
    }
    if (t1 <= t0) {
        return 0.0f;
    }
    float dt = (t1 - t0) / float(steps);
    float sum = 0.0f;
    for (uint i = 0u; i < steps; ++i) {
        sum += sigma_t_at(tex, s, p + (t0 + (float(i) + 0.5f) * dt) * d);
    }
    return sum * dt;
}

struct AnalyticShade {
    texture3d<float> tex;
    constant SceneGPU* s;
    constant AnalyticParams* p;

    // Ls = sum_{i=0}^{octaves-1} b^i E_sun p_{g c^i}(cos) exp(-a^i tau_sun)
    //    + [ambient] 0.5 (L_sky exp(-tau_up) + L_gnd exp(-tau_down)).
    float3 operator()(float3 x, float3 d) {
        float cos_theta = dot(s->sun_direction, d);
        float tau_sun = march_optical_depth_bounded(tex, *s, x, s->sun_direction, p->shadow_steps,
                                                    p->shadow_distance);
        float f_sun = 0.0f;
        float a = 1.0f;
        float b = 1.0f;
        float c = 1.0f;
        for (uint i = 0u; i < p->octaves; ++i) {
            f_sun += b * hg_eval(cos_theta, s->hg_g * c) * precise::exp(-a * tau_sun);
            a *= p->ms_a;
            b *= p->ms_b;
            c *= p->ms_c;
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
    MarchSettings m = make_march_settings(s, p.march, gid);
    AnalyticShade shade;
    shade.tex = density;
    shade.s = &s;
    shade.p = &p;
    out.write(march_view_ray(density, s, r, m, shade), gid);
}
