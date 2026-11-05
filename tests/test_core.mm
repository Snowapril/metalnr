// Host-only tests: scene/config JSON, camera basis, PFM, display mapping, metrics. No GPU.
#include "check.h"
#include "image_io.h"
#include "metrics.h"
#include "scene.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <unistd.h>

namespace {

std::string tempPath(const std::string& name) {
    return std::string(NSTemporaryDirectory().UTF8String) + "cloudnr_test_" + std::to_string(getpid()) + "_" + name;
}

std::string sourcePath(const std::string& rel) { return std::string(CLOUDNR_SOURCE_ROOT) + "/" + rel; }

bool eq3(simd_float3 a, simd_float3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
bool eq4(simd_float4 a, simd_float4 b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }

bool cloudEqual(const CloudDesc& a, const CloudDesc& b) {
    bool lobes = a.lobes.size() == b.lobes.size();
    for (size_t i = 0; lobes && i < a.lobes.size(); ++i) lobes = eq4(a.lobes[i], b.lobes[i]);
    return lobes && eq3(a.base_center, b.base_center) && a.lobe_blend == b.lobe_blend && a.shear.x == b.shear.x &&
           a.shear.y == b.shear.y && a.anvil_height == b.anvil_height && a.height == b.height &&
           a.anvil_spread == b.anvil_spread && a.radius == b.radius && a.base_softness == b.base_softness &&
           a.base_fade_height == b.base_fade_height && a.erosion_feature_size == b.erosion_feature_size &&
           a.erosion_octaves == b.erosion_octaves && a.erosion_amplitude == b.erosion_amplitude &&
           a.edge == b.edge && a.seed == b.seed;
}

void checkSceneEqual(const Scene& a, const Scene& b) {
    CHECK(a.version == b.version);
    CHECK(a.name == b.name);
    CHECK(a.width == b.width);
    CHECK(a.height == b.height);
    CHECK(a.cameras.size() == b.cameras.size());
    for (size_t i = 0; i < a.cameras.size() && i < b.cameras.size(); ++i) {
        const CameraDesc &ca = a.cameras[i], &cb = b.cameras[i];
        CHECK(ca.name == cb.name);
        CHECK(eq3(ca.position, cb.position));
        CHECK(eq3(ca.target, cb.target));
        CHECK(ca.vfov_deg == cb.vfov_deg);
        CHECK(ca.exposure_ev == cb.exposure_ev);
    }
    const VolumeDesc &va = a.volume, &vb = b.volume;
    CHECK(eq3(va.bounds_min, vb.bounds_min));
    CHECK(eq3(va.bounds_max, vb.bounds_max));
    CHECK(simd_all(va.grid == vb.grid));
    CHECK(va.density == vb.density);
    if (va.density == "cumulus")
        CHECK(cloudEqual(va.cloud, vb.cloud));
    else
        CHECK(va.constant_density == vb.constant_density);
    CHECK(a.medium.sigma_t_max == b.medium.sigma_t_max);
    CHECK(a.medium.albedo == b.medium.albedo);
    CHECK(a.medium.hg_g == b.medium.hg_g);
    // sun_direction is renormalized on every load; allow one-ulp-level drift.
    for (int i = 0; i < 3; ++i) CHECK_NEAR(a.light.sun_direction[i], b.light.sun_direction[i], 1e-6);
    CHECK(eq3(a.light.sun_irradiance, b.light.sun_irradiance));
    CHECK(eq3(a.light.sky_radiance, b.light.sky_radiance));
    CHECK(eq3(a.light.ground_radiance, b.light.ground_radiance));
}

float smoothstep01(float e0, float e1, float x) {
    float t = std::clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Conservative world AABB of the region where the baked cumulus density can be > 0: each lobe is
// inflated by (lobes - 1) * lobe_blend / 4 (polynomial smooth max) + erosion_amplitude, the base cuts
// everything at cloud-local y <= 0, and over each lobe's y range [y0, y1] the world offset is
// x = q * s(y) + shear * y with q in the inflated lobe, s(y) the monotone anvil scale.
void cloudWorldBounds(const CloudDesc& c, simd_float3* lo, simd_float3* hi) {
    const float inflate = float(c.lobes.size() - 1) * c.lobe_blend * 0.25f + c.erosion_amplitude;
    const float inf = std::numeric_limits<float>::infinity();
    *lo = simd_make_float3(inf, inf, inf);
    *hi = -*lo;
    for (simd_float4 l : c.lobes) {
        const float R = l.w + inflate;
        const float y0 = std::max(0.0f, l.y - R), y1 = l.y + R;
        if (y1 <= 0.0f) continue;
        const float k = c.anvil_spread / c.radius;
        const float s0 = 1.0f + k * smoothstep01(c.anvil_height, c.height, y0);
        const float s1 = 1.0f + k * smoothstep01(c.anvil_height, c.height, y1);
        const float centre[2] = {l.x, l.z};
        const float shear[2] = {c.shear.x, c.shear.y};
        float axisLo[2], axisHi[2];
        for (int a = 0; a < 2; ++a) {
            const float qlo = centre[a] - R, qhi = centre[a] + R;
            axisLo[a] = std::min(qlo * s0, qlo * s1) + std::min(shear[a] * y0, shear[a] * y1);
            axisHi[a] = std::max(qhi * s0, qhi * s1) + std::max(shear[a] * y0, shear[a] * y1);
        }
        lo->x = std::min(lo->x, axisLo[0]);
        hi->x = std::max(hi->x, axisHi[0]);
        lo->z = std::min(lo->z, axisLo[1]);
        hi->z = std::max(hi->z, axisHi[1]);
        lo->y = std::min(lo->y, y0);
        hi->y = std::max(hi->y, y1);
    }
    *lo += c.base_center;
    *hi += c.base_center;
}

void testSceneRoundTrip() {
    for (const char* file : {"scenes/cumulus.json", "scenes/cumulus_heldout.json", "scenes/homogeneous_test.json"}) {
        std::string err;
        auto s = loadScene(sourcePath(file), &err);
        CHECK(s.has_value());
        if (!s) {
            std::fprintf(stderr, "%s\n", err.c_str());
            continue;
        }
        CHECK(s->version == 3);
        std::string out = tempPath("scene.json");
        CHECK(saveScene(*s, out, &err));
        auto r = loadScene(out, &err);
        CHECK(r.has_value());
        if (r) checkSceneEqual(*s, *r);
        unlink(out.c_str());
    }

    std::string err;
    auto s = loadScene(sourcePath("scenes/cumulus.json"), &err);
    CHECK(s.has_value());
    if (!s) return;
    CHECK(s->name == "cumulus");
    CHECK(s->cameras.size() == 3);
    CHECK(s->camera("") == &s->cameras[0]);
    CHECK(s->camera("front") == &s->cameras[0]);
    CHECK(s->camera("side") != nullptr);
    CHECK(s->camera("below") != nullptr);
    CHECK(s->camera("nope") == nullptr);
    CHECK(s->volume.density == "cumulus");
    const CloudDesc& c = s->volume.cloud;
    CHECK(c.lobes.size() == 11);
    CHECK(c.lobes.size() >= 2 && eq4(c.lobes[1], simd_make_float4(-230, 110, 30, 150)));
    CHECK(eq3(c.base_center, simd_make_float3(0, 90, 0)));
    CHECK(c.shear.x == 0.08f && c.shear.y == 0.02f);
    CHECK(c.erosion_octaves == 5 && c.seed == 1);
    CHECK(s->medium.sigma_t_max == 0.05f);
    CHECK_NEAR(simd_length(s->light.sun_direction), 1.0, 1e-6);

    // The held-out scene expands {"generator_seed": 2500} with makeRandomCumulus at load time.
    auto h = loadScene(sourcePath("scenes/cumulus_heldout.json"), &err);
    CHECK(h.has_value());
    if (h) CHECK(cloudEqual(h->volume.cloud, makeRandomCumulus(2500, h->volume)));
}

// Loads `file` as mutable containers, applies `mutate`, writes it and tries to load it.
std::optional<Scene> loadMutated(void (^mutate)(NSMutableDictionary*), std::string* err,
                                 const char* file = "scenes/cumulus.json") {
    NSData* data = [NSData dataWithContentsOfFile:@(sourcePath(file).c_str())];
    NSMutableDictionary* root = [NSJSONSerialization JSONObjectWithData:data
                                                               options:NSJSONReadingMutableContainers
                                                                 error:nil];
    mutate(root);
    std::string path = tempPath("mutated.json");
    [[NSJSONSerialization dataWithJSONObject:root options:0 error:nil] writeToFile:@(path.c_str()) atomically:YES];
    auto s = loadScene(path, err);
    unlink(path.c_str());
    return s;
}

void testSceneRejects() {
    std::string err;
    auto s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"bogus"] = @1; }, &err);
    CHECK(!s.has_value());
    CHECK(err.find("volume.bogus") != std::string::npos);

    err.clear();
    s = loadMutated(^(NSMutableDictionary* r) { r[@"cameras"][1][@"fov"] = @1; }, &err);
    CHECK(!s.has_value());
    CHECK(err.find("cameras[1].fov") != std::string::npos);

    err.clear();
    s = loadMutated(^(NSMutableDictionary* r) { r[@"medium"][@"albedo"] = @1.5; }, &err);
    CHECK(!s.has_value());
    CHECK(err.find("albedo") != std::string::npos);

    // Cloud object: unknown key, too many lobes, non-positive radii, malformed lobe.
    err.clear();
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"cloud"][@"coverage"] = @0.5; }, &err);
    CHECK(!s.has_value());
    CHECK(err.find("volume.cloud.coverage") != std::string::npos);

    err.clear();
    s = loadMutated(
        ^(NSMutableDictionary* r) {
            NSMutableArray* lobes = r[@"volume"][@"cloud"][@"lobes"];
            while (lobes.count <= CNR_MAX_LOBES) [lobes addObject:@[ @0, @100, @0, @90 ]];
        },
        &err);
    CHECK(!s.has_value());
    CHECK(err.find("lobes") != std::string::npos);

    s = loadMutated(
        ^(NSMutableDictionary* r) {
            NSMutableArray* lobes = r[@"volume"][@"cloud"][@"lobes"];
            while (lobes.count < CNR_MAX_LOBES) [lobes addObject:@[ @0, @100, @0, @90 ]];
        },
        &err);
    CHECK(s.has_value() && s->volume.cloud.lobes.size() == CNR_MAX_LOBES);

    err.clear();
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"cloud"][@"lobes"][2][3] = @0; }, &err);
    CHECK(!s.has_value());
    CHECK(err.find("lobes[2]") != std::string::npos);
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"cloud"][@"lobes"][0][3] = @-10; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"cloud"][@"radius"] = @0; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"cloud"][@"edge"] = @0; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"cloud"][@"lobe_blend"] = @-1; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"cloud"][@"lobes"] = @[]; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"cloud"][@"lobes"][0] = @[ @0, @1, @2 ]; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"cloud"][@"height"] = @100; }, &err);
    CHECK(!s.has_value());  // height must exceed anvil_height
    s = loadMutated(^(NSMutableDictionary* r) { [r[@"volume"][@"cloud"] removeObjectForKey:@"seed"]; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { [r[@"volume"] removeObjectForKey:@"cloud"]; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"constant_density"] = @0.5; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(
        ^(NSMutableDictionary* r) {
            r[@"volume"][@"cloud"] = [@{@"generator_seed" : @2500, @"seed" : @1} mutableCopy];
        },
        &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"version"] = @2; }, &err);
    CHECK(!s.has_value());

    s = loadMutated(^(NSMutableDictionary* r) { r[@"medium"][@"hg_g"] = @1.0; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"cameras"][1][@"name"] = @"front"; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"grid"] = @[ @256, @0, @256 ]; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"grid"] = @[ @256, @128 ]; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"density"] = @"procedural"; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"light"][@"sun_direction"] = @[ @0, @0, @0 ]; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"film"][@"width"] = @0; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"cameras"] = @[]; }, &err);
    CHECK(!s.has_value());

    // Constant scenes: no cloud object; constant_density optional (default 1).
    const char* homogeneous = "scenes/homogeneous_test.json";
    s = loadMutated(^(NSMutableDictionary* r) { [r[@"volume"] removeObjectForKey:@"constant_density"]; }, &err,
                    homogeneous);
    CHECK(s.has_value() && s->volume.constant_density == 1.0f);
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"cloud"] = @{@"generator_seed" : @1}; }, &err,
                    homogeneous);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"constant_density"] = @1.5; }, &err, homogeneous);
    CHECK(!s.has_value());
}

void testRandomCumulus() {
    VolumeDesc volume;  // [-500, 0, -500] - [500, 800, 500], grid 256 x 192 x 256
    const CloudDesc a = makeRandomCumulus(1234, volume);
    CHECK(cloudEqual(a, makeRandomCumulus(1234, volume)));
    CHECK(!cloudEqual(a, makeRandomCumulus(1235, volume)));
    CHECK(a.seed == 1234);

    for (uint32_t seed : {1000u, 1001u, 1007u, 1500u, 1999u, 2000u, 2500u, 2999u}) {
        for (uint32_t k = 0; k < 8; ++k) {
            const uint32_t sd = seed + k * 997u;
            const CloudDesc c = makeRandomCumulus(sd, volume);
            CHECK(c.seed == sd);
            CHECK(c.lobes.size() >= 4 && c.lobes.size() <= 12);
            bool radii = true;
            for (simd_float4 l : c.lobes) radii = radii && l.w >= 70.0f && l.w <= 220.0f;
            CHECK(radii);
            CHECK(std::fabs(c.shear.x) <= 0.3f && std::fabs(c.shear.y) <= 0.3f);
            CHECK(c.erosion_amplitude >= 20.0f && c.erosion_amplitude <= 50.0f);
            CHECK(c.erosion_feature_size >= 35.0f && c.erosion_feature_size <= 90.0f);
            CHECK(c.edge >= 8.0f && c.edge <= 35.0f);
            CHECK(c.height > c.anvil_height && c.radius > 0 && c.lobe_blend > 0 && c.base_softness > 0);
            simd_float3 lo, hi;
            cloudWorldBounds(c, &lo, &hi);
            CHECK(simd_all(lo >= volume.bounds_min) && simd_all(hi <= volume.bounds_max));
            if (!(simd_all(lo >= volume.bounds_min) && simd_all(hi <= volume.bounds_max)))
                std::fprintf(stderr, "seed %u bounds [%g %g %g]-[%g %g %g]\n", sd, lo.x, lo.y, lo.z, hi.x, hi.y,
                             hi.z);
        }
    }
}

void testRenderConfigRoundTrip() {
    RenderConfig density;
    density.renderer = "density";
    density.step_scale = 0.75f;
    density.max_steps = 300;
    density.min_transmittance = 0.0f;
    density.dither = false;
    density.empty_skip = false;

    RenderConfig analytic;
    analytic.renderer = "analytic";
    analytic.step_scale = 2.0f;
    analytic.max_steps = 1000;
    analytic.min_transmittance = 0.05f;
    analytic.dither = false;
    analytic.empty_skip = true;
    analytic.shadow_steps = 9;
    analytic.shadow_distance = 250.0f;
    analytic.ambient_steps = 3;
    analytic.octaves = 2;
    analytic.ms_a = 0.25f;
    analytic.ms_b = 0.7f;
    analytic.ms_c = 0.15f;
    analytic.ambient = false;

    RenderConfig pt;
    pt.renderer = "pt";
    pt.spp = 64;
    pt.seed = 7;
    pt.max_bounces = 1;
    pt.rr_start_bounce = 5;

    RenderConfig neural;
    neural.renderer = "neural";
    neural.model = "models/test.cnrm";
    neural.inference = "shader_ml";
    neural.step_scale = 1.5f;
    neural.max_steps = 256;
    neural.min_transmittance = 0.02f;
    neural.dither = false;
    neural.empty_skip = false;

    std::string path = tempPath("config.json");
    for (const RenderConfig& c : {density, analytic, pt, neural}) {
        std::string err;
        CHECK(saveRenderConfig(c, path, &err));
        auto r = loadRenderConfig(path, &err);
        CHECK(r.has_value());
        if (!r) continue;
        CHECK(r->renderer == c.renderer);
        CHECK(r->step_scale == c.step_scale);
        CHECK(r->max_steps == c.max_steps);
        CHECK(r->min_transmittance == c.min_transmittance);
        CHECK(r->dither == c.dither);
        CHECK(r->empty_skip == c.empty_skip);
        CHECK(r->shadow_steps == c.shadow_steps);
        CHECK(r->shadow_distance == c.shadow_distance);
        CHECK(r->ambient_steps == c.ambient_steps);
        CHECK(r->octaves == c.octaves);
        CHECK(r->ms_a == c.ms_a);
        CHECK(r->ms_b == c.ms_b);
        CHECK(r->ms_c == c.ms_c);
        CHECK(r->ambient == c.ambient);
        CHECK(r->spp == c.spp);
        CHECK(r->seed == c.seed);
        CHECK(r->max_bounces == c.max_bounces);
        CHECK(r->rr_start_bounce == c.rr_start_bounce);
        CHECK(r->model == c.model);
        CHECK(r->inference == c.inference);
    }

    MarchParams m = makeMarchParams(analytic);
    CHECK(m.step_scale == 2.0f && m.max_steps == 1000 && m.min_transmittance == 0.05f);
    CHECK(m.flags == CNR_MARCH_EMPTY_SKIP);
    CHECK(makeMarchParams(RenderConfig{}).flags == (CNR_MARCH_DITHER | CNR_MARCH_EMPTY_SKIP));
    CHECK(makeMarchParams(density).flags == 0u);

    std::string err;
    auto tryConfig = [&](NSString* json) {
        [[json dataUsingEncoding:NSUTF8StringEncoding] writeToFile:@(path.c_str()) atomically:YES];
        return loadRenderConfig(path, &err);
    };
    CHECK(!tryConfig(@"{\"renderer\":\"pt\",\"spp\":4,\"step_scale\":3}"));
    CHECK(err.find("step_scale") != std::string::npos);
    CHECK(!tryConfig(@"{\"renderer\":\"density\",\"primary_steps\":128}"));
    CHECK(err.find("primary_steps") != std::string::npos);
    CHECK(!tryConfig(@"{\"renderer\":\"density\",\"shadow_steps\":4}"));
    CHECK(!tryConfig(@"{\"renderer\":\"magic\"}"));
    CHECK(!tryConfig(@"{\"renderer\":\"analytic\",\"ambient\":1}"));
    CHECK(!tryConfig(@"{\"renderer\":\"analytic\",\"dither\":0}"));
    CHECK(!tryConfig(@"{\"renderer\":\"analytic\",\"shadow_distance\":-1}"));
    CHECK(!tryConfig(@"{\"renderer\":\"density\",\"step_scale\":0}"));
    CHECK(!tryConfig(@"{\"renderer\":\"density\",\"max_steps\":0}"));
    CHECK(!tryConfig(@"{\"renderer\":\"density\",\"min_transmittance\":1}"));
    CHECK(!tryConfig(@"{\"renderer\":\"neural\",\"model\":\"m.cnrm\",\"inference\":\"coreml\"}"));
    CHECK(!tryConfig(@"{\"renderer\":\"neural\",\"model\":\"m.cnrm\",\"shadow_steps\":4}"));
    CHECK(!tryConfig(@"{\"renderer\":\"neural\"}"));
    auto n = tryConfig(@"{\"renderer\":\"neural\",\"model\":\"m.cnrm\"}");
    CHECK(n.has_value() && n->inference == "scalar" && n->step_scale == 1.0f && n->max_steps == 512);
    auto d = tryConfig(@"{\"renderer\":\"density\"}");
    CHECK(d.has_value() && d->step_scale == 1.0f && d->max_steps == 512 && d->min_transmittance == 0.01f &&
          d->dither && d->empty_skip);
    unlink(path.c_str());

    auto shipped = loadRenderConfig(sourcePath("configs/density.json"), &err);
    CHECK(shipped.has_value() && shipped->renderer == "density" && shipped->step_scale == 1.0f &&
          shipped->max_steps == 512);
}

void testSceneGPU() {
    std::string err;
    auto s = loadScene(sourcePath("scenes/cumulus.json"), &err);
    CHECK(s.has_value());
    if (!s) return;
    for (const CameraDesc& cam : s->cameras) {
        SceneGPU g = makeSceneGPU(*s, cam);
        simd_float3 f = simd_normalize(cam.target - cam.position);
        for (int i = 0; i < 3; ++i) CHECK_NEAR(g.cam_forward[i], f[i], 1e-6);
        CHECK_NEAR(simd_length(g.cam_forward), 1.0, 1e-6);
        CHECK_NEAR(simd_length(g.cam_right), 1.0, 1e-6);
        CHECK_NEAR(simd_length(g.cam_up), 1.0, 1e-6);
        CHECK_NEAR(simd_dot(g.cam_forward, g.cam_right), 0.0, 1e-6);
        CHECK_NEAR(simd_dot(g.cam_forward, g.cam_up), 0.0, 1e-6);
        CHECK_NEAR(simd_dot(g.cam_right, g.cam_up), 0.0, 1e-6);
        CHECK(g.cam_up.y > 0);
        CHECK_NEAR(g.tan_half_vfov, std::tan(cam.vfov_deg * M_PI / 360.0), 1e-6);
        CHECK_NEAR(g.aspect, 960.0 / 540.0, 1e-6);
        CHECK(g.mip_levels == 9);
        CHECK_NEAR(g.voxel_min_extent, 1000.0 / 256.0, 1e-6);
        CHECK(g.width == 960 && g.height == 540);
        CHECK_NEAR(simd_length(g.sun_direction), 1.0, 1e-6);
    }
    DensityBakeParams p = makeDensityBakeParams(*s);
    const CloudDesc& c = s->volume.cloud;
    CHECK(p.grid_x == 256 && p.grid_y == 192 && p.grid_z == 256);
    CHECK(p.density_type == CNR_DENSITY_CUMULUS);
    CHECK(p.lobe_count == c.lobes.size());
    bool lobes = true;
    for (size_t i = 0; i < c.lobes.size(); ++i) lobes = lobes && eq4(p.lobes[i], c.lobes[i]);
    CHECK(lobes);
    CHECK(eq3(p.base_center, c.base_center));
    CHECK(p.seed == c.seed && p.erosion_octaves == c.erosion_octaves);
    CHECK(p.shear_x == c.shear.x && p.shear_z == c.shear.y);
    CHECK(p.cloud_height == c.height && p.cloud_radius == c.radius && p.anvil_height == c.anvil_height);
    CHECK(p.anvil_spread == c.anvil_spread && p.lobe_blend == c.lobe_blend && p.edge == c.edge);
    CHECK(p.base_softness == c.base_softness && p.base_fade_height == c.base_fade_height);
    CHECK(p.erosion_feature_size == c.erosion_feature_size && p.erosion_amplitude == c.erosion_amplitude);
    auto h = loadScene(sourcePath("scenes/homogeneous_test.json"), &err);
    CHECK(h.has_value());
    if (!h) return;
    DensityBakeParams hp = makeDensityBakeParams(*h);
    CHECK(hp.density_type == CNR_DENSITY_CONSTANT && hp.constant_density == 1.0f && hp.lobe_count == 0);
    // Right-handed, +Y up: looking along +Z (homogeneous_test "front"), right is -X.
    SceneGPU g = makeSceneGPU(*h, *h->camera("front"));
    CHECK_NEAR(g.cam_right.x, -1.0, 1e-6);
    CHECK_NEAR(g.cam_up.y, 1.0, 1e-6);
}

void testPFM() {
    Image img;
    img.width = 7;
    img.height = 5;
    img.rgb.resize(7 * 5 * 3);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (size_t i = 0; i < img.rgb.size(); ++i) img.rgb[i] = u(rng) * std::pow(10.0f, float(int(i % 13) - 6));
    img.rgb[0] = 3.0e38f;
    img.rgb[1] = -1.0e-30f;
    img.rgb[2] = 0.0f;
    img.rgb[3] = -0.0f;

    std::string path = tempPath("img.pfm"), err;
    CHECK(writePFM(img, path, &err));
    Image back;
    CHECK(readPFM(path, &back, &err));
    CHECK(back.width == img.width && back.height == img.height);
    CHECK(back.rgb.size() == img.rgb.size() &&
          std::memcmp(back.rgb.data(), img.rgb.data(), img.rgb.size() * sizeof(float)) == 0);

    // Bottom-to-top storage: the first stored row is the image's last row.
    NSData* data = [NSData dataWithContentsOfFile:@(path.c_str())];
    const char* header = "PF\n7 5\n-1.0\n";
    CHECK(data.length == std::strlen(header) + img.rgb.size() * 4);
    CHECK(std::memcmp(data.bytes, header, std::strlen(header)) == 0);
    CHECK(std::memcmp((const char*)data.bytes + std::strlen(header), img.pixel(0, 4), 7 * 3 * 4) == 0);

    // Truncated raster is rejected.
    [[data subdataWithRange:NSMakeRange(0, data.length - 5)] writeToFile:@(path.c_str()) atomically:YES];
    err.clear();
    CHECK(!readPFM(path, &back, &err));
    CHECK(err.find("truncated") != std::string::npos);

    // Extra header whitespace is accepted.
    NSMutableData* spaced = [NSMutableData dataWithBytes:"PF\n 7  5 \n-1.0\n" length:15];
    [spaced appendData:[data subdataWithRange:NSMakeRange(std::strlen(header), img.rgb.size() * 4)]];
    [spaced writeToFile:@(path.c_str()) atomically:YES];
    CHECK(readPFM(path, &back, &err));
    CHECK(std::memcmp(back.rgb.data(), img.rgb.data(), img.rgb.size() * sizeof(float)) == 0);

    [[@"P6\n1 1\n255\n" dataUsingEncoding:NSUTF8StringEncoding] writeToFile:@(path.c_str()) atomically:YES];
    CHECK(!readPFM(path, &back, &err));
    unlink(path.c_str());
}

double srgb(double t) { return t <= 0.0031308 ? 12.92 * t : 1.055 * std::pow(t, 1.0 / 2.4) - 0.055; }

// Inverse of displayMap at ev = 0.
float linearForDisplay(double d) {
    double t = d <= 0.04045 ? d / 12.92 : std::pow((d + 0.055) / 1.055, 2.4);
    return float(t / (1.0 - t));
}

void testDisplayMap() {
    CHECK(displayMap(0.0f, 0.0f) == 0.0f);
    CHECK(displayMap(-3.0f, 0.0f) == 0.0f);
    CHECK_NEAR(displayMap(1.0e9f, 0.0f), 1.0, 1e-6);
    CHECK_NEAR(displayMap(INFINITY, 0.0f), 1.0, 1e-6);
    CHECK_NEAR(displayMap(1.0f, 0.0f), srgb(0.5), 1e-6);
    CHECK_NEAR(displayMap(1.0f, 0.0f), 0.735357, 1e-5);
    CHECK_NEAR(displayMap(0.5f, 1.0f), displayMap(1.0f, 0.0f), 1e-7);
    CHECK_NEAR(displayMap(0.001f, 0.0f), 12.92 * (0.001 / 1.001), 1e-6);
    CHECK_NEAR(displayMap(linearForDisplay(0.3), 0.0f), 0.3, 1e-5);
}

Image constantImage(uint32_t w, uint32_t h, float v) {
    Image img;
    img.width = w;
    img.height = h;
    img.rgb.assign(size_t(w) * h * 3, v);
    return img;
}

void testMetrics() {
    Image ref = constantImage(16, 16, 1.0f), test = constantImage(16, 16, 1.1f);
    CHECK_NEAR(relMSE(test, ref, 0.0f), (0.1 * 0.1) / (1.0 + 0.01), 1e-6);
    CHECK_NEAR(relMSE(test, ref, 1.0f), (0.2 * 0.2) / (4.0 + 0.01), 1e-6);
    CHECK(relMSE(ref, ref, 0.0f) == 0.0);
    CHECK(std::isnan(relMSE(ref, constantImage(8, 16, 1.0f), 0.0f)));

    // Constant display offset: ref displays as 0, test as d everywhere.
    Image zero = constantImage(16, 16, 0.0f), one = constantImage(16, 16, 1.0f);
    double d = displayMap(1.0f, 0.0f);
    CHECK(std::isinf(psnrDisplay(one, one, 0.0f)));
    CHECK_NEAR(psnrDisplay(one, zero, 0.0f), -20.0 * std::log10(d), 1e-4);

    // Patterned image and its display-space inverse.
    Image a = constantImage(32, 24, 0.0f), inv = a;
    for (uint32_t y = 0; y < a.height; ++y)
        for (uint32_t x = 0; x < a.width; ++x)
            for (int c = 0; c < 3; ++c) {
                double v = 0.05 + 0.9 * (((x / 4 + y / 4) % 2) * 0.6 + 0.4 * double(x + c) / (a.width + 2));
                a.pixel(x, y)[c] = linearForDisplay(v);
                inv.pixel(x, y)[c] = linearForDisplay(1.0 - v);
            }
    CHECK_NEAR(ssimDisplay(a, a, 0.0f), 1.0, 1e-9);
    CHECK(ssimDisplay(a, inv, 0.0f) < 0.5);
    CHECK_NEAR(ssimDisplay(a, inv, 0.0f), ssimDisplay(inv, a, 0.0f), 1e-12);
    CHECK(std::isnan(ssimDisplay(constantImage(10, 10, 0.0f), constantImage(10, 10, 0.0f), 0.0f)));

    // Hand-computed: constant images have zero variance, so SSIM = (2 mx my + C1) / (mx^2 + my^2 + C1).
    // With display values 0 and d: C1 / (d^2 + C1). An 11x11 image has exactly one valid window.
    const double C1 = 1e-4;
    CHECK_NEAR(ssimDisplay(constantImage(11, 11, 1.0f), constantImage(11, 11, 0.0f), 0.0f), C1 / (d * d + C1), 1e-9);
    CHECK_NEAR(ssimDisplay(one, zero, 0.0f), C1 / (d * d + C1), 1e-9);

    // Heat maps: zero error is blue, large error is red.
    Image hm = absErrorHeatmap(ref, ref, 0.0f, 1.0f);
    CHECK(hm.width == 16 && hm.rgb[0] == 0.0f && hm.rgb[1] == 0.0f && hm.rgb[2] == 1.0f);
    hm = absErrorHeatmap(constantImage(16, 16, 5.0f), ref, 0.0f, 1.0f);
    CHECK(hm.rgb[0] == 1.0f && hm.rgb[1] == 0.0f && hm.rgb[2] == 0.0f);
    hm = relErrorHeatmap(constantImage(16, 16, 0.0f), constantImage(16, 16, 0.1f), 0.0f);  // rel error 0.5
    CHECK_NEAR(hm.rgb[0], 0.0, 1e-5);  // ramp midpoint is green
    CHECK_NEAR(hm.rgb[1], 1.0, 1e-5);
    CHECK_NEAR(hm.rgb[2], 0.0, 1e-5);

    std::string path = tempPath("img.png"), err;
    CHECK(writePNG(a, 0.0f, path, &err));
    CHECK(writeDisplayPNG(hm, path, &err));
    unlink(path.c_str());
}

}  // namespace

int main() {
    @autoreleasepool {
        testSceneRoundTrip();
        testSceneRejects();
        testRandomCumulus();
        testRenderConfigRoundTrip();
        testSceneGPU();
        testPFM();
        testDisplayMap();
        testMetrics();
    }
    return checkSummary();
}
