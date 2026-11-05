// Host-only tests: scene/config JSON, camera basis, PFM, display mapping, metrics. No GPU.
#include "check.h"
#include "image_io.h"
#include "metrics.h"
#include "scene.h"

#import <Foundation/Foundation.h>

#include <cmath>
#include <cstring>
#include <random>
#include <unistd.h>

namespace {

std::string tempPath(const std::string& name) {
    return std::string(NSTemporaryDirectory().UTF8String) + "cloudnr_test_" + std::to_string(getpid()) + "_" + name;
}

std::string sourcePath(const std::string& rel) { return std::string(CLOUDNR_SOURCE_ROOT) + "/" + rel; }

bool eq3(simd_float3 a, simd_float3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

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
    CHECK(va.density_seed == vb.density_seed);
    CHECK(va.coverage == vb.coverage);
    CHECK(va.erosion == vb.erosion);
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

void testSceneRoundTrip() {
    for (const char* file : {"scenes/eval_medium.json", "scenes/homogeneous_test.json"}) {
        std::string err;
        auto s = loadScene(sourcePath(file), &err);
        CHECK(s.has_value());
        if (!s) {
            std::fprintf(stderr, "%s\n", err.c_str());
            continue;
        }
        std::string out = tempPath("scene.json");
        CHECK(saveScene(*s, out, &err));
        auto r = loadScene(out, &err);
        CHECK(r.has_value());
        if (r) checkSceneEqual(*s, *r);
        unlink(out.c_str());
    }

    std::string err;
    auto s = loadScene(sourcePath("scenes/eval_medium.json"), &err);
    if (!s) return;
    CHECK(s->name == "eval_medium");
    CHECK(s->cameras.size() == 2);
    CHECK(s->camera("") == &s->cameras[0]);
    CHECK(s->camera("backlit") != nullptr);
    CHECK(s->camera("nope") == nullptr);
    CHECK(s->volume.density == "procedural");
    CHECK(s->medium.sigma_t_max == 0.04f);
    CHECK_NEAR(simd_length(s->light.sun_direction), 1.0, 1e-6);
}

// Loads eval_medium.json as mutable containers, applies `mutate`, writes it and tries to load it.
std::optional<Scene> loadMutated(void (^mutate)(NSMutableDictionary*), std::string* err) {
    NSData* data = [NSData dataWithContentsOfFile:@(sourcePath("scenes/eval_medium.json").c_str())];
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

    s = loadMutated(^(NSMutableDictionary* r) { r[@"medium"][@"hg_g"] = @1.0; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"cameras"][1][@"name"] = @"side"; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"grid"] = @[ @256, @0, @256 ]; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"grid"] = @[ @256, @128 ]; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"volume"][@"density"] = @"vdb"; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"light"][@"sun_direction"] = @[ @0, @0, @0 ]; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"film"][@"width"] = @0; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { r[@"cameras"] = @[]; }, &err);
    CHECK(!s.has_value());
    s = loadMutated(^(NSMutableDictionary* r) { [r[@"volume"] removeObjectForKey:@"constant_density"]; }, &err);
    CHECK(s.has_value() && s->volume.constant_density == 1.0f);
}

void testRenderConfigRoundTrip() {
    RenderConfig density;
    density.renderer = "density";
    density.primary_steps = 77;

    RenderConfig analytic;
    analytic.renderer = "analytic";
    analytic.primary_steps = 200;
    analytic.shadow_steps = 9;
    analytic.ambient_steps = 3;
    analytic.octaves = 2;
    analytic.ms_a = 0.25f;
    analytic.ms_b = 0.7f;
    analytic.ms_c = 0.1f;
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
    neural.primary_steps = 64;
    neural.shadow_steps = 8;

    std::string path = tempPath("config.json");
    for (const RenderConfig& c : {density, analytic, pt, neural}) {
        std::string err;
        CHECK(saveRenderConfig(c, path, &err));
        auto r = loadRenderConfig(path, &err);
        CHECK(r.has_value());
        if (!r) continue;
        CHECK(r->renderer == c.renderer);
        CHECK(r->primary_steps == c.primary_steps);
        CHECK(r->shadow_steps == c.shadow_steps);
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
    }

    std::string err;
    auto tryConfig = [&](NSString* json) {
        [[json dataUsingEncoding:NSUTF8StringEncoding] writeToFile:@(path.c_str()) atomically:YES];
        return loadRenderConfig(path, &err);
    };
    CHECK(!tryConfig(@"{\"renderer\":\"pt\",\"spp\":4,\"primary_steps\":3}"));
    CHECK(err.find("primary_steps") != std::string::npos);
    CHECK(!tryConfig(@"{\"renderer\":\"magic\"}"));
    CHECK(!tryConfig(@"{\"renderer\":\"analytic\",\"ambient\":1}"));
    auto d = tryConfig(@"{\"renderer\":\"density\",\"primary_steps\":128}");
    CHECK(d.has_value() && d->primary_steps == 128);
    unlink(path.c_str());

    auto shipped = loadRenderConfig(sourcePath("configs/density.json"), &err);
    CHECK(shipped.has_value() && shipped->renderer == "density" && shipped->primary_steps == 128);
}

void testSceneGPU() {
    std::string err;
    auto s = loadScene(sourcePath("scenes/eval_medium.json"), &err);
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
        CHECK_NEAR(g.tan_half_vfov, std::tan(20.0 * M_PI / 180.0), 1e-6);
        CHECK_NEAR(g.aspect, 960.0 / 540.0, 1e-6);
        CHECK(g.mip_levels == 9);
        CHECK_NEAR(g.voxel_min_extent, 3.125, 1e-6);
        CHECK(g.width == 960 && g.height == 540);
        CHECK_NEAR(simd_length(g.sun_direction), 1.0, 1e-6);
    }
    DensityBakeParams p = makeDensityBakeParams(*s);
    CHECK(p.grid_x == 256 && p.grid_y == 128 && p.grid_z == 256);
    CHECK(p.density_type == CNR_DENSITY_PROCEDURAL);
    CHECK(p.density_seed == 2);
    auto h = loadScene(sourcePath("scenes/homogeneous_test.json"), &err);
    CHECK(h.has_value() && makeDensityBakeParams(*h).density_type == CNR_DENSITY_CONSTANT);
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
        testRenderConfigRoundTrip();
        testSceneGPU();
        testPFM();
        testDisplayMap();
        testMetrics();
    }
    return checkSummary();
}
