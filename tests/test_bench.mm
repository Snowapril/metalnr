// Benchmark framework: statistics, reference-spp schedule, experiment parsing, record schema,
// and a small GPU benchmark.
#import <Foundation/Foundation.h>

#include "bench.h"
#include "check.h"
#include "density_volume.h"
#include "metal_context.h"
#include "suite.h"
#include "sweep.h"

#include <cstdio>
#include <string>
#include <unistd.h>

namespace {

std::string tempPath(const std::string& name) {
    return std::string(NSTemporaryDirectory().UTF8String) + "cloudnr_test_bench_" + std::to_string(getpid()) + "_" +
           name;
}

void writeFile(const std::string& path, const std::string& text) {
    std::string error;
    CHECK(writeTextFile(path, text, &error));
}

void testStatistics() {
    // Nearest rank on 1..20: p5 -> rank 1, median -> rank 10, p95 -> rank 19.
    std::vector<double> v;
    for (int i = 20; i >= 1; --i) v.push_back(i);
    TimingStats s = summarizeTimes(v);
    CHECK_NEAR(s.p5, 1.0, 0.0);
    CHECK_NEAR(s.median, 10.0, 0.0);
    CHECK_NEAR(s.p95, 19.0, 0.0);
    // Odd count: exact median; NaN (failed frames) are ignored.
    s = summarizeTimes({5.0, NAN, 1.0, 3.0});
    CHECK_NEAR(s.median, 3.0, 0.0);
    CHECK_NEAR(s.p95, 5.0, 0.0);
    CHECK(std::isnan(summarizeTimes({}).median));

    TimingSummary t = combineRuns({{4.0, 1.0, 9.0}, {2.0, 0.5, 7.0}, {3.0, 2.0, 8.0}});
    CHECK_NEAR(t.headline.median, 3.0, 0.0);
    CHECK_NEAR(t.headline.p5, 1.0, 0.0);
    CHECK_NEAR(t.headline.p95, 8.0, 0.0);
    CHECK(t.runs.size() == 3);
}

void testReferenceSchedule() {
    // At least doubles; jumps by the 1/spp error scaling; power of two; capped.
    CHECK(nextReferenceSpp(1024, 1.1e-4, 1e-4, 16384) == 2048);
    CHECK(nextReferenceSpp(1024, 1e-3, 1e-4, 16384) == 16384);
    CHECK(nextReferenceSpp(1024, 3e-4, 1e-4, 16384) == 4096);
    CHECK(nextReferenceSpp(8192, 1e-3, 1e-5, 16384) == 16384);
}

void testExperimentParsing() {
    std::string error;
    Experiment ex;
    CHECK(parseExperiment(std::string(CLOUDNR_SOURCE_ROOT) + "/experiments/eval.json", &ex, &error));
    CHECK(ex.scenes.size() == 2 && ex.cameras.size() == 2 && ex.methods.size() == 4);
    CHECK(ex.warmup == 30 && ex.frames == 300 && ex.runs == 3);
    CHECK_NEAR(ex.abs_gate, 1e-4, 1e-12);

    const std::string base =
        R"("name":"x","scenes":["s"],"cameras":["c"],"methods":["m"],)"
        R"("reference":{"config":"p","initial_spp":4,"max_spp":8,"abs_gate":1e-4,"rel_gate":0.1},)"
        R"("equal_quality":{"seed":3,"max_spp":8})";
    const std::string path = tempPath("exp.json");
    writeFile(path, "{" + base + R"(,"timing":{"warmup":0,"frames":1,"runs":1}})");
    CHECK(parseExperiment(path, &ex, &error));
    writeFile(path, "{" + base + "}");  // missing timing
    CHECK(!parseExperiment(path, &ex, &error) && error.find("timing") != std::string::npos);
    writeFile(path, "{" + base + R"(,"timing":{"warmup":0,"frames":1,"runs":1},"extra":1})");
    CHECK(!parseExperiment(path, &ex, &error) && error.find("extra") != std::string::npos);
    writeFile(path, "{" + base + R"(,"timing":{"warmup":0,"frames":0,"runs":1}})");
    CHECK(!parseExperiment(path, &ex, &error));
    std::remove(path.c_str());
}

void testSweepSunDirection() {
    CameraDesc cam;
    cam.position = {0, 350, -2200};
    cam.target = {0, 330, 0};
    const simd_float3 forward = simd_normalize(cam.target - cam.position);
    const float elevation = 20.0f;
    for (double angle : {0.0, 45.0, 90.0, 135.0, 180.0}) {
        const simd_float3 sun = sweepSunDirection(cam, angle, elevation);
        CHECK_NEAR(simd_length(sun), 1.0, 1e-6);
        CHECK_NEAR(std::asin(sun.y) * 180.0 / M_PI, elevation, 1e-4);
    }
    // 0 deg: sun behind the camera (lights the visible side); 180 deg: behind the cloud.
    CHECK(simd_dot(sweepSunDirection(cam, 0.0, elevation), forward) < 0.0f);
    CHECK(simd_dot(sweepSunDirection(cam, 180.0, elevation), forward) > 0.0f);
    // Exact horizontal components: camera looks along +Z, so fwd_h = +Z and right_h = cross(+Z, +Y) = -X.
    const float c = std::cos(elevation * float(M_PI) / 180.0f);
    const simd_float3 s0 = sweepSunDirection(cam, 0.0, elevation);
    CHECK_NEAR(s0.x, 0.0, 1e-6);
    CHECK_NEAR(s0.z, -c, 1e-6);
    const simd_float3 s90 = sweepSunDirection(cam, 90.0, elevation);
    CHECK_NEAR(s90.x, -c, 1e-6);
    CHECK_NEAR(s90.z, 0.0, 1e-6);
}

void testMaskedLuminance() {
    // 2x2 grey images (lum = value); mask covers pixels 0, 1, 3.
    auto grey = [](std::vector<float> v) {
        Image image;
        image.width = 2;
        image.height = 2;
        for (float x : v) image.rgb.insert(image.rgb.end(), {x, x, x});
        return image;
    };
    const Image reference = grey({1.0f, 2.0f, 100.0f, 3.0f});
    const Image test = grey({1.0f, 1.0f, 0.0f, 4.0f});
    const std::vector<uint8_t> mask = {1, 1, 0, 1};
    MaskedLuminance m = maskedLuminance(test, reference, mask);
    CHECK(m.pixels == 3);
    CHECK_NEAR(m.score, 6.0 / 6.0, 1e-12);                  // (1+1+4) / (1+2+3)
    CHECK_NEAR(m.rmse, std::sqrt(2.0 / 3.0), 1e-12);        // diffs 0, -1, 1
    // Rec.709 weights.
    const float rgb[3] = {1.0f, 2.0f, 4.0f};
    CHECK_NEAR(luminance709(rgb), 0.2126 + 2 * 0.7152 + 4 * 0.0722, 1e-12);
    // Half brightness over the mask -> score 0.5.
    const Image half = grey({0.5f, 1.0f, 7.0f, 1.5f});
    CHECK_NEAR(maskedLuminance(half, reference, mask).score, 0.5, 1e-12);
    // Empty mask and size mismatch -> NaN.
    m = maskedLuminance(test, reference, {0, 0, 0, 0});
    CHECK(m.pixels == 0 && std::isnan(m.score) && std::isnan(m.rmse));
    CHECK(std::isnan(maskedLuminance(test, reference, {1, 1}).score));
}

void testSweepExperimentParsing() {
    std::string error;
    SweepExperiment ex;
    CHECK(parseSweepExperiment(std::string(CLOUDNR_SOURCE_ROOT) + "/experiments/sweep.json", &ex, &error));
    CHECK(ex.angles_deg.size() == 13 && ex.angles_deg.front() == 0.0 && ex.angles_deg.back() == 180.0);
    CHECK(ex.width == 480 && ex.height == 270 && ex.methods.size() == 4);
    CHECK(ex.reference_spp_per_half == 4096 && ex.warmup == 30 && ex.frames == 300 && ex.runs == 3);
    CHECK_NEAR(ex.elevation_deg, 20.0, 0.0);

    const std::string head = R"("name":"x","scene":"s.json","camera":"front","angles_deg":[0,90],"elevation_deg":20,)"
                             R"("width":8,"height":4,"methods":["m.json"],)";
    const std::string ref = R"("reference":{"config":"p.json","spp_per_half":4},)";
    const std::string timing = R"("timing":{"warmup":0,"frames":1,"runs":1})";
    const std::string path = tempPath("sweep.json");
    writeFile(path, "{" + head + ref + timing + "}");
    CHECK(parseSweepExperiment(path, &ex, &error));
    CHECK(ex.angles_deg.size() == 2 && ex.width == 8 && ex.height == 4);
    writeFile(path, "{" + head + timing + "}");  // missing reference
    CHECK(!parseSweepExperiment(path, &ex, &error) && error.find("reference") != std::string::npos);
    writeFile(path, "{" + head + R"("reference":{"config":"p.json"},)" + timing + "}");  // missing spp_per_half
    CHECK(!parseSweepExperiment(path, &ex, &error) && error.find("spp_per_half") != std::string::npos);
    writeFile(path, "{" + head + ref + R"("timing":{"warmup":0,"runs":1}})");  // missing frames
    CHECK(!parseSweepExperiment(path, &ex, &error) && error.find("frames") != std::string::npos);
    writeFile(path, "{" + head + ref + timing + R"(,"extra":1})");
    CHECK(!parseSweepExperiment(path, &ex, &error) && error.find("extra") != std::string::npos);
    writeFile(path, R"({"name":"x","scene":"s.json","angles_deg":[0],"elevation_deg":20,"width":8,"height":4,)"
                    R"("methods":["m.json"],)" + ref + timing + "}");  // missing camera
    CHECK(!parseSweepExperiment(path, &ex, &error) && error.find("camera") != std::string::npos);
    std::string bad = head;
    bad.replace(bad.find("[0,90]"), 6, "[]");
    writeFile(path, "{" + bad + ref + timing + "}");  // empty angles
    CHECK(!parseSweepExperiment(path, &ex, &error) && error.find("angles_deg") != std::string::npos);
    std::remove(path.c_str());
}

Scene smallScene() {
    Scene s;
    s.name = "bench_test";
    s.width = 32;
    s.height = 32;
    CameraDesc cam;
    cam.name = "front";
    cam.position = {0, 0, -5};
    cam.target = {0, 0, 0};
    cam.vfov_deg = 30;
    s.cameras.push_back(cam);
    s.volume.bounds_min = {-1, -1, -1};
    s.volume.bounds_max = {1, 1, 1};
    s.volume.grid = {16, 16, 16};
    s.volume.density = "constant";
    s.volume.constant_density = 1.0f;
    s.medium = {1.0f, 0.8f, 0.3f};
    return s;
}

void testGpuBenchmarkAndRecord() {
    std::string error;
    auto context = MetalContext::create(&error);
    CHECK(context != nullptr);
    if (!context) return;
    const Scene scene = smallScene();
    auto volume = DensityVolume::bake(*context, scene, &error);
    CHECK(volume != nullptr);
    if (!volume) return;

    const std::string configPath = tempPath("analytic.json");
    writeFile(configPath, R"({"renderer":"analytic","step_scale":1.0,"max_steps":256,"min_transmittance":0.01,)"
                          R"("dither":true,"empty_skip":true,"shadow_steps":6,"shadow_distance":0,)"
                          R"("ambient_steps":2,"octaves":2,"ms_a":0.4,"ms_b":0.35,"ms_c":0.1,"ambient":true})");
    auto config = loadRenderConfig(configPath, &error);
    CHECK(config.has_value());
    if (!config) return;

    BenchOptions options;
    options.warmup = 2;
    options.frames = 5;
    options.runs = 2;
    BenchResult result;
    Image image;
    CHECK(runBenchmark(*context, scene, *volume, scene.cameras[0], configPath, *config, options, &result, &image,
                       &error));
    CHECK(result.gpu_ms.runs.size() == 2);
    CHECK(result.gpu_ms.headline.median > 0.0 && result.gpu_ms.headline.p5 <= result.gpu_ms.headline.p95);
    CHECK(result.cpu_prep_ms.headline.median > 0.0);
    CHECK(result.memory_renderer > 0 && result.memory_density > 0);
    CHECK(image.width == 32 && image.height == 32);

    // Deterministic renderer: a second benchmark yields the identical image -> perfect metrics.
    BenchResult again;
    Image image2;
    CHECK(runBenchmark(*context, scene, *volume, scene.cameras[0], configPath, *config, options, &again, &image2,
                       &error));
    attachMetrics(&again, image2, image, 0.0f, "self");
    CHECK_NEAR(again.relMSE, 0.0, 0.0);
    CHECK_NEAR(again.ssim, 1.0, 1e-12);

    // The record has every protocol §6 key and survives a JSON round trip.
    NSDictionary* record = benchRecordDictionary(again);
    const std::string json = toJsonString(record);
    NSDictionary* parsed = [NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:json.data()
                                                                                  length:json.size()]
                                                           options:0
                                                             error:nil];
    CHECK([parsed isKindOfClass:[NSDictionary class]]);
    for (NSString* key in @[ @"renderer_version", @"renderer", @"config", @"scene", @"scene_version", @"camera",
                             @"width", @"height", @"gpu_name", @"os_version", @"spp_or_quality", @"seed", @"gpu_ms",
                             @"cpu_prep_ms", @"memory_bytes", @"excluded_costs_ms", @"model", @"metrics",
                             @"reference_id", @"bias" ]) {
        if (!parsed[key]) std::fprintf(stderr, "missing record key %s\n", key.UTF8String);
        CHECK(parsed[key] != nil);
    }
    CHECK([parsed[@"gpu_ms"][@"runs"] count] == 2);
    CHECK([parsed[@"metrics"][@"relMSE"] doubleValue] == 0.0);
    CHECK([parsed[@"config"][@"octaves"] intValue] == 2);
    CHECK(parsed[@"model"] == [NSNull null]);
    std::remove(configPath.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    ensureValidationEnvironment(argc, argv);
    @autoreleasepool {
        testStatistics();
        testReferenceSchedule();
        testExperimentParsing();
        testSweepSunDirection();
        testMaskedLuminance();
        testSweepExperimentParsing();
        testGpuBenchmarkAndRecord();
    }
    return checkSummary();
}
