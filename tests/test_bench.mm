// Benchmark framework: statistics, reference-spp schedule, experiment parsing, record schema,
// and a small GPU benchmark.
#import <Foundation/Foundation.h>

#include "bench.h"
#include "check.h"
#include "density_volume.h"
#include "metal_context.h"
#include "suite.h"

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
    CHECK(ex.scenes.size() == 3 && ex.cameras.size() == 2 && ex.methods.size() == 3);
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
    writeFile(configPath, R"({"renderer":"analytic","primary_steps":32,"shadow_steps":8,"ambient_steps":2,)"
                          R"("octaves":2,"ms_a":0.5,"ms_b":0.5,"ms_c":0.5,"ambient":true})");
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
        testGpuBenchmarkAndRecord();
    }
    return checkSummary();
}
