#include "suite.h"

#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "bench.h"
#include "density_volume.h"
#include "image_io.h"
#include "metal_context.h"
#include "metrics.h"
#include "reference.h"
#include "scene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <map>
#include <set>

namespace {

NSString* ns(const std::string& s) { return [NSString stringWithUTF8String:s.c_str()]; }

int fail(const std::string& message) {
    std::fprintf(stderr, "cloudnr suite: %s\n", message.c_str());
    return 1;
}

std::string stem(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    const size_t dot = name.find_last_of('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

bool makeDirectory(const std::string& path, std::string* error) {
    NSError* e = nil;
    if ([[NSFileManager defaultManager] createDirectoryAtPath:ns(path) withIntermediateDirectories:YES
                                                   attributes:nil error:&e])
        return true;
    *error = "cannot create " + path + ": " + e.localizedDescription.UTF8String;
    return false;
}

std::string sha256File(const std::string& path) {
    NSData* data = [NSData dataWithContentsOfFile:ns(path)];
    if (!data) return "";
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, CC_LONG(data.length), digest);
    std::string hex;
    char buf[3];
    for (unsigned char b : digest) {
        std::snprintf(buf, sizeof(buf), "%02x", b);
        hex += buf;
    }
    return hex;
}

std::string fmt(double v, const char* format = "%.4g") {
    if (!std::isfinite(v)) return std::isinf(v) ? "inf" : "n/a";
    char buf[64];
    std::snprintf(buf, sizeof(buf), format, v);
    return buf;
}

// Keys of an experiment JSON object, all required.
bool checkKeys(NSDictionary* d, const std::set<std::string>& keys, const std::string& where, std::string* error) {
    if (![d isKindOfClass:[NSDictionary class]]) {
        *error = where + " must be an object";
        return false;
    }
    for (NSString* key in d) {
        if (!keys.count(key.UTF8String)) {
            *error = "unknown key " + where + "." + key.UTF8String;
            return false;
        }
    }
    for (const std::string& key : keys) {
        if (!d[ns(key)]) {
            *error = "missing key " + where + "." + key;
            return false;
        }
    }
    return true;
}

bool readStrings(id value, const std::string& where, std::vector<std::string>* out, std::string* error) {
    if (![value isKindOfClass:[NSArray class]] || [value count] == 0) {
        *error = where + " must be a non-empty array of strings";
        return false;
    }
    for (id item in (NSArray*)value) {
        if (![item isKindOfClass:[NSString class]]) {
            *error = where + " must contain strings";
            return false;
        }
        out->push_back([item UTF8String]);
    }
    return true;
}

bool readUInt(id value, const std::string& where, uint32_t* out, std::string* error) {
    if (![value isKindOfClass:[NSNumber class]] || [value doubleValue] < 1 ||
        [value doubleValue] != std::floor([value doubleValue])) {
        *error = where + " must be a positive integer";
        return false;
    }
    *out = [value unsignedIntValue];
    return true;
}

bool readPositive(id value, const std::string& where, double* out, std::string* error) {
    if (![value isKindOfClass:[NSNumber class]] || !([value doubleValue] > 0)) {
        *error = where + " must be a positive number";
        return false;
    }
    *out = [value doubleValue];
    return true;
}

struct MethodRun {
    std::string configPath;
    RenderConfig config;
    BenchResult result;
    Image image;
    uint32_t equalQualitySpp = 0;  // 0: not reached within equal_quality_max_spp
    double equalQualityGpuMs = NAN;
};

struct CaseResult {
    std::string scene;
    std::string camera;
    float exposure = 0.0f;
    ReferenceResult reference;
    std::string referenceDir;
    bool referenceAccepted = false;
    double minMethodRelMSE = NAN;
    std::vector<MethodRun> methods;
    std::vector<std::pair<uint32_t, double>> ptCurve;  // equal-quality pt: spp -> relMSE
};

bool sameReferenceSetup(const ReferenceResult& r, const Scene& scene, const CameraDesc& camera,
                        const RenderConfig& ptConfig) {
    return r.scene == scene.name && r.scene_version == scene.version && r.camera == camera.name &&
           r.width == scene.width && r.height == scene.height && r.seed_a == ptConfig.seed &&
           r.max_bounces == ptConfig.max_bounces && r.rr_start_bounce == ptConfig.rr_start_bounce &&
           r.renderer_version == CLOUDNR_VERSION;
}

// Generates (or reuses) a reference until est_ref_relMSE <= target or the spp cap is reached.
bool ensureReference(MetalContext& context, const Scene& scene, const DensityVolume& volume,
                     const CameraDesc& camera, const RenderConfig& ptConfig, const Experiment& ex,
                     const std::string& dir, double target, ReferenceResult* ref, bool* changed, std::string* error) {
    *changed = false;
    ReferenceResult existing;
    std::string ignored;
    bool have = loadReferenceResult(dir + "/ref.json", &existing, &ignored) &&
                sameReferenceSetup(existing, scene, camera, ptConfig);
    uint32_t spp = ex.reference_initial_spp;
    if (have) *ref = existing;
    while (!have || (ref->est_ref_relMSE > target && ref->spp_per_half < ex.reference_max_spp)) {
        if (have) spp = nextReferenceSpp(ref->spp_per_half, ref->est_ref_relMSE, target, ex.reference_max_spp);
        std::fprintf(stderr, "suite: reference %s/%s at %u spp per half\n", scene.name.c_str(), camera.name.c_str(),
                     spp);
        if (!generateReference(context, scene, volume, camera, ptConfig, spp, dir, ref, error)) return false;
        std::fprintf(stderr, "suite:   est_ref_relMSE %.3g (target %.3g)\n", ref->est_ref_relMSE, target);
        have = true;
        *changed = true;
    }
    return true;
}

std::string csvHeader() {
    return "scene,scene_version,camera,method,renderer,quality,gpu_ms_median,gpu_ms_p5,gpu_ms_p95,"
           "cpu_prep_ms_median,memory_renderer_bytes,memory_density_bytes,memory_device_delta_bytes,relMSE,psnr,"
           "ssim,equal_quality_pt_spp,equal_quality_pt_gpu_ms,reference_spp_per_half,reference_est_relMSE,"
           "reference_accepted,compile_ms,bake_ms,model_load_ms\n";
}

std::string qualityLabel(const RenderConfig& c, const BenchResult& r) {
    if (c.renderer == "pt") return std::to_string(r.quality_spp) + " spp";
    std::string q = "step " + fmt(c.step_scale, "%.3g") + "x";
    if (c.renderer == "analytic")
        q += ", " + std::to_string(c.octaves) + " oct" + (c.ambient ? ", ambient" : "");
    else if (c.renderer == "neural")
        q += ", " + c.inference;
    return q;
}

std::string csvRow(const CaseResult& c, uint32_t sceneVersion, const MethodRun& m) {
    const BenchResult& r = m.result;
    auto num = [](double v) { return std::isfinite(v) ? fmt(v, "%.9g") : std::string(); };
    return c.scene + "," + std::to_string(sceneVersion) + "," + c.camera + "," + stem(m.configPath) + "," +
           m.config.renderer + ",\"" + qualityLabel(m.config, r) + "\"," + num(r.gpu_ms.headline.median) + "," +
           num(r.gpu_ms.headline.p5) + "," + num(r.gpu_ms.headline.p95) + "," + num(r.cpu_prep_ms.headline.median) +
           "," + std::to_string(r.memory_renderer) + "," + std::to_string(r.memory_density) + "," +
           std::to_string(r.memory_device_delta) + "," + num(r.relMSE) + "," + num(r.psnr) + "," + num(r.ssim) + "," +
           (m.config.renderer == "pt" ? "" : (m.equalQualitySpp ? std::to_string(m.equalQualitySpp) : ">max")) + "," +
           num(m.equalQualityGpuMs) + "," + std::to_string(c.reference.spp_per_half) + "," +
           num(c.reference.est_ref_relMSE) + "," + (c.referenceAccepted ? "true" : "false") + "," +
           num(r.compile_ms) + "," + num(r.bake_ms) + "," + num(r.model_load_ms) + "\n";
}

}  // namespace

uint32_t nextReferenceSpp(uint32_t spp, double estimate, double target, uint32_t maxSpp) {
    double wanted = 2.0 * spp;
    if (target > 0 && std::isfinite(estimate)) wanted = std::max(wanted, spp * estimate / target);
    uint64_t next = 1;
    while (next < wanted && next < maxSpp) next <<= 1;
    return uint32_t(std::min<uint64_t>(next, maxSpp));
}

bool parseExperiment(const std::string& path, Experiment* ex, std::string* error) {
    NSData* data = [NSData dataWithContentsOfFile:ns(path)];
    NSDictionary* d = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    if (!d) {
        *error = "cannot read experiment " + path;
        return false;
    }
    if (!checkKeys(d, {"name", "scenes", "cameras", "methods", "reference", "timing", "equal_quality"}, "experiment",
                   error))
        return false;
    if (![d[@"name"] isKindOfClass:[NSString class]]) {
        *error = "experiment.name must be a string";
        return false;
    }
    Experiment e;
    e.name = [d[@"name"] UTF8String];
    if (!readStrings(d[@"scenes"], "experiment.scenes", &e.scenes, error) ||
        !readStrings(d[@"cameras"], "experiment.cameras", &e.cameras, error) ||
        !readStrings(d[@"methods"], "experiment.methods", &e.methods, error))
        return false;
    NSDictionary* ref = d[@"reference"];
    NSDictionary* timing = d[@"timing"];
    NSDictionary* eq = d[@"equal_quality"];
    if (!checkKeys(ref, {"config", "initial_spp", "max_spp", "abs_gate", "rel_gate"}, "reference", error) ||
        !checkKeys(timing, {"warmup", "frames", "runs"}, "timing", error) ||
        !checkKeys(eq, {"seed", "max_spp"}, "equal_quality", error))
        return false;
    if (![ref[@"config"] isKindOfClass:[NSString class]]) {
        *error = "reference.config must be a string";
        return false;
    }
    e.reference_config = [ref[@"config"] UTF8String];
    if (![timing[@"warmup"] isKindOfClass:[NSNumber class]] || [timing[@"warmup"] intValue] < 0) {
        *error = "timing.warmup must be a non-negative integer";
        return false;
    }
    e.warmup = [timing[@"warmup"] unsignedIntValue];
    if (!readUInt(ref[@"initial_spp"], "reference.initial_spp", &e.reference_initial_spp, error) ||
        !readUInt(ref[@"max_spp"], "reference.max_spp", &e.reference_max_spp, error) ||
        !readPositive(ref[@"abs_gate"], "reference.abs_gate", &e.abs_gate, error) ||
        !readPositive(ref[@"rel_gate"], "reference.rel_gate", &e.rel_gate, error) ||
        !readUInt(timing[@"frames"], "timing.frames", &e.frames, error) ||
        !readUInt(timing[@"runs"], "timing.runs", &e.runs, error) ||
        !readUInt(eq[@"seed"], "equal_quality.seed", &e.equal_quality_seed, error) ||
        !readUInt(eq[@"max_spp"], "equal_quality.max_spp", &e.equal_quality_max_spp, error))
        return false;
    *ex = e;
    return true;
}

int cmdSuite(int argc, char** argv) {
    std::string experimentPath, outDir, refsDir = "results/refs";
    bool quick = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--quick") {
            quick = true;
        } else if ((arg == "--experiment" || arg == "--out" || arg == "--refs") && i + 1 < argc) {
            (arg == "--experiment" ? experimentPath : arg == "--out" ? outDir : refsDir) = argv[++i];
        } else {
            return fail("unknown or incomplete argument '" + arg + "'");
        }
    }
    if (experimentPath.empty())
        return fail("usage: cloudnr suite --experiment E.json [--out DIR] [--refs DIR] [--quick]");

    std::string error;
    Experiment ex;
    if (!parseExperiment(experimentPath, &ex, &error)) return fail(error);
    if (quick) {
        // Smoke-test settings; the report is labelled as not protocol-compliant.
        ex.warmup = 3;
        ex.frames = 20;
        ex.runs = 1;
        ex.reference_initial_spp = 64;
        ex.reference_max_spp = 256;
        ex.equal_quality_max_spp = 256;
        refsDir += "_quick";
    }
    char stamp[32];
    const std::time_t now = std::time(nullptr);
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", std::localtime(&now));
    if (outDir.empty()) outDir = "results/" + ex.name + (quick ? "_quick_" : "_") + stamp;
    if (!makeDirectory(outDir + "/records", &error) || !makeDirectory(outDir + "/images", &error) ||
        !makeDirectory(refsDir, &error))
        return fail(error);

    auto ptConfig = loadRenderConfig(ex.reference_config, &error);
    if (!ptConfig) return fail(error);
    if (ptConfig->renderer != "pt") return fail("reference.config must be a pt config");
    std::vector<RenderConfig> configs;
    for (const std::string& path : ex.methods) {
        auto c = loadRenderConfig(path, &error);
        if (!c) return fail(error);
        configs.push_back(*c);
    }

    auto context = MetalContext::create(&error);
    if (!context) return fail(error);
    const bool validation = context->validationEnabled();
    if (validation) std::fprintf(stderr, "cloudnr suite: WARNING: validation enabled; timings are not protocol-valid\n");

    BenchOptions options;
    options.warmup = ex.warmup;
    options.frames = ex.frames;
    options.runs = ex.runs;

    std::vector<CaseResult> cases;
    std::map<std::string, uint32_t> sceneVersions;
    std::map<std::string, double> bakeMs;
    for (const std::string& scenePath : ex.scenes) {
        auto scene = loadScene(scenePath, &error);
        if (!scene) return fail(error);
        sceneVersions[scene->name] = scene->version;
        auto volume = DensityVolume::bake(*context, *scene, &error);
        if (!volume) return fail(error);
        bakeMs[scene->name] = volume->bakeGpuMs();

        for (const std::string& cameraName : ex.cameras) {
            const CameraDesc* camera = scene->camera(cameraName);
            if (!camera) return fail("camera '" + cameraName + "' not in " + scenePath);
            CaseResult c;
            c.scene = scene->name;
            c.camera = camera->name;
            c.exposure = camera->exposure_ev;
            c.referenceDir = refsDir + "/" + c.scene + "_" + c.camera;
            const std::string prefix = c.scene + "_" + c.camera;

            bool changed = false;
            if (!ensureReference(*context, *scene, *volume, *camera, *ptConfig, ex, c.referenceDir, ex.abs_gate,
                                 &c.reference, &changed, &error))
                return fail(error);

            for (size_t m = 0; m < configs.size(); ++m) {
                MethodRun run;
                run.configPath = ex.methods[m];
                run.config = configs[m];
                std::fprintf(stderr, "suite: bench %s %s\n", prefix.c_str(), run.configPath.c_str());
                if (!runBenchmark(*context, *scene, *volume, *camera, run.configPath, run.config, options, &run.result,
                                  &run.image, &error))
                    return fail(prefix + " " + run.configPath + ": " + error);
                c.methods.push_back(std::move(run));
            }

            // Metrics, then the relative gate: the reference must be 10x better than every method.
            Image reference;
            for (;;) {
                if (!readPFM(c.referenceDir + "/ref.pfm", &reference, &error)) return fail(error);
                c.minMethodRelMSE = INFINITY;
                for (MethodRun& run : c.methods) {
                    attachMetrics(&run.result, run.image, reference, c.exposure, c.referenceDir + "/ref.json");
                    if (std::isfinite(run.result.relMSE))
                        c.minMethodRelMSE = std::min(c.minMethodRelMSE, run.result.relMSE);
                }
                const double target = std::min(ex.abs_gate, ex.rel_gate * c.minMethodRelMSE);
                if (c.reference.est_ref_relMSE <= target || c.reference.spp_per_half >= ex.reference_max_spp) break;
                if (!ensureReference(*context, *scene, *volume, *camera, *ptConfig, ex, c.referenceDir, target,
                                     &c.reference, &changed, &error))
                    return fail(error);
            }
            c.referenceAccepted = c.reference.est_ref_relMSE <= ex.abs_gate &&
                                  c.reference.est_ref_relMSE <= ex.rel_gate * c.minMethodRelMSE;
            if (!c.referenceAccepted)
                std::fprintf(stderr, "suite: WARNING: reference %s NOT ACCEPTED (est %.3g, min method relMSE %.3g)\n",
                             prefix.c_str(), c.reference.est_ref_relMSE, c.minMethodRelMSE);

            // Equal-quality spp of a progressive pt with an independent seed.
            RenderConfig eqConfig = *ptConfig;
            eqConfig.seed = ex.equal_quality_seed;
            ProgressiveResult progressive;
            std::vector<double> curveGpu;
            if (!renderProgressive(
                    *context, *scene, eqConfig, *volume, makeSceneGPU(*scene, *camera), ex.equal_quality_max_spp,
                    [&](uint32_t n, const Image& image, double gpuMs) {
                        c.ptCurve.emplace_back(n, relMSE(image, reference, c.exposure));
                        curveGpu.push_back(gpuMs);
                    },
                    &progressive, &error))
                return fail(error);
            for (MethodRun& run : c.methods) {
                if (run.config.renderer == "pt") continue;
                for (size_t i = 0; i < c.ptCurve.size(); ++i) {
                    if (c.ptCurve[i].second <= run.result.relMSE) {
                        run.equalQualitySpp = c.ptCurve[i].first;
                        run.equalQualityGpuMs = curveGpu[i];
                        break;
                    }
                }
            }

            // Records, images and heat maps.
            const float absMax = [&] {
                float peak = 0.0f;
                for (float v : reference.rgb) peak = std::max(peak, v * std::exp2(c.exposure));
                return std::max(0.1f * peak, 1e-6f);
            }();
            for (MethodRun& run : c.methods) {
                const std::string name = prefix + "_" + stem(run.configPath);
                NSMutableDictionary* record = [benchRecordDictionary(run.result) mutableCopy];
                record[@"reference_accepted"] = @(c.referenceAccepted);
                record[@"reference_spp_per_half"] = @(c.reference.spp_per_half);
                record[@"reference_est_relMSE"] = @(c.reference.est_ref_relMSE);
                if (run.config.renderer != "pt")
                    record[@"equal_quality_pt_spp"] = run.equalQualitySpp ? (id)@(run.equalQualitySpp) : (id)@">max";
                if (!writeTextFile(outDir + "/records/" + name + "_" + stamp + ".json", toJsonString(record), &error) ||
                    !writePFM(run.image, outDir + "/images/" + name + ".pfm", &error) ||
                    !writePNG(run.image, c.exposure, outDir + "/images/" + name + ".png", &error) ||
                    !writeDisplayPNG(absErrorHeatmap(run.image, reference, c.exposure, absMax),
                                     outDir + "/images/" + name + "_abs.png", &error) ||
                    !writeDisplayPNG(relErrorHeatmap(run.image, reference, c.exposure),
                                     outDir + "/images/" + name + "_rel.png", &error))
                    return fail(error);
            }
            if (!writePNG(reference, c.exposure, outDir + "/images/" + prefix + "_reference.png", &error))
                return fail(error);
            cases.push_back(std::move(c));
        }
    }
    if (context->gpuErrorCount() > 0) return fail("GPU errors during the suite");

    // summary.csv
    std::string csv = csvHeader();
    for (const CaseResult& c : cases)
        for (const MethodRun& m : c.methods) csv += csvRow(c, sceneVersions[c.scene], m);
    if (!writeTextFile(outDir + "/summary.csv", csv, &error)) return fail(error);

    // report.md
    const BenchResult& any = cases.front().methods.front().result;
    std::string md = "# Cloud renderer comparison: " + ex.name + (quick ? " (QUICK - not protocol-compliant)" : "") +
                     "\n\n";
    md += "| Item | Value |\n|---|---|\n";
    md += "| GPU | " + any.gpu_name + " |\n| OS | " + any.os_version + " |\n";
    md += std::string("| Renderer version | ") + CLOUDNR_VERSION + " |\n";
    md += "| Experiment | `" + experimentPath + "` sha256 `" + sha256File(experimentPath) + "` |\n";
    md += "| Timing | " + std::to_string(ex.warmup) + " warm-up + " + std::to_string(ex.frames) + " frames x " +
          std::to_string(ex.runs) + " runs; GPU time from MTL4CommitFeedback; headline = median of run medians; "
          "p5/p95 nearest-rank |\n";
    md += std::string("| Validation layer | ") + (validation ? "ENABLED (timings invalid)" : "off") + " |\n";
    md += "| Reference gates | est. relMSE <= " + fmt(ex.abs_gate) + " and <= " + fmt(ex.rel_gate) +
          " x min method relMSE; spp per half " + std::to_string(ex.reference_initial_spp) + ".." +
          std::to_string(ex.reference_max_spp) + " |\n\n";
    md += "Metrics are against the path-traced reference (relMSE on linear radiance; PSNR and SSIM on display-mapped "
          "images). Methods run at their recorded configurations, which differ in quality settings (column "
          "*Quality*). *Equal-quality pt spp* is the first power-of-two spp at which a progressive path tracer "
          "(seed " + std::to_string(ex.equal_quality_seed) + ") reaches the method's relMSE; its GPU time is the "
          "cumulative GPU time of that run.\n\n";
    for (const CaseResult& c : cases) {
        md += "## " + c.scene + " / " + c.camera + "\n\n";
        md += "Reference: " + std::to_string(c.reference.spp_per_half) + " spp per half, est. relMSE " +
              fmt(c.reference.est_ref_relMSE) + ", " + (c.referenceAccepted ? "accepted" : "**NOT ACCEPTED**") +
              ", bounce-cap hits " + fmt(c.reference.bounce_cap_hits) + ", mean bounces " +
              fmt(c.reference.mean_bounces) + ". ![reference](images/" + c.scene + "_" + c.camera +
              "_reference.png)\n\n";
        md += "| Method | Quality | GPU ms median [p5, p95] | CPU prep ms | relMSE | PSNR dB | SSIM | Renderer MB | "
              "Equal-quality pt spp (GPU ms) |\n|---|---|---|---|---|---|---|---|---|\n";
        for (const MethodRun& m : c.methods) {
            const BenchResult& r = m.result;
            std::string eq = "-";
            if (m.config.renderer != "pt")
                eq = m.equalQualitySpp ? std::to_string(m.equalQualitySpp) + " (" + fmt(m.equalQualityGpuMs) + ")"
                                       : "> " + std::to_string(ex.equal_quality_max_spp);
            md += "| " + stem(m.configPath) + " | " + qualityLabel(m.config, r) + " | " + fmt(r.gpu_ms.headline.median) +
                  " [" + fmt(r.gpu_ms.headline.p5) + ", " + fmt(r.gpu_ms.headline.p95) + "] | " +
                  fmt(r.cpu_prep_ms.headline.median) + " | " + fmt(r.relMSE) + " | " + fmt(r.psnr) + " | " +
                  fmt(r.ssim) + " | " + fmt(r.memory_renderer / 1048576.0) + " | " + eq + " |\n";
        }
        md += "\n";
        for (const MethodRun& m : c.methods) {
            const std::string name = c.scene + "_" + c.camera + "_" + stem(m.configPath);
            md += "- " + stem(m.configPath) + ": [image](images/" + name + ".png), [abs error](images/" + name +
                  "_abs.png), [rel error](images/" + name + "_rel.png)\n";
        }
        md += "\n";
    }
    md += "## Excluded costs (not part of steady-state timing)\n\n| Scene | Camera | Method | Compile ms | Bake ms | "
          "Model load ms | Reference GPU ms | Memory density MB | Device delta MB |\n|---|---|---|---|---|---|---|---|---|\n";
    for (const CaseResult& c : cases)
        for (const MethodRun& m : c.methods) {
            const BenchResult& r = m.result;
            md += "| " + c.scene + " | " + c.camera + " | " + stem(m.configPath) + " | " + fmt(r.compile_ms) + " | " +
                  fmt(r.bake_ms) + " | " + fmt(r.model_load_ms) + " | " + fmt(c.reference.gpu_ms_total) + " | " +
                  fmt(r.memory_density / 1048576.0) + " | " + fmt(r.memory_device_delta / 1048576.0) + " |\n";
        }
    md += "\nCompile times are wall time of MTL4Compiler calls and include Metal's shader cache hits.\n\n";
    md += "## Bias sources\n\n- pt: bounce cap " + std::to_string(4096) +
          " (hits listed per reference); low-spp pt methods carry Monte Carlo noise, not bias.\n"
          "- analytic: ray-march discretization, two-direction ambient term, octave multiple-scattering "
          "approximation (not energy-conserving).\n"
          "- neural: ray-march discretization, network approximation of the in-scattered radiance, Monte Carlo "
          "noise in training targets; shader_ml uses fp16 activations and weights.\n";
    for (const CaseResult& c : cases) {
        for (const MethodRun& m : c.methods) {
            if (!m.result.has_model) continue;
            const std::string sidecar = m.result.model_path.substr(0, m.result.model_path.size() - 5) + ".json";
            NSData* data = [NSData dataWithContentsOfFile:ns(sidecar)];
            NSDictionary* side = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
            md += "\n## Neural model\n\n- `" + m.result.model_path + "` sha256 `" + m.result.model_sha256 +
                  "`, format " + std::to_string(m.result.model_format_version) + "\n- training shape seeds " +
                  std::to_string(m.result.model_train_seed_first) + "-" +
                  std::to_string(m.result.model_train_seed_last) + "\n";
            if ([side[@"best_val_loss"] isKindOfClass:[NSNumber class]])
                md += "- best validation loss (MSE on raw targets) " + fmt([side[@"best_val_loss"] doubleValue]) +
                      ", validation seeds " + fmt([side[@"val_seed_range"][0] doubleValue], "%.0f") + "-" +
                      fmt([side[@"val_seed_range"][1] doubleValue], "%.0f") + "\n";
            break;
        }
        break;
    }
    if (!writeTextFile(outDir + "/report.md", md, &error)) return fail(error);
    NSError* copyError = nil;
    [[NSFileManager defaultManager] removeItemAtPath:ns(outDir + "/experiment.json") error:nil];
    if (![[NSFileManager defaultManager] copyItemAtPath:ns(experimentPath) toPath:ns(outDir + "/experiment.json")
                                                  error:&copyError])
        return fail("cannot copy experiment file");
    std::printf("%s\n", outDir.c_str());
    bool allAccepted = true;
    for (const CaseResult& c : cases) allAccepted = allAccepted && c.referenceAccepted;
    return allAccepted ? 0 : 2;
}
