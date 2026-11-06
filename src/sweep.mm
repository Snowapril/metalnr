#include "sweep.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "bench.h"
#include "density_volume.h"
#include "metal_context.h"
#include "reference.h"
#include "renderer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <set>

namespace {

NSString* ns(const std::string& s) { return [NSString stringWithUTF8String:s.c_str()]; }

int fail(const std::string& message) {
    std::fprintf(stderr, "cloudnr sweep: %s\n", message.c_str());
    return 1;
}

std::string stem(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    const size_t dot = name.find_last_of('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

std::string fmt(double v, const char* format = "%.4g") {
    if (!std::isfinite(v)) return std::isinf(v) ? "inf" : "n/a";
    char buf[64];
    std::snprintf(buf, sizeof(buf), format, v);
    return buf;
}

bool makeDirectory(const std::string& path, std::string* error) {
    NSError* e = nil;
    if ([[NSFileManager defaultManager] createDirectoryAtPath:ns(path) withIntermediateDirectories:YES
                                                   attributes:nil error:&e])
        return true;
    *error = "cannot create " + path + ": " + e.localizedDescription.UTF8String;
    return false;
}

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

bool readString(id value, const std::string& where, std::string* out, std::string* error) {
    if (![value isKindOfClass:[NSString class]] || [value length] == 0) {
        *error = where + " must be a non-empty string";
        return false;
    }
    *out = [value UTF8String];
    return true;
}

bool readUInt(id value, const std::string& where, bool allowZero, uint32_t* out, std::string* error) {
    const double v = [value isKindOfClass:[NSNumber class]] ? [value doubleValue] : NAN;
    if (!(v >= (allowZero ? 0.0 : 1.0)) || v > 4294967295.0 || v != std::floor(v)) {
        *error = where + (allowZero ? " must be a non-negative integer" : " must be a positive integer");
        return false;
    }
    *out = uint32_t(v);
    return true;
}

// The analytic output's alpha (1 - T) from the renderer's shared RGBA32Float texture.
bool readAlpha(const Renderer& renderer, std::vector<float>* alpha, std::string* error) {
    id<MTLTexture> texture = renderer.output();
    if (!texture || texture.pixelFormat != MTLPixelFormatRGBA32Float ||
        texture.storageMode != MTLStorageModeShared) {
        *error = "renderer output must be a shared RGBA32Float texture";
        return false;
    }
    const size_t width = texture.width, height = texture.height;
    std::vector<float> rgba(width * height * 4);
    [texture getBytes:rgba.data()
          bytesPerRow:width * 4 * sizeof(float)
           fromRegion:MTLRegionMake2D(0, 0, width, height)
          mipmapLevel:0];
    alpha->resize(width * height);
    for (size_t i = 0; i < width * height; ++i) (*alpha)[i] = rgba[i * 4 + 3];
    return true;
}

std::string sunText(simd_float3 sun) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%.9g %.9g %.9g\n", sun.x, sun.y, sun.z);
    return buf;
}

// An existing reference in dir is reused only if it was rendered for exactly this setup and sun.
bool reusableReference(const std::string& dir, const Scene& scene, const CameraDesc& camera,
                       const RenderConfig& pt, uint32_t sppPerHalf, ReferenceResult* out) {
    ReferenceResult r;
    std::string ignored;
    if (!loadReferenceResult(dir + "/ref.json", &r, &ignored)) return false;
    NSString* sun = [NSString stringWithContentsOfFile:ns(dir + "/sun.txt") encoding:NSUTF8StringEncoding error:nil];
    const bool same = sun && sunText(scene.light.sun_direction) == sun.UTF8String && r.scene == scene.name &&
                      r.scene_version == scene.version && r.camera == camera.name && r.width == scene.width &&
                      r.height == scene.height && r.spp_per_half == sppPerHalf && r.seed_a == pt.seed &&
                      r.max_bounces == pt.max_bounces && r.rr_start_bounce == pt.rr_start_bounce &&
                      r.renderer_version == CLOUDNR_VERSION;
    if (same) *out = r;
    return same;
}

double meanFinite(const std::vector<double>& values) {
    double sum = 0.0;
    size_t n = 0;
    for (double v : values)
        if (std::isfinite(v)) {
            sum += v;
            ++n;
        }
    return n ? sum / double(n) : NAN;
}

struct MethodResult {
    BenchResult bench;
    MaskedLuminance masked;
};

struct AngleResult {
    double angle = 0.0;
    simd_float3 sun{0, 0, 0};
    ReferenceResult reference;
    bool referenceReused = false;
    uint64_t maskPixels = 0;
    std::vector<MethodResult> methods;
};

}  // namespace

bool parseSweepExperiment(const std::string& path, SweepExperiment* ex, std::string* error) {
    NSData* data = [NSData dataWithContentsOfFile:ns(path)];
    NSDictionary* d = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    if (!d) {
        *error = "cannot read sweep experiment " + path;
        return false;
    }
    if (!checkKeys(d, {"name", "scene", "camera", "angles_deg", "elevation_deg", "width", "height", "methods",
                       "reference", "timing"},
                   "experiment", error))
        return false;
    SweepExperiment e;
    if (!readString(d[@"name"], "experiment.name", &e.name, error) ||
        !readString(d[@"scene"], "experiment.scene", &e.scene, error) ||
        !readString(d[@"camera"], "experiment.camera", &e.camera, error))
        return false;

    id angles = d[@"angles_deg"];
    if (![angles isKindOfClass:[NSArray class]] || [angles count] == 0) {
        *error = "experiment.angles_deg must be a non-empty array of numbers";
        return false;
    }
    for (id a in (NSArray*)angles) {
        if (![a isKindOfClass:[NSNumber class]] || !std::isfinite([a doubleValue])) {
            *error = "experiment.angles_deg must contain finite numbers";
            return false;
        }
        e.angles_deg.push_back([a doubleValue]);
    }
    id elevation = d[@"elevation_deg"];
    if (![elevation isKindOfClass:[NSNumber class]] || !([elevation doubleValue] > -90.0) ||
        !([elevation doubleValue] < 90.0)) {
        *error = "experiment.elevation_deg must be a number in (-90, 90)";
        return false;
    }
    e.elevation_deg = [elevation doubleValue];

    id methods = d[@"methods"];
    if (![methods isKindOfClass:[NSArray class]] || [methods count] == 0) {
        *error = "experiment.methods must be a non-empty array of strings";
        return false;
    }
    for (id m in (NSArray*)methods) {
        std::string s;
        if (!readString(m, "experiment.methods[]", &s, error)) return false;
        e.methods.push_back(s);
    }

    NSDictionary* ref = d[@"reference"];
    NSDictionary* timing = d[@"timing"];
    if (!checkKeys(ref, {"config", "spp_per_half"}, "reference", error) ||
        !checkKeys(timing, {"warmup", "frames", "runs"}, "timing", error))
        return false;
    if (!readUInt(d[@"width"], "experiment.width", false, &e.width, error) ||
        !readUInt(d[@"height"], "experiment.height", false, &e.height, error) ||
        !readString(ref[@"config"], "reference.config", &e.reference_config, error) ||
        !readUInt(ref[@"spp_per_half"], "reference.spp_per_half", false, &e.reference_spp_per_half, error) ||
        !readUInt(timing[@"warmup"], "timing.warmup", true, &e.warmup, error) ||
        !readUInt(timing[@"frames"], "timing.frames", false, &e.frames, error) ||
        !readUInt(timing[@"runs"], "timing.runs", false, &e.runs, error))
        return false;
    *ex = e;
    return true;
}

simd_float3 sweepSunDirection(const CameraDesc& camera, double angleDeg, double elevationDeg) {
    const simd_float3 up = simd_make_float3(0.0f, 1.0f, 0.0f);
    simd_float3 forward = camera.target - camera.position;
    forward.y = 0.0f;
    const simd_float3 fwdH = simd_length(forward) > 1e-6f ? simd_normalize(forward) : simd_make_float3(0, 0, 1);
    const simd_float3 rightH = simd_normalize(simd_cross(fwdH, up));
    const double a = angleDeg * M_PI / 180.0;
    const double e = elevationDeg * M_PI / 180.0;
    const simd_float3 horizontal = float(-std::cos(a)) * fwdH + float(std::sin(a)) * rightH;
    return simd_normalize(float(std::cos(e)) * horizontal + float(std::sin(e)) * up);
}

double luminance709(const float* rgb) { return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]; }

MaskedLuminance maskedLuminance(const Image& test, const Image& reference, const std::vector<uint8_t>& mask) {
    MaskedLuminance out;
    const size_t n = size_t(reference.width) * reference.height;
    if (test.width != reference.width || test.height != reference.height || mask.size() != n ||
        test.rgb.size() != n * 3 || reference.rgb.size() != n * 3) {
        out.score = out.rmse = NAN;
        return out;
    }
    double sumTest = 0.0, sumRef = 0.0, sumSq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!mask[i]) continue;
        const double lt = luminance709(&test.rgb[i * 3]);
        const double lr = luminance709(&reference.rgb[i * 3]);
        sumTest += lt;
        sumRef += lr;
        sumSq += (lt - lr) * (lt - lr);
        ++out.pixels;
    }
    if (out.pixels == 0) {
        out.score = out.rmse = NAN;
        return out;
    }
    out.score = sumRef > 0.0 ? sumTest / sumRef : NAN;
    out.rmse = std::sqrt(sumSq / double(out.pixels));
    return out;
}

int cmdSweep(int argc, char** argv) {
    std::string experimentPath, outDir;
    bool quick = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--quick") {
            quick = true;
        } else if ((arg == "--experiment" || arg == "--out") && i + 1 < argc) {
            (arg == "--experiment" ? experimentPath : outDir) = argv[++i];
        } else {
            return fail("unknown or incomplete argument '" + arg + "'");
        }
    }
    if (experimentPath.empty()) return fail("usage: cloudnr sweep --experiment E.json [--out DIR] [--quick]");

    std::string error;
    SweepExperiment ex;
    if (!parseSweepExperiment(experimentPath, &ex, &error)) return fail(error);
    if (quick) {
        // Smoke-test settings; the report is labelled as not protocol-compliant.
        ex.angles_deg = {0.0, 90.0, 180.0};
        ex.reference_spp_per_half = 128;
        ex.warmup = 2;
        ex.frames = 10;
        ex.runs = 1;
    }
    char stamp[32];
    const std::time_t now = std::time(nullptr);
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", std::localtime(&now));
    if (outDir.empty()) outDir = std::string("results/sweep_") + (quick ? "quick_" : "") + stamp;
    if (!makeDirectory(outDir + "/images", &error) || !makeDirectory(outDir + "/refs", &error) ||
        !makeDirectory(outDir + "/records", &error))
        return fail(error);

    auto loaded = loadScene(ex.scene, &error);
    if (!loaded) return fail(error);
    Scene base = *loaded;
    base.width = ex.width;
    base.height = ex.height;
    const CameraDesc* cameraPtr = base.camera(ex.camera);
    if (!cameraPtr) return fail("camera '" + ex.camera + "' not found in " + ex.scene);
    const CameraDesc camera = *cameraPtr;

    auto ptConfig = loadRenderConfig(ex.reference_config, &error);
    if (!ptConfig) return fail(error);
    if (ptConfig->renderer != "pt") return fail("reference.config must be a pt config");
    std::vector<RenderConfig> configs;
    std::vector<std::string> names;
    int maskIndex = -1;
    for (const std::string& path : ex.methods) {
        auto c = loadRenderConfig(path, &error);
        if (!c) return fail(error);
        if (c->renderer == "analytic" && maskIndex < 0) maskIndex = int(configs.size());
        configs.push_back(*c);
        names.push_back(stem(path));
    }
    if (maskIndex < 0) return fail("experiment.methods must include an analytic config (it defines the cloud mask)");

    auto context = MetalContext::create(&error);
    if (!context) return fail(error);
    const bool validation = context->validationEnabled();
    if (validation) std::fprintf(stderr, "cloudnr sweep: WARNING: validation enabled; timings are not protocol-valid\n");

    // The density does not depend on the sun: one bake for the whole sweep.
    auto volume = DensityVolume::bake(*context, base, &error);
    if (!volume) return fail(error);
    // Mask renderer: the analytic method's config, one frame per angle (sun direction is per-frame).
    auto maskRenderer = makeRenderer(*context, base, configs[maskIndex], *volume, &error);
    if (!maskRenderer) return fail(error);

    BenchOptions options;
    options.warmup = ex.warmup;
    options.frames = ex.frames;
    options.runs = ex.runs;
    const float exposure = camera.exposure_ev;
    const uint32_t width = base.width, height = base.height;

    std::vector<AngleResult> results;
    for (double angle : ex.angles_deg) {
        AngleResult ar;
        ar.angle = angle;
        ar.sun = sweepSunDirection(camera, angle, ex.elevation_deg);
        Scene scene = base;
        scene.light.sun_direction = ar.sun;
        const std::string tag = "a" + fmt(angle, "%g");
        std::fprintf(stderr, "sweep: angle %g deg, sun (%.4f, %.4f, %.4f)\n", angle, ar.sun.x, ar.sun.y, ar.sun.z);

        // Reference.
        const std::string refDir = outDir + "/refs/" + tag;
        ar.referenceReused = reusableReference(refDir, scene, camera, *ptConfig, ex.reference_spp_per_half,
                                               &ar.reference);
        if (!ar.referenceReused) {
            std::fprintf(stderr, "sweep:   reference %u spp per half\n", ex.reference_spp_per_half);
            if (!generateReference(*context, scene, *volume, camera, *ptConfig, ex.reference_spp_per_half, refDir,
                                   &ar.reference, &error) ||
                !writeTextFile(refDir + "/sun.txt", sunText(ar.sun), &error))
                return fail(tag + " reference: " + error);
        }
        Image reference;
        if (!readPFM(refDir + "/ref.pfm", &reference, &error)) return fail(error);

        // Mask: analytic alpha > 0.05.
        const SceneGPU sceneGPU = makeSceneGPU(scene, camera);
        maskRenderer->resetAccumulation();
        if (!context->runSync(
                [&](id<MTL4CommandBuffer> commandBuffer) {
                    id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                    maskRenderer->encode(encoder, FrameInputs{sceneGPU, context->currentSlot()});
                    [encoder endEncoding];
                },
                nullptr, &error))
            return fail(tag + " mask: " + error);
        std::vector<float> alpha;
        if (!readAlpha(*maskRenderer, &alpha, &error)) return fail(error);
        std::vector<uint8_t> mask(alpha.size());
        for (size_t i = 0; i < alpha.size(); ++i) mask[i] = alpha[i] > 0.05f ? 1 : 0;

        // Methods.
        Image triptych;
        triptych.width = width * uint32_t(configs.size() + 1);
        triptych.height = height;
        triptych.rgb.assign(size_t(triptych.width) * height * 3, 0.0f);
        auto place = [&](const Image& image, uint32_t column) {
            for (uint32_t y = 0; y < height; ++y)
                std::copy(image.pixel(0, y), image.pixel(0, y) + size_t(width) * 3,
                          triptych.pixel(column * width, y));
        };
        place(reference, 0);
        for (size_t m = 0; m < configs.size(); ++m) {
            std::fprintf(stderr, "sweep:   bench %s\n", ex.methods[m].c_str());
            MethodResult mr;
            Image image;
            if (!runBenchmark(*context, scene, *volume, camera, ex.methods[m], configs[m], options, &mr.bench, &image,
                              &error))
                return fail(tag + " " + ex.methods[m] + ": " + error);
            if (image.width != width || image.height != height) return fail("method image size mismatch");
            attachMetrics(&mr.bench, image, reference, exposure, refDir + "/ref.json");
            mr.masked = maskedLuminance(image, reference, mask);
            ar.maskPixels = mr.masked.pixels;
            place(image, uint32_t(m + 1));

            NSMutableDictionary* record = [benchRecordDictionary(mr.bench) mutableCopy];
            record[@"sweep"] = @{@"angle_deg" : @(angle), @"elevation_deg" : @(ex.elevation_deg),
                                 @"sun_direction" : @[ @(ar.sun.x), @(ar.sun.y), @(ar.sun.z) ],
                                 @"score" : std::isfinite(mr.masked.score) ? (id)@(mr.masked.score) : (id)[NSNull null],
                                 @"rmse" : std::isfinite(mr.masked.rmse) ? (id)@(mr.masked.rmse) : (id)[NSNull null],
                                 @"mask_pixels" : @(mr.masked.pixels),
                                 @"reference_spp_per_half" : @(ar.reference.spp_per_half),
                                 @"reference_est_relMSE" : @(ar.reference.est_ref_relMSE)};
            if (!writeTextFile(outDir + "/records/" + tag + "_" + names[m] + ".json", toJsonString(record), &error))
                return fail(error);
            ar.methods.push_back(std::move(mr));
        }
        if (!writePNG(triptych, exposure, outDir + "/images/" + tag + ".png", &error)) return fail(error);
        results.push_back(std::move(ar));
    }
    if (context->gpuErrorCount() > 0) return fail("GPU errors during the sweep");

    // sweep.csv
    auto num = [](double v) { return std::isfinite(v) ? fmt(v, "%.9g") : std::string(); };
    std::string csv = "angle,method,score,rmse,relMSE,psnr,ssim,gpu_ms,ref_est_relMSE,mask_pixels\n";
    for (const AngleResult& ar : results)
        for (size_t m = 0; m < ar.methods.size(); ++m) {
            const MethodResult& r = ar.methods[m];
            csv += fmt(ar.angle, "%g") + "," + names[m] + "," + num(r.masked.score) + "," + num(r.masked.rmse) + "," +
                   num(r.bench.relMSE) + "," + num(r.bench.psnr) + "," + num(r.bench.ssim) + "," +
                   num(r.bench.gpu_ms.headline.median) + "," + num(ar.reference.est_ref_relMSE) + "," +
                   std::to_string(r.masked.pixels) + "\n";
        }
    if (!writeTextFile(outDir + "/sweep.csv", csv, &error)) return fail(error);

    // report.md
    const BenchResult& any = results.front().methods.front().bench;
    std::string header = "| Angle (deg) |", rule = "|---|";
    for (const std::string& n : names) {
        header += " " + n + " |";
        rule += "---|";
    }
    header += "\n" + rule + "\n";
    auto table = [&](const std::string& title, const std::string& note, auto value, const char* format,
                     const std::string& summaryLabel, auto summary) {
        std::string t = "## " + title + "\n\n" + note + "\n\n" + header;
        std::vector<std::vector<double>> columns(names.size());
        for (const AngleResult& ar : results) {
            t += "| " + fmt(ar.angle, "%g") + " |";
            for (size_t m = 0; m < ar.methods.size(); ++m) {
                const double v = value(ar.methods[m]);
                columns[m].push_back(v);
                t += " " + fmt(v, format) + " |";
            }
            t += "\n";
        }
        t += "| " + summaryLabel + " |";
        for (const auto& column : columns) t += " " + fmt(summary(column), format) + " |";
        return t + "\n\n";
    };
    auto median = [](const std::vector<double>& v) { return summarizeTimes(v).median; };

    std::string md = "# Sun sweep: " + ex.name + (quick ? " (QUICK - not protocol-compliant)" : "") + "\n\n";
    md += "| Item | Value |\n|---|---|\n";
    md += "| Scene | `" + ex.scene + "` (" + base.name + ", version " + std::to_string(base.version) + ") |\n";
    md += "| Camera | " + camera.name + " |\n";
    md += "| Resolution | " + std::to_string(width) + " x " + std::to_string(height) + " |\n";
    md += "| Sun | azimuth around the cloud relative to the camera (0 = behind the camera, 180 = behind the cloud), "
          "elevation " + fmt(ex.elevation_deg, "%g") + " deg |\n";
    md += "| GPU | " + any.gpu_name + " |\n| OS | " + any.os_version + " |\n";
    md += std::string("| Renderer version | ") + CLOUDNR_VERSION + " |\n";
    md += "| Reference | `" + ex.reference_config + "`, " + std::to_string(ex.reference_spp_per_half) +
          " spp per half, seeds " + std::to_string(ptConfig->seed) + "/" + std::to_string(ptConfig->seed + 1) + " |\n";
    md += "| Timing | " + std::to_string(ex.warmup) + " warm-up + " + std::to_string(ex.frames) + " frames x " +
          std::to_string(ex.runs) + " runs; GPU time from MTL4CommitFeedback; median of run medians |\n";
    md += std::string("| Validation layer | ") + (validation ? "ENABLED (timings invalid)" : "off") + " |\n\n";
    md += "Mask: pixels where the `" + names[maskIndex] + "` output alpha (1 - transmittance) exceeds 0.05. "
          "Score = sum of Rec.709 luminance of the method over the mask / the same sum for the path-traced "
          "reference (1 = matches, < 1 too dark). RMSE is the luminance RMSE over the mask (linear radiance). "
          "relMSE, PSNR and SSIM are full-image metrics against the reference with the camera exposure.\n\n";

    md += table("Score", "Sum of luminance over the mask relative to the reference.",
                [](const MethodResult& r) { return r.masked.score; }, "%.3f", "Mean", meanFinite);
    md += table("Luminance RMSE", "Over the mask, linear radiance.",
                [](const MethodResult& r) { return r.masked.rmse; }, "%.4g", "Mean", meanFinite);
    md += table("GPU time (ms)", "Steady-state GPU ms per frame, median of run medians.",
                [](const MethodResult& r) { return r.bench.gpu_ms.headline.median; }, "%.3f", "Median", median);
    md += table("relMSE (full image)", "Linear radiance, exposure applied.",
                [](const MethodResult& r) { return r.bench.relMSE; }, "%.4g", "Mean", meanFinite);
    md += table("PSNR dB (full image)", "Display-mapped.",
                [](const MethodResult& r) { return r.bench.psnr; }, "%.2f", "Mean", meanFinite);
    md += table("SSIM (full image)", "Display-mapped.",
                [](const MethodResult& r) { return r.bench.ssim; }, "%.4f", "Mean", meanFinite);

    md += "## References and images\n\nEstimated reference error is relMSE(A, B) / 4 of the two independent "
          "halves.\n\n| Angle (deg) | Sun direction | spp per half | Est. relMSE | Bounce-cap hits | Mask pixels | "
          "Reused | Images (reference";
    for (const std::string& n : names) md += " \\| " + n;
    md += ") |\n|---|---|---|---|---|---|---|---|\n";
    for (const AngleResult& ar : results) {
        const std::string tag = "a" + fmt(ar.angle, "%g");
        md += "| " + fmt(ar.angle, "%g") + " | (" + fmt(ar.sun.x, "%.3f") + ", " + fmt(ar.sun.y, "%.3f") + ", " +
              fmt(ar.sun.z, "%.3f") + ") | " + std::to_string(ar.reference.spp_per_half) + " | " +
              fmt(ar.reference.est_ref_relMSE) + " | " + fmt(ar.reference.bounce_cap_hits) + " | " +
              std::to_string(ar.maskPixels) + " | " + (ar.referenceReused ? "yes" : "no") + " | ![" + tag +
              "](images/" + tag + ".png) |\n";
    }
    md += "\n## Bias sources\n\n- reference: Monte Carlo noise (estimated above) and the path tracer bounce cap.\n"
          "- analytic: ray-march discretization, octave multiple-scattering approximation, two-direction ambient "
          "term.\n- neural: ray-march discretization, network approximation of the in-scattered radiance, Monte "
          "Carlo noise in training targets; shader_ml uses fp16 weights with fp32 activations and accumulation.\n";
    if (!writeTextFile(outDir + "/report.md", md, &error)) return fail(error);

    [[NSFileManager defaultManager] removeItemAtPath:ns(outDir + "/experiment.json") error:nil];
    if (![[NSFileManager defaultManager] copyItemAtPath:ns(experimentPath) toPath:ns(outDir + "/experiment.json")
                                                  error:nil])
        return fail("cannot copy experiment file");
    std::printf("%s\n", outDir.c_str());
    return 0;
}
