#include "scene.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <random>
#include <set>

namespace {

struct ParseError {
    std::string message;
};

[[noreturn]] void fail(const std::string& message) { throw ParseError{message}; }

std::string joinPath(const std::string& path, const std::string& key) {
    return path.empty() ? key : path + "." + key;
}

bool isBool(id v) {
    return [v isKindOfClass:[NSNumber class]] && CFGetTypeID((__bridge CFTypeRef)v) == CFBooleanGetTypeID();
}

bool isNumber(id v) { return [v isKindOfClass:[NSNumber class]] && !isBool(v); }

NSDictionary* asDict(id v, const std::string& path) {
    if (![v isKindOfClass:[NSDictionary class]]) fail(path.empty() ? "root must be an object" : path + " must be an object");
    return v;
}

// Every key of `d` must be in `allowed`.
void checkKeys(NSDictionary* d, const std::string& path, std::initializer_list<const char*> allowed) {
    for (id key in d) {
        if (![key isKindOfClass:[NSString class]]) fail(path + ": non-string key");
        std::string k = [key UTF8String];
        bool found = false;
        for (const char* a : allowed) found = found || k == a;
        if (!found) fail("unknown key " + joinPath(path, k));
    }
}

id member(NSDictionary* d, const std::string& path, const char* key, bool required) {
    id v = d[@(key)];
    if (!v && required) fail("missing key " + joinPath(path, key));
    return v;
}

double toNumber(id v, const std::string& path) {
    if (!isNumber(v)) fail(path + " must be a number");
    double x = [v doubleValue];
    if (!std::isfinite(x)) fail(path + " must be finite");
    return x;
}

float toFloat(id v, const std::string& path) {
    double x = toNumber(v, path);
    if (std::fabs(x) > 3.0e38) fail(path + " is out of float range");
    return float(x);
}

uint32_t toUint(id v, const std::string& path) {
    double x = toNumber(v, path);
    if (x < 0 || x > 4294967295.0 || x != std::floor(x)) fail(path + " must be an unsigned 32-bit integer");
    return uint32_t(x);
}

NSArray* toArray3(id v, const std::string& path) {
    if (![v isKindOfClass:[NSArray class]] || [v count] != 3) fail(path + " must be an array of 3 numbers");
    return v;
}

simd_float3 toFloat3(id v, const std::string& path) {
    NSArray* a = toArray3(v, path);
    simd_float3 r;
    for (int i = 0; i < 3; ++i) r[i] = toFloat(a[i], path + "[" + std::to_string(i) + "]");
    return r;
}

simd_uint3 toUint3(id v, const std::string& path) {
    NSArray* a = toArray3(v, path);
    simd_uint3 r;
    for (int i = 0; i < 3; ++i) r[i] = toUint(a[i], path + "[" + std::to_string(i) + "]");
    return r;
}

std::string toString(id v, const std::string& path) {
    if (![v isKindOfClass:[NSString class]]) fail(path + " must be a string");
    return [v UTF8String];
}

bool toBool(id v, const std::string& path) {
    if (!isBool(v)) fail(path + " must be a boolean");
    return [v boolValue];
}

// Field readers: required keys throw when absent; optional keys leave `out` unchanged.
void readFloat(NSDictionary* d, const std::string& path, const char* key, float& out, bool required = true) {
    if (id v = member(d, path, key, required)) out = toFloat(v, joinPath(path, key));
}
void readUint(NSDictionary* d, const std::string& path, const char* key, uint32_t& out, bool required = true) {
    if (id v = member(d, path, key, required)) out = toUint(v, joinPath(path, key));
}
void readFloat3(NSDictionary* d, const std::string& path, const char* key, simd_float3& out) {
    out = toFloat3(member(d, path, key, true), joinPath(path, key));
}
void readString(NSDictionary* d, const std::string& path, const char* key, std::string& out, bool required = true) {
    if (id v = member(d, path, key, required)) out = toString(v, joinPath(path, key));
}
void readBool(NSDictionary* d, const std::string& path, const char* key, bool& out, bool required = true) {
    if (id v = member(d, path, key, required)) out = toBool(v, joinPath(path, key));
}

// Shortest decimal that round-trips to the same float, so saved files stay readable
// and load(save(s)) reproduces every float bit-exactly.
NSNumber* jsonFloat(float f) {
    if (!std::isfinite(f)) fail("cannot save a non-finite number");
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof(buf) - 1, f);
    *r.ptr = '\0';
    return @(std::strtod(buf, nullptr));
}

NSArray* jsonFloat3(simd_float3 v) { return @[ jsonFloat(v.x), jsonFloat(v.y), jsonFloat(v.z) ]; }

// Oldest scene file version with the cumulus schema (volume.density "cumulus" | "constant").
constexpr uint32_t kMinSceneVersion = 3;

// Every float of the cloud must be finite; lengths that divide or define widths must be positive.
void validateCloud(const CloudDesc& c, const std::string& path) {
    if (c.lobes.empty() || c.lobes.size() > CNR_MAX_LOBES)
        fail(path + ".lobes must have 1.." + std::to_string(CNR_MAX_LOBES) + " entries");
    for (size_t i = 0; i < c.lobes.size(); ++i)
        if (!(c.lobes[i].w > 0)) fail(path + ".lobes[" + std::to_string(i) + "] radius must be > 0");
    if (!(c.lobe_blend > 0)) fail(path + ".lobe_blend must be > 0");
    if (!(c.radius > 0)) fail(path + ".radius must be > 0");
    if (!(c.height > c.anvil_height)) fail(path + ".height must exceed " + path + ".anvil_height");
    if (!(c.anvil_spread >= 0)) fail(path + ".anvil_spread must be >= 0");
    if (!(c.base_softness > 0)) fail(path + ".base_softness must be > 0");
    if (!(c.base_fade_height > 0)) fail(path + ".base_fade_height must be > 0");
    if (!(c.erosion_feature_size > 0)) fail(path + ".erosion_feature_size must be > 0");
    if (c.erosion_octaves < 1 || c.erosion_octaves > 8) fail(path + ".erosion_octaves must be in [1, 8]");
    if (!(c.erosion_amplitude >= 0)) fail(path + ".erosion_amplitude must be >= 0");
    if (!(c.edge > 0)) fail(path + ".edge must be > 0");
}

// volume.cloud is either the full CloudDesc or {"generator_seed": N}, which expands to
// makeRandomCumulus(N, volume) at load time (saved scenes always contain the expanded cloud).
CloudDesc parseCloud(NSDictionary* d, const VolumeDesc& volume) {
    const std::string path = "volume.cloud";
    if (d[@"generator_seed"]) {
        checkKeys(d, path, {"generator_seed"});
        uint32_t seed = 0;
        readUint(d, path, "generator_seed", seed);
        return makeRandomCumulus(seed, volume);
    }
    checkKeys(d, path, {"base_center", "lobes", "lobe_blend", "shear", "anvil_height", "height", "anvil_spread",
                        "radius", "base_softness", "base_fade_height", "erosion_feature_size", "erosion_octaves",
                        "erosion_amplitude", "edge", "seed"});
    CloudDesc c;
    readFloat3(d, path, "base_center", c.base_center);
    id lobes = member(d, path, "lobes", true);
    if (![lobes isKindOfClass:[NSArray class]] || [lobes count] == 0 || [lobes count] > CNR_MAX_LOBES)
        fail(path + ".lobes must be an array of 1.." + std::to_string(CNR_MAX_LOBES) + " [x, y, z, radius] entries");
    for (NSUInteger i = 0; i < [lobes count]; ++i) {
        const std::string lp = path + ".lobes[" + std::to_string(i) + "]";
        id l = lobes[i];
        if (![l isKindOfClass:[NSArray class]] || [l count] != 4) fail(lp + " must be an array of 4 numbers");
        simd_float4 lobe;
        for (int j = 0; j < 4; ++j) lobe[j] = toFloat(l[j], lp + "[" + std::to_string(j) + "]");
        c.lobes.push_back(lobe);
    }
    readFloat(d, path, "lobe_blend", c.lobe_blend);
    id shear = member(d, path, "shear", true);
    if (![shear isKindOfClass:[NSArray class]] || [shear count] != 2) fail(path + ".shear must be an array of 2 numbers");
    for (int j = 0; j < 2; ++j) c.shear[j] = toFloat(shear[j], path + ".shear[" + std::to_string(j) + "]");
    readFloat(d, path, "anvil_height", c.anvil_height);
    readFloat(d, path, "height", c.height);
    readFloat(d, path, "anvil_spread", c.anvil_spread);
    readFloat(d, path, "radius", c.radius);
    readFloat(d, path, "base_softness", c.base_softness);
    readFloat(d, path, "base_fade_height", c.base_fade_height);
    readFloat(d, path, "erosion_feature_size", c.erosion_feature_size);
    readUint(d, path, "erosion_octaves", c.erosion_octaves);
    readFloat(d, path, "erosion_amplitude", c.erosion_amplitude);
    readFloat(d, path, "edge", c.edge);
    readUint(d, path, "seed", c.seed);
    validateCloud(c, path);
    return c;
}

NSDictionary* cloudToJSON(const CloudDesc& c) {
    NSMutableArray* lobes = [NSMutableArray array];
    for (simd_float4 l : c.lobes) [lobes addObject:@[ jsonFloat(l.x), jsonFloat(l.y), jsonFloat(l.z), jsonFloat(l.w) ]];
    return @{
        @"base_center" : jsonFloat3(c.base_center),
        @"lobes" : lobes,
        @"lobe_blend" : jsonFloat(c.lobe_blend),
        @"shear" : @[ jsonFloat(c.shear.x), jsonFloat(c.shear.y) ],
        @"anvil_height" : jsonFloat(c.anvil_height),
        @"height" : jsonFloat(c.height),
        @"anvil_spread" : jsonFloat(c.anvil_spread),
        @"radius" : jsonFloat(c.radius),
        @"base_softness" : jsonFloat(c.base_softness),
        @"base_fade_height" : jsonFloat(c.base_fade_height),
        @"erosion_feature_size" : jsonFloat(c.erosion_feature_size),
        @"erosion_octaves" : @(c.erosion_octaves),
        @"erosion_amplitude" : jsonFloat(c.erosion_amplitude),
        @"edge" : jsonFloat(c.edge),
        @"seed" : @(c.seed),
    };
}

float hostSmoothstep(float e0, float e1, float x) {
    float t = std::clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Conservative cloud-local AABB of the region where the baked cumulus density can be > 0
// (bake_density.metal): smooth_max adds at most lobe_blend / 4 per lobe, erosion at most
// erosion_amplitude, and the field is <= p.y, so density is 0 for p.y <= 0. The anvil scale
// s(y) = 1 + anvil_spread / radius * smoothstep(anvil_height, height, y) is monotone in y.
void cloudLocalBounds(const CloudDesc& c, simd_float3* lo, simd_float3* hi) {
    const float inflate = float(c.lobes.size() - 1) * c.lobe_blend * 0.25f + c.erosion_amplitude;
    const float k = c.anvil_spread / c.radius;
    const float inf = std::numeric_limits<float>::infinity();
    *lo = simd_make_float3(inf, inf, inf);
    *hi = -*lo;
    for (simd_float4 l : c.lobes) {
        const float R = l.w + inflate;
        const float ylo = std::max(l.y - R, 0.0f);
        const float yhi = l.y + R;
        if (yhi <= 0.0f) continue;
        const float s0 = 1.0f + k * hostSmoothstep(c.anvil_height, c.height, ylo);
        const float s1 = 1.0f + k * hostSmoothstep(c.anvil_height, c.height, yhi);
        lo->y = std::min(lo->y, ylo);
        hi->y = std::max(hi->y, yhi);
        const int axes[2] = {0, 2};
        for (int a = 0; a < 2; ++a) {
            const int axis = axes[a];
            const float qlo = l[axis] - R, qhi = l[axis] + R;
            const float sh = c.shear[a];
            (*lo)[axis] = std::min((*lo)[axis], std::min(qlo * s0, qlo * s1) + std::min(sh * ylo, sh * yhi));
            (*hi)[axis] = std::max((*hi)[axis], std::max(qhi * s0, qhi * s1) + std::max(sh * ylo, sh * yhi));
        }
    }
    if (lo->y > hi->y) *lo = *hi = simd_make_float3(0.0f, 0.0f, 0.0f);  // every lobe lies below the base
}

NSDictionary* readJSONFile(const std::string& path) {
    NSError* err = nil;
    NSData* data = [NSData dataWithContentsOfFile:@(path.c_str()) options:0 error:&err];
    if (!data) fail("cannot read " + path + ": " + std::string(err.localizedDescription.UTF8String));
    id root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
    if (!root) fail("invalid JSON in " + path + ": " + std::string(err.localizedDescription.UTF8String));
    return asDict(root, "");
}

void writeJSONFile(id object, const std::string& path) {
    NSError* err = nil;
    NSData* data = [NSJSONSerialization dataWithJSONObject:object
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:&err];
    if (!data) fail("cannot serialize JSON: " + std::string(err.localizedDescription.UTF8String));
    if (![data writeToFile:@(path.c_str()) options:NSDataWritingAtomic error:&err])
        fail("cannot write " + path + ": " + std::string(err.localizedDescription.UTF8String));
}

Scene parseScene(NSDictionary* root) {
    Scene s;
    checkKeys(root, "", {"version", "name", "cameras", "film", "volume", "medium", "light"});
    readUint(root, "", "version", s.version);
    if (s.version < kMinSceneVersion)
        fail("version " + std::to_string(s.version) + " predates the cumulus scene schema (version >= " +
             std::to_string(kMinSceneVersion) + " required)");
    readString(root, "", "name", s.name);

    id cams = member(root, "", "cameras", true);
    if (![cams isKindOfClass:[NSArray class]] || [cams count] == 0) fail("cameras must be a non-empty array");
    std::set<std::string> names;
    for (NSUInteger i = 0; i < [cams count]; ++i) {
        std::string path = "cameras[" + std::to_string(i) + "]";
        NSDictionary* c = asDict(cams[i], path);
        checkKeys(c, path, {"name", "position", "target", "vfov_deg", "exposure_ev"});
        CameraDesc cam;
        readString(c, path, "name", cam.name);
        readFloat3(c, path, "position", cam.position);
        readFloat3(c, path, "target", cam.target);
        readFloat(c, path, "vfov_deg", cam.vfov_deg);
        readFloat(c, path, "exposure_ev", cam.exposure_ev);
        if (cam.name.empty()) fail(path + ".name must be non-empty");
        if (!names.insert(cam.name).second) fail("duplicate camera name " + cam.name);
        if (!(cam.vfov_deg > 0 && cam.vfov_deg < 180)) fail(path + ".vfov_deg must be in (0, 180)");
        simd_float3 f = cam.target - cam.position;
        if (simd_length(f) == 0) fail(path + ": position equals target");
        if (simd_length(simd_cross(simd_normalize(f), simd_make_float3(0.0f, 1.0f, 0.0f))) < 1e-6f)
            fail(path + ": view direction is parallel to +Y");
        s.cameras.push_back(cam);
    }

    NSDictionary* film = asDict(member(root, "", "film", true), "film");
    checkKeys(film, "film", {"width", "height"});
    readUint(film, "film", "width", s.width);
    readUint(film, "film", "height", s.height);
    if (s.width == 0 || s.height == 0) fail("film.width and film.height must be > 0");

    NSDictionary* vol = asDict(member(root, "", "volume", true), "volume");
    checkKeys(vol, "volume", {"bounds_min", "bounds_max", "grid", "density", "cloud", "constant_density"});
    VolumeDesc& v = s.volume;
    readFloat3(vol, "volume", "bounds_min", v.bounds_min);
    readFloat3(vol, "volume", "bounds_max", v.bounds_max);
    v.grid = toUint3(member(vol, "volume", "grid", true), "volume.grid");
    readString(vol, "volume", "density", v.density);
    if (!simd_all(v.bounds_max > v.bounds_min)) fail("volume.bounds_max must exceed volume.bounds_min on every axis");
    if (v.grid.x == 0 || v.grid.y == 0 || v.grid.z == 0) fail("volume.grid must be positive");
    if (v.density == "cumulus") {
        if (vol[@"constant_density"]) fail("volume.constant_density is only allowed with volume.density \"constant\"");
        v.cloud = parseCloud(asDict(member(vol, "volume", "cloud", true), "volume.cloud"), v);
    } else if (v.density == "constant") {
        if (vol[@"cloud"]) fail("volume.cloud is only allowed with volume.density \"cumulus\"");
        readFloat(vol, "volume", "constant_density", v.constant_density, false);
        if (!(v.constant_density >= 0 && v.constant_density <= 1)) fail("volume.constant_density must be in [0, 1]");
    } else {
        fail("volume.density must be \"cumulus\" or \"constant\"");
    }

    NSDictionary* med = asDict(member(root, "", "medium", true), "medium");
    checkKeys(med, "medium", {"sigma_t_max", "albedo", "hg_g"});
    readFloat(med, "medium", "sigma_t_max", s.medium.sigma_t_max);
    readFloat(med, "medium", "albedo", s.medium.albedo);
    readFloat(med, "medium", "hg_g", s.medium.hg_g);
    if (!(s.medium.sigma_t_max >= 0)) fail("medium.sigma_t_max must be >= 0");
    if (!(s.medium.albedo >= 0 && s.medium.albedo <= 1)) fail("medium.albedo must be in [0, 1]");
    if (!(std::fabs(s.medium.hg_g) < 1)) fail("medium.hg_g must satisfy |g| < 1");

    NSDictionary* light = asDict(member(root, "", "light", true), "light");
    checkKeys(light, "light", {"sun_direction", "sun_irradiance", "sky_radiance", "ground_radiance"});
    readFloat3(light, "light", "sun_direction", s.light.sun_direction);
    readFloat3(light, "light", "sun_irradiance", s.light.sun_irradiance);
    readFloat3(light, "light", "sky_radiance", s.light.sky_radiance);
    readFloat3(light, "light", "ground_radiance", s.light.ground_radiance);
    if (!(simd_length(s.light.sun_direction) > 0)) fail("light.sun_direction must be non-zero");
    s.light.sun_direction = simd_normalize(s.light.sun_direction);
    return s;
}

NSDictionary* sceneToJSON(const Scene& s) {
    NSMutableArray* cams = [NSMutableArray array];
    for (const CameraDesc& c : s.cameras) {
        [cams addObject:@{
            @"name" : @(c.name.c_str()),
            @"position" : jsonFloat3(c.position),
            @"target" : jsonFloat3(c.target),
            @"vfov_deg" : jsonFloat(c.vfov_deg),
            @"exposure_ev" : jsonFloat(c.exposure_ev),
        }];
    }
    const VolumeDesc& v = s.volume;
    NSMutableDictionary* volume = [@{
        @"bounds_min" : jsonFloat3(v.bounds_min),
        @"bounds_max" : jsonFloat3(v.bounds_max),
        @"grid" : @[ @(v.grid.x), @(v.grid.y), @(v.grid.z) ],
        @"density" : @(v.density.c_str()),
    } mutableCopy];
    if (v.density == "cumulus")
        volume[@"cloud"] = cloudToJSON(v.cloud);
    else
        volume[@"constant_density"] = jsonFloat(v.constant_density);
    return @{
        @"version" : @(s.version),
        @"name" : @(s.name.c_str()),
        @"cameras" : cams,
        @"film" : @{@"width" : @(s.width), @"height" : @(s.height)},
        @"volume" : volume,
        @"medium" : @{
            @"sigma_t_max" : jsonFloat(s.medium.sigma_t_max),
            @"albedo" : jsonFloat(s.medium.albedo),
            @"hg_g" : jsonFloat(s.medium.hg_g),
        },
        @"light" : @{
            @"sun_direction" : jsonFloat3(s.light.sun_direction),
            @"sun_irradiance" : jsonFloat3(s.light.sun_irradiance),
            @"sky_radiance" : jsonFloat3(s.light.sky_radiance),
            @"ground_radiance" : jsonFloat3(s.light.ground_radiance),
        },
    };
}

// The shared march keys (density, analytic, neural); all optional with the RenderConfig defaults.
void readMarch(NSDictionary* root, RenderConfig& c) {
    readFloat(root, "", "step_scale", c.step_scale, false);
    readUint(root, "", "max_steps", c.max_steps, false);
    readFloat(root, "", "min_transmittance", c.min_transmittance, false);
    readBool(root, "", "dither", c.dither, false);
    readBool(root, "", "empty_skip", c.empty_skip, false);
    if (!(c.step_scale > 0)) fail("step_scale must be > 0");
    if (c.max_steps == 0) fail("max_steps must be > 0");
    if (!(c.min_transmittance >= 0 && c.min_transmittance < 1)) fail("min_transmittance must be in [0, 1)");
}

void writeMarch(NSMutableDictionary* d, const RenderConfig& c) {
    d[@"step_scale"] = jsonFloat(c.step_scale);
    d[@"max_steps"] = @(c.max_steps);
    d[@"min_transmittance"] = jsonFloat(c.min_transmittance);
    d[@"dither"] = c.dither ? @YES : @NO;
    d[@"empty_skip"] = c.empty_skip ? @YES : @NO;
}

RenderConfig parseRenderConfig(NSDictionary* root) {
    RenderConfig c;
    readString(root, "", "renderer", c.renderer);
    if (c.renderer == "density") {
        checkKeys(root, "", {"renderer", "step_scale", "max_steps", "min_transmittance", "dither", "empty_skip"});
        readMarch(root, c);
    } else if (c.renderer == "analytic") {
        checkKeys(root, "", {"renderer", "step_scale", "max_steps", "min_transmittance", "dither", "empty_skip",
                             "shadow_steps", "shadow_distance", "ambient_steps", "octaves", "ms_a", "ms_b", "ms_c",
                             "ambient"});
        readMarch(root, c);
        readUint(root, "", "shadow_steps", c.shadow_steps, false);
        readFloat(root, "", "shadow_distance", c.shadow_distance, false);
        readUint(root, "", "ambient_steps", c.ambient_steps, false);
        readUint(root, "", "octaves", c.octaves, false);
        readFloat(root, "", "ms_a", c.ms_a, false);
        readFloat(root, "", "ms_b", c.ms_b, false);
        readFloat(root, "", "ms_c", c.ms_c, false);
        readBool(root, "", "ambient", c.ambient, false);
        if (c.shadow_steps == 0) fail("shadow_steps must be > 0");
        if (!(c.shadow_distance >= 0)) fail("shadow_distance must be >= 0");
        if (c.octaves == 0) fail("octaves must be >= 1");
        if (c.ambient && c.ambient_steps == 0) fail("ambient_steps must be > 0 when ambient is true");
    } else if (c.renderer == "pt") {
        checkKeys(root, "", {"renderer", "spp", "seed", "max_bounces", "rr_start_bounce"});
        readUint(root, "", "spp", c.spp, false);
        readUint(root, "", "seed", c.seed, false);
        readUint(root, "", "max_bounces", c.max_bounces, false);
        readUint(root, "", "rr_start_bounce", c.rr_start_bounce, false);
        if (c.spp == 0) fail("spp must be > 0");
    } else if (c.renderer == "neural") {
        checkKeys(root, "", {"renderer", "model", "inference", "step_scale", "max_steps", "min_transmittance",
                             "dither", "empty_skip"});
        readString(root, "", "model", c.model);
        readString(root, "", "inference", c.inference, false);
        readMarch(root, c);
        if (c.model.empty()) fail("model must be non-empty");
        if (c.inference != "scalar" && c.inference != "shader_ml" && c.inference != "ml_encoder")
            fail("inference must be one of scalar, shader_ml, ml_encoder (got \"" + c.inference + "\")");
    } else {
        fail("renderer must be one of density, analytic, pt, neural (got \"" + c.renderer + "\")");
    }
    return c;
}

NSDictionary* renderConfigToJSON(const RenderConfig& c) {
    NSMutableDictionary* d = [NSMutableDictionary dictionary];
    d[@"renderer"] = @(c.renderer.c_str());
    if (c.renderer == "density") {
        writeMarch(d, c);
    } else if (c.renderer == "analytic") {
        writeMarch(d, c);
        d[@"shadow_steps"] = @(c.shadow_steps);
        d[@"shadow_distance"] = jsonFloat(c.shadow_distance);
        d[@"ambient_steps"] = @(c.ambient_steps);
        d[@"octaves"] = @(c.octaves);
        d[@"ms_a"] = jsonFloat(c.ms_a);
        d[@"ms_b"] = jsonFloat(c.ms_b);
        d[@"ms_c"] = jsonFloat(c.ms_c);
        d[@"ambient"] = c.ambient ? @YES : @NO;
    } else if (c.renderer == "pt") {
        d[@"spp"] = @(c.spp);
        d[@"seed"] = @(c.seed);
        d[@"max_bounces"] = @(c.max_bounces);
        d[@"rr_start_bounce"] = @(c.rr_start_bounce);
    } else if (c.renderer == "neural") {
        d[@"model"] = @(c.model.c_str());
        d[@"inference"] = @(c.inference.c_str());
        writeMarch(d, c);
    } else {
        fail("renderer must be one of density, analytic, pt, neural (got \"" + c.renderer + "\")");
    }
    return d;
}

}  // namespace

const CameraDesc* Scene::camera(const std::string& name) const {
    if (name.empty()) return cameras.empty() ? nullptr : &cameras[0];
    for (const CameraDesc& c : cameras)
        if (c.name == name) return &c;
    return nullptr;
}

std::optional<Scene> loadScene(const std::string& path, std::string* error) {
    @autoreleasepool {
        try {
            return parseScene(readJSONFile(path));
        } catch (const ParseError& e) {
            if (error) *error = path + ": " + e.message;
            return std::nullopt;
        }
    }
}

bool saveScene(const Scene& scene, const std::string& path, std::string* error) {
    @autoreleasepool {
        try {
            writeJSONFile(sceneToJSON(scene), path);
            return true;
        } catch (const ParseError& e) {
            if (error) *error = e.message;
            return false;
        }
    }
}

std::optional<RenderConfig> loadRenderConfig(const std::string& path, std::string* error) {
    @autoreleasepool {
        try {
            return parseRenderConfig(readJSONFile(path));
        } catch (const ParseError& e) {
            if (error) *error = path + ": " + e.message;
            return std::nullopt;
        }
    }
}

bool saveRenderConfig(const RenderConfig& config, const std::string& path, std::string* error) {
    @autoreleasepool {
        try {
            writeJSONFile(renderConfigToJSON(config), path);
            return true;
        } catch (const ParseError& e) {
            if (error) *error = e.message;
            return false;
        }
    }
}

SceneGPU makeSceneGPU(const Scene& scene, const CameraDesc& camera) {
    SceneGPU g{};
    simd_float3 forward = simd_normalize(camera.target - camera.position);
    simd_float3 right = simd_normalize(simd_cross(forward, simd_make_float3(0.0f, 1.0f, 0.0f)));
    g.cam_position = camera.position;
    g.cam_forward = forward;
    g.cam_right = right;
    g.cam_up = simd_cross(right, forward);
    g.bounds_min = scene.volume.bounds_min;
    g.bounds_max = scene.volume.bounds_max;
    g.sun_direction = simd_normalize(scene.light.sun_direction);
    g.sun_irradiance = scene.light.sun_irradiance;
    g.sky_radiance = scene.light.sky_radiance;
    g.ground_radiance = scene.light.ground_radiance;
    g.tan_half_vfov = std::tan(camera.vfov_deg * float(M_PI) / 360.0f);
    g.aspect = float(scene.width) / float(scene.height);
    g.exposure_ev = camera.exposure_ev;
    g.sigma_t_max = scene.medium.sigma_t_max;
    g.albedo = scene.medium.albedo;
    g.hg_g = scene.medium.hg_g;
    simd_float3 voxel = (scene.volume.bounds_max - scene.volume.bounds_min) / simd_float(scene.volume.grid);
    g.voxel_min_extent = simd_reduce_min(voxel);
    g.width = scene.width;
    g.height = scene.height;
    uint32_t n = simd_reduce_max(scene.volume.grid);
    uint32_t levels = 1;
    while (n >>= 1) ++levels;  // 1 + floor(log2(max(grid)))
    g.mip_levels = levels;
    return g;
}

DensityBakeParams makeDensityBakeParams(const Scene& scene) {
    const VolumeDesc& v = scene.volume;
    DensityBakeParams p{};
    p.bounds_min = v.bounds_min;
    p.bounds_max = v.bounds_max;
    p.grid_x = v.grid.x;
    p.grid_y = v.grid.y;
    p.grid_z = v.grid.z;
    if (v.density == "constant") {
        p.density_type = CNR_DENSITY_CONSTANT;
        p.constant_density = v.constant_density;
        return p;
    }
    const CloudDesc& c = v.cloud;
    p.density_type = CNR_DENSITY_CUMULUS;
    p.base_center = c.base_center;
    p.seed = c.seed;
    p.lobe_count = uint32_t(std::min<size_t>(c.lobes.size(), CNR_MAX_LOBES));
    for (uint32_t i = 0; i < p.lobe_count; ++i) p.lobes[i] = c.lobes[i];
    p.erosion_octaves = c.erosion_octaves;
    p.edge = c.edge;
    p.lobe_blend = c.lobe_blend;
    p.shear_x = c.shear.x;
    p.shear_z = c.shear.y;
    p.anvil_height = c.anvil_height;
    p.cloud_height = c.height;
    p.anvil_spread = c.anvil_spread;
    p.cloud_radius = c.radius;
    p.base_softness = c.base_softness;
    p.base_fade_height = c.base_fade_height;
    p.erosion_feature_size = c.erosion_feature_size;
    p.erosion_amplitude = c.erosion_amplitude;
    return p;
}

MarchParams makeMarchParams(const RenderConfig& config) {
    MarchParams m{};
    m.step_scale = config.step_scale;
    m.min_transmittance = config.min_transmittance;
    m.max_steps = config.max_steps;
    m.flags = (config.dither ? CNR_MARCH_DITHER : 0u) | (config.empty_skip ? CNR_MARCH_EMPTY_SKIP : 0u);
    return m;
}

// Layout: a big core lobe (r 170-220 m) sitting on the flat base, then 3-8 smaller lobes (r 80-220 m,
// 0.6-0.9 of their parent) each stacked on a random earlier lobe, offset sideways by 0.25-0.6 and upward
// by 0.15-0.5 of the parent radius. Shear per axis in [-0.3, 0.3], anvil from 55-80 % of the cloud
// height, erosion amplitude 20-50 m, feature size 35-90 m, edge 8-35 m. Layouts whose conservative
// bounds (cloudLocalBounds) do not fit the volume with a margin are redrawn from the same stream; after
// kAttempts the last layout is scaled down uniformly (only for volumes far smaller than ~1 km).
// base_center places the bounds inside the volume at a random horizontal position and a base height of
// up to 150 m above the lowest valid position.
CloudDesc makeRandomCumulus(uint32_t seed, const VolumeDesc& volume) {
    std::mt19937_64 rng(seed);
    // Explicit 53-bit mapping instead of std::uniform_real_distribution: identical shapes on every library.
    auto uniform = [&rng](float a, float b) { return a + (b - a) * float(double(rng() >> 11) * 0x1.0p-53); };
    auto pick = [&rng](uint32_t n) { return uint32_t(rng() % n); };
    constexpr float kTwoPi = 6.28318530717958647692f;
    constexpr int kAttempts = 256;

    const simd_float3 vext = volume.bounds_max - volume.bounds_min;
    const simd_float3 voxel = vext / simd_float(simd_max(volume.grid, simd_make_uint3(1, 1, 1)));
    const float margin = std::min(std::max(20.0f, 2.0f * simd_reduce_max(voxel)), 0.05f * simd_reduce_min(vext));
    const simd_float3 kZero3 = simd_make_float3(0.0f, 0.0f, 0.0f);

    CloudDesc c;
    simd_float3 lo{}, hi{};
    bool fits = false;
    for (int attempt = 0; attempt < kAttempts && !fits; ++attempt) {
        c = CloudDesc{};
        c.seed = seed;
        // One draw per statement: function-argument evaluation order is unspecified.
        const uint32_t lobeCount = 4 + pick(9);
        const float r0 = uniform(150.0f, 220.0f);
        const float coreX = uniform(-20.0f, 20.0f);
        const float coreY = r0 * uniform(0.3f, 0.5f);
        const float coreZ = uniform(-20.0f, 20.0f);
        c.lobes.push_back(simd_make_float4(coreX, coreY, coreZ, r0));
        for (uint32_t i = 1; i < lobeCount; ++i) {
            const simd_float4 parent = c.lobes[pick(i)];
            const float r = std::clamp(parent.w * uniform(0.55f, 0.9f), 70.0f, 220.0f);
            const float phi = uniform(0.0f, kTwoPi);
            const float side = parent.w * uniform(0.25f, 0.6f);
            const float up = parent.w * uniform(0.15f, 0.5f);
            c.lobes.push_back(simd_make_float4(parent.x + side * std::cos(phi), std::max(parent.y + up, 0.3f * r),
                                               parent.z + side * std::sin(phi), r));
        }
        float top = 0.0f;
        for (simd_float4 l : c.lobes) top = std::max(top, l.y + l.w);
        c.lobe_blend = uniform(15.0f, 45.0f);
        c.shear.x = uniform(-0.3f, 0.3f);
        c.shear.y = uniform(-0.3f, 0.3f);
        c.height = top * uniform(0.85f, 1.0f);
        c.anvil_height = c.height * uniform(0.55f, 0.8f);
        c.radius = uniform(250.0f, 350.0f);
        c.anvil_spread = uniform(30.0f, 120.0f);
        c.base_softness = uniform(8.0f, 20.0f);
        c.base_fade_height = uniform(40.0f, 90.0f);
        c.erosion_feature_size = uniform(35.0f, 90.0f);
        c.erosion_octaves = 4 + pick(2);
        c.erosion_amplitude = uniform(20.0f, 50.0f);
        c.edge = uniform(8.0f, 35.0f);
        cloudLocalBounds(c, &lo, &hi);
        fits = simd_all(hi - lo + 2.0f * margin <= vext);
    }
    if (!fits) {
        // Every length scales by s and the shear and anvil ratio are scale-free, so the bounds scale by s.
        const float s = simd_reduce_min(simd_max(vext - 2.0f * margin, kZero3) / (hi - lo));
        for (simd_float4& l : c.lobes) l *= s;
        c.lobe_blend *= s;
        c.height *= s;
        c.anvil_height *= s;
        c.radius *= s;
        c.anvil_spread *= s;
        c.base_softness *= s;
        c.base_fade_height *= s;
        c.erosion_feature_size *= s;
        c.erosion_amplitude *= s;
        c.edge *= s;
        lo *= s;
        hi *= s;
    }
    const simd_float3 slack = simd_max(vext - (hi - lo) - 2.0f * margin, kZero3);
    c.base_center.x = volume.bounds_min.x + margin - lo.x + uniform(0.0f, 1.0f) * slack.x;
    c.base_center.z = volume.bounds_min.z + margin - lo.z + uniform(0.0f, 1.0f) * slack.z;
    c.base_center.y = volume.bounds_min.y + margin - lo.y + uniform(0.0f, 1.0f) * std::min(slack.y, 150.0f);
    return c;
}
