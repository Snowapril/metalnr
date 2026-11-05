#include "scene.h"

#import <Foundation/Foundation.h>

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
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
    checkKeys(vol, "volume", {"bounds_min", "bounds_max", "grid", "density", "density_seed", "coverage", "erosion",
                              "constant_density"});
    VolumeDesc& v = s.volume;
    readFloat3(vol, "volume", "bounds_min", v.bounds_min);
    readFloat3(vol, "volume", "bounds_max", v.bounds_max);
    v.grid = toUint3(member(vol, "volume", "grid", true), "volume.grid");
    readString(vol, "volume", "density", v.density, false);
    readUint(vol, "volume", "density_seed", v.density_seed);
    readFloat(vol, "volume", "coverage", v.coverage);
    readFloat(vol, "volume", "erosion", v.erosion);
    readFloat(vol, "volume", "constant_density", v.constant_density, false);
    if (!simd_all(v.bounds_max > v.bounds_min)) fail("volume.bounds_max must exceed volume.bounds_min on every axis");
    if (v.grid.x == 0 || v.grid.y == 0 || v.grid.z == 0) fail("volume.grid must be positive");
    if (v.density != "procedural" && v.density != "constant")
        fail("volume.density must be \"procedural\" or \"constant\"");
    if (!(v.coverage >= 0 && v.coverage <= 1)) fail("volume.coverage must be in [0, 1]");
    if (!(v.erosion >= 0)) fail("volume.erosion must be >= 0");
    if (!(v.constant_density >= 0 && v.constant_density <= 1)) fail("volume.constant_density must be in [0, 1]");

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
    return @{
        @"version" : @(s.version),
        @"name" : @(s.name.c_str()),
        @"cameras" : cams,
        @"film" : @{@"width" : @(s.width), @"height" : @(s.height)},
        @"volume" : @{
            @"bounds_min" : jsonFloat3(v.bounds_min),
            @"bounds_max" : jsonFloat3(v.bounds_max),
            @"grid" : @[ @(v.grid.x), @(v.grid.y), @(v.grid.z) ],
            @"density" : @(v.density.c_str()),
            @"density_seed" : @(v.density_seed),
            @"coverage" : jsonFloat(v.coverage),
            @"erosion" : jsonFloat(v.erosion),
            @"constant_density" : jsonFloat(v.constant_density),
        },
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

RenderConfig parseRenderConfig(NSDictionary* root) {
    RenderConfig c;
    readString(root, "", "renderer", c.renderer);
    if (c.renderer == "density") {
        checkKeys(root, "", {"renderer", "primary_steps"});
        readUint(root, "", "primary_steps", c.primary_steps, false);
    } else if (c.renderer == "analytic") {
        checkKeys(root, "", {"renderer", "primary_steps", "shadow_steps", "ambient_steps", "octaves", "ms_a", "ms_b",
                             "ms_c", "ambient"});
        readUint(root, "", "primary_steps", c.primary_steps, false);
        readUint(root, "", "shadow_steps", c.shadow_steps, false);
        readUint(root, "", "ambient_steps", c.ambient_steps, false);
        readUint(root, "", "octaves", c.octaves, false);
        readFloat(root, "", "ms_a", c.ms_a, false);
        readFloat(root, "", "ms_b", c.ms_b, false);
        readFloat(root, "", "ms_c", c.ms_c, false);
        readBool(root, "", "ambient", c.ambient, false);
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
        checkKeys(root, "", {"renderer", "model", "primary_steps", "shadow_steps"});
        readString(root, "", "model", c.model);
        readUint(root, "", "primary_steps", c.primary_steps, false);
        readUint(root, "", "shadow_steps", c.shadow_steps, false);
        if (c.model.empty()) fail("model must be non-empty");
    } else {
        fail("renderer must be one of density, analytic, pt, neural (got \"" + c.renderer + "\")");
    }
    if (c.renderer != "pt" && c.primary_steps == 0) fail("primary_steps must be > 0");
    if ((c.renderer == "analytic" || c.renderer == "neural") && c.shadow_steps == 0) fail("shadow_steps must be > 0");
    return c;
}

NSDictionary* renderConfigToJSON(const RenderConfig& c) {
    if (c.renderer == "density") return @{@"renderer" : @"density", @"primary_steps" : @(c.primary_steps)};
    if (c.renderer == "analytic")
        return @{
            @"renderer" : @"analytic",
            @"primary_steps" : @(c.primary_steps),
            @"shadow_steps" : @(c.shadow_steps),
            @"ambient_steps" : @(c.ambient_steps),
            @"octaves" : @(c.octaves),
            @"ms_a" : jsonFloat(c.ms_a),
            @"ms_b" : jsonFloat(c.ms_b),
            @"ms_c" : jsonFloat(c.ms_c),
            @"ambient" : c.ambient ? @YES : @NO,
        };
    if (c.renderer == "pt")
        return @{
            @"renderer" : @"pt",
            @"spp" : @(c.spp),
            @"seed" : @(c.seed),
            @"max_bounces" : @(c.max_bounces),
            @"rr_start_bounce" : @(c.rr_start_bounce),
        };
    if (c.renderer == "neural")
        return @{
            @"renderer" : @"neural",
            @"model" : @(c.model.c_str()),
            @"primary_steps" : @(c.primary_steps),
            @"shadow_steps" : @(c.shadow_steps),
        };
    fail("renderer must be one of density, analytic, pt, neural (got \"" + c.renderer + "\")");
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
    p.density_type = v.density == "constant" ? CNR_DENSITY_CONSTANT : CNR_DENSITY_PROCEDURAL;
    p.density_seed = v.density_seed;
    p.coverage = v.coverage;
    p.erosion = v.erosion;
    p.constant_density = v.constant_density;
    return p;
}
