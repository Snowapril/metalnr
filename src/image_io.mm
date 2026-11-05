#include "image_io.h"

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static_assert(std::endian::native == std::endian::little, "PFM I/O assumes a little-endian host");

namespace {

bool setError(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

bool validImage(const Image& image) {
    return image.width > 0 && image.height > 0 && image.rgb.size() == size_t(image.width) * image.height * 3;
}

bool isSpace(uint8_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }

// Reads one whitespace-delimited header token starting at `pos`.
bool nextToken(const std::vector<uint8_t>& data, size_t& pos, std::string* token) {
    while (pos < data.size() && isSpace(data[pos])) ++pos;
    size_t start = pos;
    while (pos < data.size() && !isSpace(data[pos])) ++pos;
    if (pos == start) return false;
    token->assign(reinterpret_cast<const char*>(data.data()) + start, pos - start);
    return true;
}

bool parseDimension(const std::string& token, uint32_t* out) {
    char* end = nullptr;
    unsigned long v = std::strtoul(token.c_str(), &end, 10);
    if (end != token.c_str() + token.size() || token[0] == '-' || v == 0 || v > 1u << 20) return false;
    *out = uint32_t(v);
    return true;
}

float srgbOETF(float t) { return t <= 0.0031308f ? 12.92f * t : 1.055f * std::pow(t, 1.0f / 2.4f) - 0.055f; }

uint8_t toByte(float v) {
    if (!(v > 0)) return 0;  // also maps NaN to 0
    if (v >= 1) return 255;
    return uint8_t(v * 255.0f + 0.5f);
}

// Writes interleaved 8-bit RGB (row 0 = top) as an sRGB PNG.
bool writeRGB8(uint32_t width, uint32_t height, const std::vector<uint8_t>& bytes, const std::string& path,
               std::string* error) {
    @autoreleasepool {
        CFDataRef data = CFDataCreate(nullptr, bytes.data(), CFIndex(bytes.size()));
        CGDataProviderRef provider = CGDataProviderCreateWithCFData(data);
        CGColorSpaceRef space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        CGImageRef cgImage = CGImageCreate(width, height, 8, 24, size_t(width) * 3, space,
                                           CGBitmapInfo(kCGImageAlphaNone), provider, nullptr, false,
                                           kCGRenderingIntentDefault);
        bool ok = false;
        std::string message;
        if (!cgImage) {
            message = "CGImageCreate failed";
        } else {
            NSURL* url = [NSURL fileURLWithPath:@(path.c_str())];
            CGImageDestinationRef dest = CGImageDestinationCreateWithURL(
                (__bridge CFURLRef)url, (__bridge CFStringRef)UTTypePNG.identifier, 1, nullptr);
            if (!dest) {
                message = "cannot create PNG destination " + path;
            } else {
                CGImageDestinationAddImage(dest, cgImage, nullptr);
                ok = CGImageDestinationFinalize(dest);
                if (!ok) message = "cannot write PNG " + path;
                CFRelease(dest);
            }
            CGImageRelease(cgImage);
        }
        CGColorSpaceRelease(space);
        CGDataProviderRelease(provider);
        CFRelease(data);
        return ok || setError(error, message);
    }
}

}  // namespace

bool writePFM(const Image& image, const std::string& path, std::string* error) {
    if (!validImage(image)) return setError(error, "writePFM: invalid image dimensions");
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return setError(error, "cannot open " + path + " for writing");
    bool ok = std::fprintf(f, "PF\n%u %u\n-1.0\n", image.width, image.height) > 0;
    const size_t rowFloats = size_t(image.width) * 3;
    for (uint32_t row = 0; ok && row < image.height; ++row) {
        const float* src = image.pixel(0, image.height - 1 - row);  // PFM rows run bottom to top
        ok = std::fwrite(src, sizeof(float), rowFloats, f) == rowFloats;
    }
    ok = (std::fclose(f) == 0) && ok;
    return ok || setError(error, "write failed for " + path);
}

bool readPFM(const std::string& path, Image* image, std::string* error) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return setError(error, "cannot open " + path);
    std::vector<uint8_t> data;
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) data.insert(data.end(), buf, buf + n);
    bool readError = std::ferror(f) != 0;
    std::fclose(f);
    if (readError) return setError(error, "read failed for " + path);

    size_t pos = 0;
    std::string magic, w, h, scaleToken;
    if (!nextToken(data, pos, &magic) || magic != "PF")
        return setError(error, path + ": not a color PFM (expected magic PF)");
    uint32_t width = 0, height = 0;
    if (!nextToken(data, pos, &w) || !parseDimension(w, &width) || !nextToken(data, pos, &h) ||
        !parseDimension(h, &height))
        return setError(error, path + ": invalid PFM dimensions");
    if (!nextToken(data, pos, &scaleToken)) return setError(error, path + ": missing PFM scale");
    char* end = nullptr;
    double scale = std::strtod(scaleToken.c_str(), &end);
    if (end != scaleToken.c_str() + scaleToken.size() || scale == 0 || !std::isfinite(scale))
        return setError(error, path + ": invalid PFM scale " + scaleToken);
    // Exactly one whitespace character separates the header from the raster.
    if (pos >= data.size() || !isSpace(data[pos])) return setError(error, path + ": truncated PFM header");
    ++pos;

    const size_t count = size_t(width) * height * 3;
    if (data.size() - pos < count * sizeof(float))
        return setError(error, path + ": truncated PFM data (expected " + std::to_string(count * sizeof(float)) +
                                   " bytes, found " + std::to_string(data.size() - pos) + ")");

    const bool bigEndian = scale > 0;
    const size_t rowFloats = size_t(width) * 3;
    Image out;
    out.width = width;
    out.height = height;
    out.rgb.resize(count);
    for (uint32_t row = 0; row < height; ++row) {
        float* dst = out.pixel(0, height - 1 - row);
        std::memcpy(dst, data.data() + pos + row * rowFloats * sizeof(float), rowFloats * sizeof(float));
        if (bigEndian)
            for (size_t i = 0; i < rowFloats; ++i)
                dst[i] = std::bit_cast<float>(__builtin_bswap32(std::bit_cast<uint32_t>(dst[i])));
    }
    *image = std::move(out);
    return true;
}

float displayMap(float linear, float exposure_ev) {
    float v = std::fmax(linear, 0.0f) * std::exp2(exposure_ev);
    float t = std::isinf(v) ? 1.0f : v / (1.0f + v);
    return srgbOETF(t);
}

bool writePNG(const Image& image, float exposure_ev, const std::string& path, std::string* error) {
    if (!validImage(image)) return setError(error, "writePNG: invalid image dimensions");
    std::vector<uint8_t> bytes(image.rgb.size());
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = toByte(displayMap(image.rgb[i], exposure_ev));
    return writeRGB8(image.width, image.height, bytes, path, error);
}

bool writeDisplayPNG(const Image& display, const std::string& path, std::string* error) {
    if (!validImage(display)) return setError(error, "writeDisplayPNG: invalid image dimensions");
    std::vector<uint8_t> bytes(display.rgb.size());
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = toByte(display.rgb[i]);
    return writeRGB8(display.width, display.height, bytes, path, error);
}
