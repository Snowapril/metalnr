// Linear float RGB images, PFM and PNG I/O, GPU texture readback.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Row 0 is the top row; rgb holds width*height*3 linear floats.
struct Image {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<float> rgb;

    float* pixel(uint32_t x, uint32_t y) { return &rgb[(size_t(y) * width + x) * 3]; }
    const float* pixel(uint32_t x, uint32_t y) const { return &rgb[(size_t(y) * width + x) * 3]; }
};

// Little-endian color PFM ("PF", scale -1.0). PFM stores bottom-to-top; the
// reader/writer convert so Image row 0 is always the top.
bool writePFM(const Image& image, const std::string& path, std::string* error);
bool readPFM(const std::string& path, Image* image, std::string* error);

// Display mapping, identical everywhere (docs/architecture.md §4):
// srgb_oetf(v / (1 + v)) with v = x * 2^exposure_ev, per channel, x clamped to >= 0.
float displayMap(float linear, float exposure_ev);

// 8-bit sRGB PNG through ImageIO, using displayMap.
bool writePNG(const Image& image, float exposure_ev, const std::string& path, std::string* error);

// Writes an already display-referred [0,1] RGB image (heat maps); values clamped.
bool writeDisplayPNG(const Image& display, const std::string& path, std::string* error);
