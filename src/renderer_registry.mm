#include "renderer.h"

#include "analytic_renderer.h"
#include "density_renderer.h"
#include "image_io.h"
#include "neural_renderer.h"
#include "path_tracer.h"

std::unique_ptr<Renderer> makeRenderer(MetalContext& context, const Scene& scene,
                                       const RenderConfig& config, const DensityVolume& density,
                                       std::string* error) {
    if (config.renderer == "density") return makeDensityRenderer(context, scene, config, density, error);
    if (config.renderer == "analytic") return makeAnalyticRenderer(context, scene, config, density, error);
    if (config.renderer == "pt") return makePathTracerRenderer(context, scene, config, density, error);
    if (config.renderer == "neural") return makeNeuralRenderer(context, scene, config, density, error);
    if (error) *error = "renderer '" + config.renderer + "' not available";
    return nullptr;
}

bool readOutput(const Renderer& renderer, Image* image, std::string* error) {
    id<MTLTexture> texture = renderer.output();
    if (!texture || texture.pixelFormat != MTLPixelFormatRGBA32Float ||
        texture.storageMode != MTLStorageModeShared) {
        if (error) *error = "renderer output must be a shared RGBA32Float texture";
        return false;
    }
    const uint32_t width = static_cast<uint32_t>(texture.width);
    const uint32_t height = static_cast<uint32_t>(texture.height);
    std::vector<float> rgba(size_t(width) * height * 4);
    [texture getBytes:rgba.data()
          bytesPerRow:size_t(width) * 4 * sizeof(float)
           fromRegion:MTLRegionMake2D(0, 0, width, height)
          mipmapLevel:0];

    image->width = width;
    image->height = height;
    image->rgb.resize(size_t(width) * height * 3);
    for (size_t i = 0, n = size_t(width) * height; i < n; ++i) {
        image->rgb[i * 3 + 0] = rgba[i * 4 + 0];
        image->rgb[i * 3 + 1] = rgba[i * 4 + 1];
        image->rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    return true;
}
