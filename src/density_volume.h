// Baked density field shared by all renderers (docs/architecture.md §4).
#pragma once

#import <Metal/Metal.h>

#include "scene.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class MetalContext;

class DensityVolume {
public:
    // Allocates an r16Float 3D texture (private storage, full mip chain), dispatches
    // kernel `bake_density` (src/shaders/bake_density.metal) for level 0, then
    // generateMipmapsForTexture. Waits for completion; records GPU bake time.
    // The texture is added to the context residency set.
    static std::unique_ptr<DensityVolume> bake(MetalContext& context, const Scene& scene,
                                               std::string* error);
    ~DensityVolume();

    id<MTLTexture> texture() const;
    uint32_t mipLevels() const;
    double bakeGpuMs() const;
    uint64_t memoryBytes() const;  // texture allocatedSize

    // Copies one mip level to host memory (row-major x, then y, then z), as
    // IEEE half bit patterns. For tests and data inspection only.
    bool readLevel(MetalContext& context, uint32_t level, std::vector<uint16_t>* out,
                   std::string* error) const;

private:
    DensityVolume() = default;
    id<MTLTexture> texture_ = nil;
    MetalContext* context_ = nullptr;
    double bakeGpuMs_ = 0.0;
};
