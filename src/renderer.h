// Minimal interface shared by the renderers, the CLI, the viewer and the benchmark.
// Algorithm-specific state stays inside each implementation.
#pragma once

#import <Metal/Metal.h>

#include "scene.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

class MetalContext;
class DensityVolume;

struct FrameInputs {
    SceneGPU scene;   // camera may change between frames (viewer)
    uint32_t slot;    // MetalContext::currentSlot(), selects per-frame uniform storage
};

class Renderer {
public:
    virtual ~Renderer() = default;
    virtual const char* name() const = 0;

    // Records this frame's dispatches into `encoder` (created by the caller from the
    // frame command buffer). Must not end the encoder, commit, or wait on the CPU.
    // Per-frame constants go into storage indexed by in.slot.
    virtual void encode(id<MTL4ComputeCommandEncoder> encoder, const FrameInputs& in) = 0;

    // Progressive renderers restart accumulation; others ignore it.
    virtual void resetAccumulation() {}
    // Samples accumulated in output() after all encoded frames complete (1 if not progressive).
    virtual uint32_t accumulatedSamples() const { return 1; }

    // width x height, MTLPixelFormatRGBA32Float, storageModeShared, usage
    // ShaderRead|ShaderWrite. Linear radiance, no exposure. Valid to read on the CPU
    // after the frames that wrote it completed.
    virtual id<MTLTexture> output() const = 0;

    // Sum of allocatedSize of renderer-owned buffers and textures (excludes the
    // shared density volume).
    virtual uint64_t memoryBytes() const = 0;

    // Renderer-specific counters, read after GPU completion (e.g. bounce cap hits).
    virtual std::vector<std::pair<std::string, double>> stats() const { return {}; }
};

// Defined in renderer_registry.mm. Dispatches on config.renderer.
std::unique_ptr<Renderer> makeRenderer(MetalContext& context, const Scene& scene,
                                       const RenderConfig& config, const DensityVolume& density,
                                       std::string* error);

// Reads renderer.output() into a linear RGB image (drops alpha). Caller guarantees
// the GPU finished writing it.
struct Image;
bool readOutput(const Renderer& renderer, Image* image, std::string* error);
