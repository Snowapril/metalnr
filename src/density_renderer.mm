#include "density_renderer.h"

#include "density_volume.h"
#include "metal_context.h"

#include <cstring>

namespace {

// Per frame slot: SceneGPU at offset 0, DensityViewParams at kViewOffset.
constexpr NSUInteger kViewOffset = 256;
constexpr NSUInteger kSlotStride = 512;
static_assert(sizeof(SceneGPU) <= kViewOffset, "SceneGPU must fit before the view params");
static_assert(kViewOffset + sizeof(DensityViewParams) <= kSlotStride, "uniform slot too small");

class DensityRenderer final : public Renderer {
public:
    ~DensityRenderer() override {
        if (!context_) return;
        context_->waitIdle();
        if (uniforms_) context_->removeResident(uniforms_);
        if (output_) context_->removeResident(output_);
        context_->commitResidency();
    }

    bool init(MetalContext& context, const Scene& scene, const RenderConfig& config,
              const DensityVolume& density, std::string* error) {
        if (config.primary_steps == 0) {
            if (error) *error = "density renderer: primary_steps must be > 0";
            return false;
        }
        context_ = &context;
        id<MTLDevice> device = context.device();

        pipeline_ = context.makeComputePipeline({"src/shaders/density_view.metal"}, "density_view", "", error);
        if (!pipeline_) return false;
        table_ = context.makeArgumentTable(2, 2, error);
        if (!table_) return false;

        uniforms_ = [device newBufferWithLength:kSlotStride * MetalContext::kFramesInFlight
                                        options:MTLResourceStorageModeShared];
        MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                        width:scene.width
                                                                                       height:scene.height
                                                                                    mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        desc.storageMode = MTLStorageModeShared;
        output_ = [device newTextureWithDescriptor:desc];
        if (!uniforms_ || !output_) {
            if (error) *error = "density renderer: allocation failed";
            return false;
        }
        uniforms_.label = @"density_view uniforms";
        output_.label = @"density_view output";

        view_.steps = config.primary_steps;
        [table_ setTexture:density.texture().gpuResourceID atIndex:0];
        [table_ setTexture:output_.gpuResourceID atIndex:1];

        context.addResident(uniforms_);
        context.addResident(output_);
        context.commitResidency();
        return true;
    }

    const char* name() const override { return "density"; }

    void encode(id<MTL4ComputeCommandEncoder> encoder, const FrameInputs& in) override {
        const NSUInteger slotOffset = in.slot * kSlotStride;
        uint8_t* slot = static_cast<uint8_t*>(uniforms_.contents) + slotOffset;
        std::memcpy(slot, &in.scene, sizeof(SceneGPU));
        std::memcpy(slot + kViewOffset, &view_, sizeof(DensityViewParams));

        // The argument table is snapshotted at dispatch, so rebinding per frame is safe.
        [table_ setAddress:uniforms_.gpuAddress + slotOffset atIndex:0];
        [table_ setAddress:uniforms_.gpuAddress + slotOffset + kViewOffset atIndex:1];
        [encoder setComputePipelineState:pipeline_];
        [encoder setArgumentTable:table_];
        [encoder dispatchThreads:MTLSizeMake(output_.width, output_.height, 1)
           threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    }

    id<MTLTexture> output() const override { return output_; }

    uint64_t memoryBytes() const override { return uniforms_.allocatedSize + output_.allocatedSize; }

private:
    MetalContext* context_ = nullptr;
    id<MTLComputePipelineState> pipeline_ = nil;
    id<MTL4ArgumentTable> table_ = nil;
    id<MTLBuffer> uniforms_ = nil;
    id<MTLTexture> output_ = nil;
    DensityViewParams view_{};
};

}  // namespace

std::unique_ptr<Renderer> makeDensityRenderer(MetalContext& context, const Scene& scene,
                                              const RenderConfig& config, const DensityVolume& density,
                                              std::string* error) {
    auto renderer = std::make_unique<DensityRenderer>();
    if (!renderer->init(context, scene, config, density, error)) return nullptr;
    return renderer;
}
