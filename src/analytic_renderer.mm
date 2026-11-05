#include "analytic_renderer.h"

#include "density_volume.h"
#include "metal_context.h"

#include <cstring>

namespace {

// Per frame slot: SceneGPU at offset 0, AnalyticParams at kParamsOffset.
constexpr NSUInteger kParamsOffset = 256;
constexpr NSUInteger kSlotStride = 512;
static_assert(sizeof(SceneGPU) <= kParamsOffset, "SceneGPU must fit before the analytic params");
static_assert(kParamsOffset + sizeof(AnalyticParams) <= kSlotStride, "uniform slot too small");

class AnalyticRenderer final : public Renderer {
public:
    ~AnalyticRenderer() override {
        if (!context_) return;
        context_->waitIdle();
        if (uniforms_) context_->removeResident(uniforms_);
        if (output_) context_->removeResident(output_);
        context_->commitResidency();
    }

    bool init(MetalContext& context, const Scene& scene, const RenderConfig& config,
              const DensityVolume& density, std::string* error) {
        if (config.primary_steps == 0 || config.shadow_steps == 0) {
            if (error) *error = "analytic renderer: primary_steps and shadow_steps must be > 0";
            return false;
        }
        if (config.octaves == 0) {
            if (error) *error = "analytic renderer: octaves must be >= 1";
            return false;
        }
        if (config.ambient && config.ambient_steps == 0) {
            if (error) *error = "analytic renderer: ambient_steps must be > 0 when ambient is enabled";
            return false;
        }
        context_ = &context;
        id<MTLDevice> device = context.device();

        pipeline_ = context.makeComputePipeline({"src/shaders/analytic.metal"}, "analytic_render", "", error);
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
            if (error) *error = "analytic renderer: allocation failed";
            return false;
        }
        uniforms_.label = @"analytic uniforms";
        output_.label = @"analytic output";

        params_.primary_steps = config.primary_steps;
        params_.shadow_steps = config.shadow_steps;
        params_.ambient_steps = config.ambient_steps;
        params_.octaves = config.octaves;
        params_.ms_a = config.ms_a;
        params_.ms_b = config.ms_b;
        params_.ms_c = config.ms_c;
        params_.flags = config.ambient ? CNR_ANALYTIC_AMBIENT : 0u;

        [table_ setTexture:density.texture().gpuResourceID atIndex:0];
        [table_ setTexture:output_.gpuResourceID atIndex:1];

        context.addResident(uniforms_);
        context.addResident(output_);
        context.commitResidency();
        return true;
    }

    const char* name() const override { return "analytic"; }

    void encode(id<MTL4ComputeCommandEncoder> encoder, const FrameInputs& in) override {
        const NSUInteger slotOffset = in.slot * kSlotStride;
        uint8_t* slot = static_cast<uint8_t*>(uniforms_.contents) + slotOffset;
        std::memcpy(slot, &in.scene, sizeof(SceneGPU));
        std::memcpy(slot + kParamsOffset, &params_, sizeof(AnalyticParams));

        // The argument table is snapshotted at dispatch, so rebinding per frame is safe.
        [table_ setAddress:uniforms_.gpuAddress + slotOffset atIndex:0];
        [table_ setAddress:uniforms_.gpuAddress + slotOffset + kParamsOffset atIndex:1];
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
    AnalyticParams params_{};
};

}  // namespace

std::unique_ptr<Renderer> makeAnalyticRenderer(MetalContext& context, const Scene& scene,
                                               const RenderConfig& config, const DensityVolume& density,
                                               std::string* error) {
    auto renderer = std::make_unique<AnalyticRenderer>();
    if (!renderer->init(context, scene, config, density, error)) return nullptr;
    return renderer;
}
