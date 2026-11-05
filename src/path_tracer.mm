#include "path_tracer.h"

#include "density_volume.h"
#include "metal_context.h"

#include <cstring>

namespace {

// Per frame slot: SceneGPU at offset 0, PathTracerParams at kParamsOffset.
constexpr NSUInteger kParamsOffset = 256;
constexpr NSUInteger kSlotStride = 512;
static_assert(sizeof(SceneGPU) <= kParamsOffset, "SceneGPU must fit before the path tracer params");
static_assert(kParamsOffset + sizeof(PathTracerParams) <= kSlotStride, "uniform slot too small");

// Two 64-bit sums stored as (lo, hi) uint32 pairs; layout shared with pt_render.
constexpr NSUInteger kCounterWords = 4;
constexpr NSUInteger kCounterBytes = kCounterWords * sizeof(uint32_t);

class PathTracer final : public Renderer {
public:
    ~PathTracer() override {
        if (!context_) return;
        context_->waitIdle();
        if (uniforms_) context_->removeResident(uniforms_);
        if (counters_) context_->removeResident(counters_);
        if (output_) context_->removeResident(output_);
        context_->commitResidency();
    }

    bool init(MetalContext& context, const Scene& scene, const RenderConfig& config,
              const DensityVolume& density, std::string* error) {
        context_ = &context;
        id<MTLDevice> device = context.device();

        pipeline_ = context.makeComputePipeline({"src/shaders/path_tracer.metal"}, "pt_render", "", error);
        if (!pipeline_) return false;
        table_ = context.makeArgumentTable(3, 2, error);
        if (!table_) return false;

        uniforms_ = [device newBufferWithLength:kSlotStride * MetalContext::kFramesInFlight
                                        options:MTLResourceStorageModeShared];
        counters_ = [device newBufferWithLength:kCounterBytes options:MTLResourceStorageModeShared];
        MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                        width:scene.width
                                                                                       height:scene.height
                                                                                    mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        desc.storageMode = MTLStorageModeShared;
        output_ = [device newTextureWithDescriptor:desc];
        if (!uniforms_ || !counters_ || !output_) {
            if (error) *error = "pt renderer: allocation failed";
            return false;
        }
        uniforms_.label = @"pt uniforms";
        counters_.label = @"pt counters";
        output_.label = @"pt accumulation";
        std::memset(counters_.contents, 0, kCounterBytes);

        params_.seed = config.seed;
        params_.sample_index = 0;
        params_.max_bounces = config.max_bounces;
        params_.rr_start_bounce = config.rr_start_bounce;
        params_.bounce_cap = kPathTracerBounceCap;

        [table_ setAddress:counters_.gpuAddress atIndex:2];
        [table_ setTexture:density.texture().gpuResourceID atIndex:0];
        [table_ setTexture:output_.gpuResourceID atIndex:1];

        context.addResident(uniforms_);
        context.addResident(counters_);
        context.addResident(output_);
        context.commitResidency();
        return true;
    }

    const char* name() const override { return "pt"; }

    void encode(id<MTL4ComputeCommandEncoder> encoder, const FrameInputs& in) override {
        // The running mean reads what earlier frames (earlier encoders on this queue) wrote, and
        // a display pass of an earlier frame may still read the output: wait for those stages.
        [encoder barrierAfterQueueStages:MTLStageDispatch | MTLStageBlit | MTLStageFragment
                            beforeStages:MTLStageDispatch | MTLStageBlit
                       visibilityOptions:MTL4VisibilityOptionDevice];
        if (sampleIndex_ == 0) {
            [encoder fillBuffer:counters_ range:NSMakeRange(0, kCounterBytes) value:0];
            [encoder barrierAfterEncoderStages:MTLStageBlit
                           beforeEncoderStages:MTLStageDispatch
                             visibilityOptions:MTL4VisibilityOptionDevice];
        }

        params_.sample_index = sampleIndex_;
        const NSUInteger slotOffset = in.slot * kSlotStride;
        uint8_t* slot = static_cast<uint8_t*>(uniforms_.contents) + slotOffset;
        std::memcpy(slot, &in.scene, sizeof(SceneGPU));
        std::memcpy(slot + kParamsOffset, &params_, sizeof(PathTracerParams));

        // The argument table is snapshotted at dispatch, so rebinding per frame is safe.
        [table_ setAddress:uniforms_.gpuAddress + slotOffset atIndex:0];
        [table_ setAddress:uniforms_.gpuAddress + slotOffset + kParamsOffset atIndex:1];
        [encoder setComputePipelineState:pipeline_];
        [encoder setArgumentTable:table_];
        [encoder dispatchThreads:MTLSizeMake(output_.width, output_.height, 1)
           threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        ++sampleIndex_;
    }

    void resetAccumulation() override { sampleIndex_ = 0; }
    uint32_t accumulatedSamples() const override { return sampleIndex_; }

    id<MTLTexture> output() const override { return output_; }

    uint64_t memoryBytes() const override {
        return uniforms_.allocatedSize + counters_.allocatedSize + output_.allocatedSize;
    }

    std::vector<std::pair<std::string, double>> stats() const override {
        const uint32_t* words = static_cast<const uint32_t*>(counters_.contents);
        // The counters are zeroed by the first encode after a reset; until then report nothing.
        const bool fresh = sampleIndex_ == 0;
        const uint64_t capHits = fresh ? 0 : uint64_t(words[0]) | (uint64_t(words[1]) << 32);
        const uint64_t bounces = fresh ? 0 : uint64_t(words[2]) | (uint64_t(words[3]) << 32);
        const double paths = double(output_.width) * double(output_.height) * double(sampleIndex_);
        return {
            {"bounce_cap_hits", double(capHits)},
            {"mean_bounces", paths > 0 ? double(bounces) / paths : 0.0},
            {"paths", paths},
        };
    }

private:
    MetalContext* context_ = nullptr;
    id<MTLComputePipelineState> pipeline_ = nil;
    id<MTL4ArgumentTable> table_ = nil;
    id<MTLBuffer> uniforms_ = nil;
    id<MTLBuffer> counters_ = nil;
    id<MTLTexture> output_ = nil;
    PathTracerParams params_{};
    uint32_t sampleIndex_ = 0;
};

}  // namespace

std::unique_ptr<Renderer> makePathTracerRenderer(MetalContext& context, const Scene& scene,
                                                 const RenderConfig& config, const DensityVolume& density,
                                                 std::string* error) {
    auto renderer = std::make_unique<PathTracer>();
    if (!renderer->init(context, scene, config, density, error)) return nullptr;
    return renderer;
}
