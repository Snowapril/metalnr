#include "density_volume.h"

#include "metal_context.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

uint32_t fullMipCount(simd_uint3 grid) {
    const uint32_t largest = std::max({grid.x, grid.y, grid.z});
    return 1 + static_cast<uint32_t>(std::floor(std::log2(static_cast<double>(largest))));
}

}  // namespace

std::unique_ptr<DensityVolume> DensityVolume::bake(MetalContext& context, const Scene& scene,
                                                   std::string* error) {
    const simd_uint3 grid = scene.volume.grid;
    if (grid.x == 0 || grid.y == 0 || grid.z == 0) {
        if (error) *error = "volume grid must be positive in every axis";
        return nullptr;
    }
    id<MTLDevice> device = context.device();

    MTLTextureDescriptor* desc = [MTLTextureDescriptor new];
    desc.textureType = MTLTextureType3D;
    desc.pixelFormat = MTLPixelFormatR16Float;
    desc.width = grid.x;
    desc.height = grid.y;
    desc.depth = grid.z;
    desc.mipmapLevelCount = fullMipCount(grid);
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModePrivate;
    id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
    if (!texture) {
        if (error) *error = "density texture allocation failed";
        return nullptr;
    }
    texture.label = @"density";

    id<MTLComputePipelineState> pipeline =
        context.makeComputePipeline({"src/shaders/bake_density.metal"}, "bake_density", "", error);
    if (!pipeline) return nullptr;
    id<MTL4ArgumentTable> table = context.makeArgumentTable(1, 1, error);
    if (!table) return nullptr;

    const DensityBakeParams params = makeDensityBakeParams(scene);
    id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params
                                                     length:sizeof(params)
                                                    options:MTLResourceStorageModeShared];
    if (!paramsBuffer) {
        if (error) *error = "bake parameter buffer allocation failed";
        return nullptr;
    }
    [table setAddress:paramsBuffer.gpuAddress atIndex:0];
    [table setTexture:texture.gpuResourceID atIndex:0];

    context.addResident(texture);
    context.addResident(paramsBuffer);
    context.commitResidency();

    // 256 threads per group when the pipeline allows it; 64 is always available.
    const MTLSize group = pipeline.maxTotalThreadsPerThreadgroup >= 256 ? MTLSizeMake(8, 8, 4)
                                                                         : MTLSizeMake(4, 4, 4);
    std::unique_ptr<DensityVolume> volume(new DensityVolume());
    const bool ok = context.runSync(
        [&](id<MTL4CommandBuffer> commandBuffer) {
            id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setArgumentTable:table];
            [encoder dispatchThreads:MTLSizeMake(grid.x, grid.y, grid.z) threadsPerThreadgroup:group];
            [encoder barrierAfterEncoderStages:MTLStageDispatch
                           beforeEncoderStages:MTLStageBlit
                             visibilityOptions:MTL4VisibilityOptionDevice];
            [encoder generateMipmapsForTexture:texture];
            [encoder endEncoding];
        },
        &volume->bakeGpuMs_, error);

    context.removeResident(paramsBuffer);
    context.commitResidency();
    if (!ok) {
        context.removeResident(texture);
        context.commitResidency();
        return nullptr;
    }
    volume->texture_ = texture;
    volume->context_ = &context;
    return volume;
}

DensityVolume::~DensityVolume() {
    if (context_ && texture_) {
        // Frames that sample the volume may still be in flight.
        context_->waitIdle();
        context_->removeResident(texture_);
        context_->commitResidency();
    }
}

id<MTLTexture> DensityVolume::texture() const { return texture_; }
uint32_t DensityVolume::mipLevels() const { return static_cast<uint32_t>(texture_.mipmapLevelCount); }
double DensityVolume::bakeGpuMs() const { return bakeGpuMs_; }
uint64_t DensityVolume::memoryBytes() const { return texture_.allocatedSize; }

bool DensityVolume::readLevel(MetalContext& context, uint32_t level, std::vector<uint16_t>* out,
                              std::string* error) const {
    if (level >= texture_.mipmapLevelCount) {
        if (error) *error = "mip level out of range";
        return false;
    }
    const NSUInteger w = std::max<NSUInteger>(1, texture_.width >> level);
    const NSUInteger h = std::max<NSUInteger>(1, texture_.height >> level);
    const NSUInteger d = std::max<NSUInteger>(1, texture_.depth >> level);
    const NSUInteger bytesPerRow = w * sizeof(uint16_t);
    const NSUInteger bytesPerImage = bytesPerRow * h;
    const NSUInteger totalBytes = bytesPerImage * d;

    id<MTLBuffer> buffer = [context.device() newBufferWithLength:totalBytes
                                                         options:MTLResourceStorageModeShared];
    if (!buffer) {
        if (error) *error = "readback buffer allocation failed";
        return false;
    }
    context.addResident(buffer);
    context.commitResidency();
    const bool ok = context.runSync(
        [&](id<MTL4CommandBuffer> commandBuffer) {
            id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder copyFromTexture:texture_
                         sourceSlice:0
                         sourceLevel:level
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:MTLSizeMake(w, h, d)
                            toBuffer:buffer
                   destinationOffset:0
              destinationBytesPerRow:bytesPerRow
            destinationBytesPerImage:bytesPerImage];
            [encoder endEncoding];
        },
        nullptr, error);
    context.removeResident(buffer);
    context.commitResidency();
    if (!ok) return false;

    out->resize(totalBytes / sizeof(uint16_t));
    std::memcpy(out->data(), buffer.contents, totalBytes);
    return true;
}
