// Owns the Metal 4 device objects shared by every renderer and tool.
// Objective-C++ only (include from .mm files).
#pragma once

#import <Metal/Metal.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class MetalContext {
public:
    static constexpr uint32_t kFramesInFlight = 2;

    // Creates device, MTL4CommandQueue, MTL4Compiler, one MTL4CommandAllocator per
    // frame slot, one MTLResidencySet (attached to the queue), and the frame
    // MTLSharedEvent. Returns nullptr and sets *error on failure (including when
    // MTLGPUFamilyMetal4 is unsupported).
    static std::unique_ptr<MetalContext> create(std::string* error);
    ~MetalContext();
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    id<MTLDevice> device() const;
    id<MTL4CommandQueue> queue() const;
    id<MTL4Compiler> compiler() const;

    // Root for shader source paths: the project source directory (CMake injects
    // CLOUDNR_SOURCE_ROOT). All paths below are relative to it.
    const std::string& sourceRoot() const;

    // Builds one MSL source string: `defines` + src/shared_types.h +
    // src/shaders/common.metal + each file in `extraFiles`, in that order.
    // Lines starting with `#pragma once` are dropped. Project files never
    // #include each other; concatenation order provides declarations.
    std::string buildShaderSource(const std::vector<std::string>& extraFiles,
                                  const std::string& defines, std::string* error) const;

    // Runtime compile via MTL4Compiler (MTL4LibraryDescriptor.source ->
    // MTL4LibraryFunctionDescriptor -> MTL4ComputePipelineDescriptor).
    // Wall time is added to totalCompileMs(). Returns nil and sets *error on failure
    // (the error contains the compiler log).
    id<MTLComputePipelineState> makeComputePipeline(const std::vector<std::string>& extraFiles,
                                                    const std::string& function,
                                                    const std::string& defines,
                                                    std::string* error);
    double totalCompileMs() const;
    void addCompileMs(double ms);  // for pipelines built directly through compiler()

    // Residency: every renderer-owned buffer/texture must be added, then
    // commitResidency() called once before its first use.
    void addResident(id<MTLAllocation> allocation);
    void removeResident(id<MTLAllocation> allocation);
    void commitResidency();

    // Creates an argument table for compute (maxBufferBindCount/maxTextureBindCount).
    id<MTL4ArgumentTable> makeArgumentTable(uint32_t maxBuffers, uint32_t maxTextures,
                                            std::string* error);

    // ---- Frame loop (render loop, benchmarks, viewer) ----
    // Returns a command buffer already begun with the allocator of the next frame
    // slot. Blocks only if kFramesInFlight frames are still executing on the GPU
    // (waits on the shared event), then resets that slot's allocator.
    // slot() is valid until the matching endFrame and is in [0, kFramesInFlight).
    id<MTL4CommandBuffer> beginFrame();
    uint32_t currentSlot() const;
    uint64_t currentFrameIndex() const;  // monotonically increasing, first frame = 0

    // Called from a Metal-owned thread when the GPU finished the commit.
    // gpu_ms = (GPUEndTime - GPUStartTime) * 1000 from MTL4CommitFeedback.
    using FrameFeedback = std::function<void(double gpu_ms, NSError* error)>;

    // endCommandBuffer, commit:count:options: with a feedback handler, then
    // signalEvent: for the frame. Returns the frame index. Feedback errors are also
    // counted in gpuErrorCount().
    uint64_t endFrame(id<MTL4CommandBuffer> commandBuffer, FrameFeedback feedback);

    // CPU waits; never used inside a timed render loop.
    void waitForFrame(uint64_t frameIndex);
    void waitIdle();

    // Setup / readback helper: begins a frame, calls encode, ends it, waits.
    // Returns false and sets *error if the GPU reported an error.
    bool runSync(const std::function<void(id<MTL4CommandBuffer>)>& encode, double* gpu_ms,
                 std::string* error);

    uint64_t gpuErrorCount() const;

    // Debug-layer state observed at creation (MTL_DEBUG_LAYER / MTL_SHADER_VALIDATION).
    bool validationEnabled() const;

private:
    MetalContext();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// If the process was not started with MTL_DEBUG_LAYER=1 and MTL_SHADER_VALIDATION=1,
// re-executes the current binary with both set (they must be present before Metal
// initializes). Returns only if already enabled; exits on exec failure.
void ensureValidationEnvironment(int argc, char** argv);
