#include "metal_context.h"

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>

#include <mach-o/dyld.h>
#include <unistd.h>

#ifndef CLOUDNR_SOURCE_ROOT
#error "CLOUDNR_SOURCE_ROOT must be defined by the build"
#endif

namespace {

bool envIsOne(const char* name) {
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

std::string describe(NSError* error) {
    return error ? std::string(error.localizedDescription.UTF8String) : std::string("unknown error");
}

// waitUntilSignaledValue:timeoutMS: with a periodic timeout so a hung GPU is reported
// instead of blocking silently forever.
void waitForEventValue(id<MTLSharedEvent> event, uint64_t value) {
    while (![event waitUntilSignaledValue:value timeoutMS:10000]) {
        std::fprintf(stderr, "cloudnr: still waiting for GPU (event value %llu)\n",
                     static_cast<unsigned long long>(value));
    }
}

}  // namespace

struct MetalContext::Impl {
    id<MTLDevice> device = nil;
    id<MTL4CommandQueue> queue = nil;
    id<MTL4Compiler> compiler = nil;
    id<MTL4CommandAllocator> allocators[kFramesInFlight] = {};
    id<MTL4CommandBuffer> commandBuffers[kFramesInFlight] = {};
    id<MTLResidencySet> residency = nil;
    // Frame N signals N + 1 once its command buffer completed.
    id<MTLSharedEvent> frameEvent = nil;

    std::string sourceRoot = CLOUDNR_SOURCE_ROOT;
    double compileMs = 0.0;
    bool validation = false;

    uint64_t nextFrame = 0;
    uint64_t currentFrame = 0;
    uint32_t currentSlot = 0;

    std::atomic<uint64_t> gpuErrors{0};

    // Frames whose commit feedback handler has not run yet.
    std::mutex feedbackMutex;
    std::condition_variable feedbackCv;
    std::set<uint64_t> pendingFeedback;
};

MetalContext::MetalContext() : impl_(std::make_unique<Impl>()) {}

MetalContext::~MetalContext() {
    if (impl_ && impl_->queue) waitIdle();
}

std::unique_ptr<MetalContext> MetalContext::create(std::string* error) {
    std::unique_ptr<MetalContext> context(new MetalContext());
    Impl& impl = *context->impl_;
    impl.validation = envIsOne("MTL_DEBUG_LAYER") && envIsOne("MTL_SHADER_VALIDATION");

    impl.device = MTLCreateSystemDefaultDevice();
    if (!impl.device) {
        if (error) *error = "no Metal device";
        return nullptr;
    }
    if (![impl.device supportsFamily:MTLGPUFamilyMetal4]) {
        if (error) *error = std::string("device '") + impl.device.name.UTF8String +
                            "' does not support MTLGPUFamilyMetal4";
        return nullptr;
    }

    impl.queue = [impl.device newMTL4CommandQueue];
    if (!impl.queue) {
        if (error) *error = "newMTL4CommandQueue failed";
        return nullptr;
    }

    NSError* nsError = nil;
    MTL4CompilerDescriptor* compilerDesc = [MTL4CompilerDescriptor new];
    compilerDesc.label = @"cloudnr";
    impl.compiler = [impl.device newCompilerWithDescriptor:compilerDesc error:&nsError];
    if (!impl.compiler) {
        if (error) *error = "newCompilerWithDescriptor failed: " + describe(nsError);
        return nullptr;
    }

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        impl.allocators[i] = [impl.device newCommandAllocator];
        impl.commandBuffers[i] = [impl.device newCommandBuffer];
        if (!impl.allocators[i] || !impl.commandBuffers[i]) {
            if (error) *error = "newCommandAllocator/newCommandBuffer failed";
            return nullptr;
        }
    }

    MTLResidencySetDescriptor* residencyDesc = [MTLResidencySetDescriptor new];
    residencyDesc.label = @"cloudnr";
    residencyDesc.initialCapacity = 16;
    impl.residency = [impl.device newResidencySetWithDescriptor:residencyDesc error:&nsError];
    if (!impl.residency) {
        if (error) *error = "newResidencySetWithDescriptor failed: " + describe(nsError);
        return nullptr;
    }
    [impl.queue addResidencySet:impl.residency];

    impl.frameEvent = [impl.device newSharedEvent];
    if (!impl.frameEvent) {
        if (error) *error = "newSharedEvent failed";
        return nullptr;
    }
    impl.frameEvent.signaledValue = 0;
    return context;
}

id<MTLDevice> MetalContext::device() const { return impl_->device; }
id<MTL4CommandQueue> MetalContext::queue() const { return impl_->queue; }
id<MTL4Compiler> MetalContext::compiler() const { return impl_->compiler; }
const std::string& MetalContext::sourceRoot() const { return impl_->sourceRoot; }

std::string MetalContext::buildShaderSource(const std::vector<std::string>& extraFiles,
                                            const std::string& defines, std::string* error) const {
    std::vector<std::string> files = {"src/shared_types.h", "src/shaders/common.metal"};
    files.insert(files.end(), extraFiles.begin(), extraFiles.end());

    std::string source = defines;
    if (!source.empty() && source.back() != '\n') source += '\n';
    for (const std::string& file : files) {
        const std::string path = impl_->sourceRoot + "/" + file;
        std::ifstream in(path);
        if (!in) {
            if (error) *error = "cannot open shader source " + path;
            return {};
        }
        source += "// ---- " + file + " ----\n";
        std::string line;
        while (std::getline(in, line)) {
            if (line.rfind("#pragma once", 0) == 0) continue;
            source += line;
            source += '\n';
        }
    }
    return source;
}

id<MTLComputePipelineState> MetalContext::makeComputePipeline(const std::vector<std::string>& extraFiles,
                                                              const std::string& function,
                                                              const std::string& defines,
                                                              std::string* error) {
    const std::string source = buildShaderSource(extraFiles, defines, error);
    if (source.empty()) return nil;

    const auto start = std::chrono::steady_clock::now();

    MTL4LibraryDescriptor* libraryDesc = [MTL4LibraryDescriptor new];
    libraryDesc.source = [NSString stringWithUTF8String:source.c_str()];
    libraryDesc.name = [NSString stringWithUTF8String:function.c_str()];
    NSError* nsError = nil;
    id<MTLLibrary> library = [impl_->compiler newLibraryWithDescriptor:libraryDesc error:&nsError];
    if (!library) {
        if (error) *error = "shader compile failed for '" + function + "': " + describe(nsError);
        return nil;
    }
    // The debug layer asserts (instead of returning an error) when the function is missing.
    NSString* functionName = [NSString stringWithUTF8String:function.c_str()];
    if (![library.functionNames containsObject:functionName]) {
        addCompileMs(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        if (error) *error = "shader function '" + function + "' not found in compiled library";
        return nil;
    }


    MTL4LibraryFunctionDescriptor* functionDesc = [MTL4LibraryFunctionDescriptor new];
    functionDesc.name = functionName;
    functionDesc.library = library;

    MTL4ComputePipelineDescriptor* pipelineDesc = [MTL4ComputePipelineDescriptor new];
    pipelineDesc.computeFunctionDescriptor = functionDesc;
    pipelineDesc.label = libraryDesc.name;

    id<MTLComputePipelineState> pipeline =
        [impl_->compiler newComputePipelineStateWithDescriptor:pipelineDesc
                                           compilerTaskOptions:nil
                                                         error:&nsError];
    addCompileMs(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    if (!pipeline) {
        if (error) *error = "pipeline creation failed for '" + function + "': " + describe(nsError);
        return nil;
    }
    return pipeline;
}

double MetalContext::totalCompileMs() const { return impl_->compileMs; }
void MetalContext::addCompileMs(double ms) { impl_->compileMs += ms; }

void MetalContext::addResident(id<MTLAllocation> allocation) { [impl_->residency addAllocation:allocation]; }
void MetalContext::removeResident(id<MTLAllocation> allocation) { [impl_->residency removeAllocation:allocation]; }
void MetalContext::commitResidency() { [impl_->residency commit]; }

id<MTL4ArgumentTable> MetalContext::makeArgumentTable(uint32_t maxBuffers, uint32_t maxTextures,
                                                      std::string* error) {
    MTL4ArgumentTableDescriptor* desc = [MTL4ArgumentTableDescriptor new];
    desc.maxBufferBindCount = maxBuffers;
    desc.maxTextureBindCount = maxTextures;
    desc.initializeBindings = YES;
    NSError* nsError = nil;
    id<MTL4ArgumentTable> table = [impl_->device newArgumentTableWithDescriptor:desc error:&nsError];
    if (!table && error) *error = "newArgumentTableWithDescriptor failed: " + describe(nsError);
    return table;
}

id<MTL4CommandBuffer> MetalContext::beginFrame() {
    Impl& impl = *impl_;
    const uint64_t frame = impl.nextFrame++;
    const uint32_t slot = static_cast<uint32_t>(frame % kFramesInFlight);
    // Frame (frame - kFramesInFlight) used this slot; it signals frame - kFramesInFlight + 1.
    if (frame >= kFramesInFlight) waitForEventValue(impl.frameEvent, frame - kFramesInFlight + 1);
    impl.currentFrame = frame;
    impl.currentSlot = slot;
    [impl.allocators[slot] reset];
    id<MTL4CommandBuffer> commandBuffer = impl.commandBuffers[slot];
    [commandBuffer beginCommandBufferWithAllocator:impl.allocators[slot]];
    return commandBuffer;
}

uint32_t MetalContext::currentSlot() const { return impl_->currentSlot; }
uint64_t MetalContext::currentFrameIndex() const { return impl_->currentFrame; }

uint64_t MetalContext::endFrame(id<MTL4CommandBuffer> commandBuffer, FrameFeedback feedback) {
    Impl* impl = impl_.get();
    const uint64_t frame = impl->currentFrame;
    [commandBuffer endCommandBuffer];

    {
        std::lock_guard<std::mutex> lock(impl->feedbackMutex);
        impl->pendingFeedback.insert(frame);
    }

    MTL4CommitOptions* options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> commitFeedback) {
        const double gpuMs = (commitFeedback.GPUEndTime - commitFeedback.GPUStartTime) * 1000.0;
        NSError* gpuError = commitFeedback.error;
        if (gpuError) {
            impl->gpuErrors.fetch_add(1);
            std::fprintf(stderr, "cloudnr: GPU error in frame %llu: %s\n",
                         static_cast<unsigned long long>(frame), describe(gpuError).c_str());
        }
        if (feedback) feedback(gpuMs, gpuError);
        {
            std::lock_guard<std::mutex> lock(impl->feedbackMutex);
            impl->pendingFeedback.erase(frame);
        }
        impl->feedbackCv.notify_all();
    }];

    id<MTL4CommandBuffer> buffers[1] = {commandBuffer};
    [impl->queue commit:buffers count:1 options:options];
    [impl->queue signalEvent:impl->frameEvent value:frame + 1];
    return frame;
}

void MetalContext::waitForFrame(uint64_t frameIndex) {
    Impl& impl = *impl_;
    waitForEventValue(impl.frameEvent, frameIndex + 1);
    // The feedback handler may run after the event signal; callers rely on its results.
    std::unique_lock<std::mutex> lock(impl.feedbackMutex);
    impl.feedbackCv.wait(lock, [&] {
        return impl.pendingFeedback.empty() || *impl.pendingFeedback.begin() > frameIndex;
    });
}

void MetalContext::waitIdle() {
    if (impl_->nextFrame > 0) waitForFrame(impl_->nextFrame - 1);
}

bool MetalContext::runSync(const std::function<void(id<MTL4CommandBuffer>)>& encode, double* gpu_ms,
                           std::string* error) {
    id<MTL4CommandBuffer> commandBuffer = beginFrame();
    encode(commandBuffer);
    double frameMs = 0.0;
    std::string frameError;
    const uint64_t frame = endFrame(commandBuffer, [&frameMs, &frameError](double ms, NSError* e) {
        frameMs = ms;
        if (e) frameError = describe(e);
    });
    waitForFrame(frame);
    if (gpu_ms) *gpu_ms = frameMs;
    if (!frameError.empty()) {
        if (error) *error = "GPU error: " + frameError;
        return false;
    }
    return true;
}

uint64_t MetalContext::gpuErrorCount() const { return impl_->gpuErrors.load(); }
bool MetalContext::validationEnabled() const { return impl_->validation; }

void ensureValidationEnvironment(int argc, char** argv) {
    (void)argc;
    if (envIsOne("MTL_DEBUG_LAYER") && envIsOne("MTL_SHADER_VALIDATION")) return;
    setenv("MTL_DEBUG_LAYER", "1", 1);
    setenv("MTL_SHADER_VALIDATION", "1", 1);

    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> path(size + 1, '\0');
    if (_NSGetExecutablePath(path.data(), &size) != 0) {
        std::fprintf(stderr, "cloudnr: cannot resolve executable path for --validate\n");
        std::exit(1);
    }
    execv(path.data(), argv);
    std::fprintf(stderr, "cloudnr: execv failed for --validate: %s\n", std::strerror(errno));
    std::exit(1);
}
