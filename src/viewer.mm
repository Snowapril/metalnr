// Interactive viewer: AppKit window + CAMetalLayer, Metal 4 queue only.
// Each display-link frame: renderer compute pass, then a tone-mapping render pass
// (src/shaders/display.metal) into the drawable.

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include "viewer.h"

#include "density_volume.h"
#include "image_io.h"
#include "metal_context.h"
#include "renderer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

namespace {

constexpr float kMaxPitch = 89.0f * 3.14159265358979f / 180.0f;
constexpr float kOrbitRadPerPoint = 0.005f;
constexpr float kPanFraction = 0.05f;   // of the target distance, per key press
constexpr float kMinDistance = 1.0f;    // m
constexpr float kExposureStep = 0.5f;   // EV
constexpr NSUInteger kExposureStride = 256;  // per-slot offset in the exposure buffer
constexpr CFTimeInterval kTitleInterval = 0.25;
constexpr float kScrollPerPreciseUnit = 0.01f;  // trackpad points
constexpr float kScrollPerNotch = 0.1f;         // mouse wheel lines
const char* const kRendererKeys[4] = {"density", "analytic", "pt", "neural"};

// Written from the Metal feedback thread; shared so late callbacks stay valid.
struct FrameStats {
    std::atomic<double> gpuMs{0.0};
    std::atomic<bool> errorReported{false};
};

struct Viewer {
    Viewer(MetalContext& c, const Scene& s, const CameraDesc& cam, const RenderConfig& cfg)
        : context(c), scene(s), initialCamera(cam), camera(cam), config(cfg) {}

    MetalContext& context;
    const Scene& scene;
    const CameraDesc initialCamera;
    CameraDesc camera;
    RenderConfig config;
    std::unique_ptr<DensityVolume> volume;
    std::unique_ptr<Renderer> renderer;
    id<MTLRenderPipelineState> displayPipeline = nil;
    id<MTL4ArgumentTable> displayTable = nil;
    id<MTLBuffer> exposureBuffer = nil;  // kFramesInFlight slots of one float
    MTL4RenderPassDescriptor* displayPass = nil;
    CAMetalLayer* layer = nil;
    NSWindow* window = nil;
    CADisplayLink* displayLink = nil;
    std::shared_ptr<FrameStats> stats = std::make_shared<FrameStats>();
    CFTimeInterval lastTitleTime = 0.0;
    // Autotest (empty prefix = interactive only).
    std::string autotestPrefix;
    id<MTLBuffer> captureBuffer = nil;  // BGRA8 drawable copy, width*height*4 bytes
    uint64_t presentedFrames = 0;
    bool autotestFinished = false;
    std::string autotestError;
};

std::string describe(NSError* error)
{
    return error ? std::string(error.localizedDescription.UTF8String) : std::string("unknown error");
}

id<MTLRenderPipelineState> makeDisplayPipeline(MetalContext& context, std::string* error)
{
    std::string sourceError;
    std::string source = context.buildShaderSource({"src/shaders/display.metal"}, "", &sourceError);
    if (source.empty()) {
        *error = "display shader source: " + sourceError;
        return nil;
    }
    auto t0 = std::chrono::steady_clock::now();
    MTL4LibraryDescriptor* libraryDesc = [MTL4LibraryDescriptor new];
    libraryDesc.source = [NSString stringWithUTF8String:source.c_str()];
    libraryDesc.name = @"display";
    NSError* nsError = nil;
    id<MTLLibrary> library = [context.compiler() newLibraryWithDescriptor:libraryDesc error:&nsError];
    if (!library) {
        *error = "display shader compile failed: " + describe(nsError);
        return nil;
    }
    MTL4LibraryFunctionDescriptor* vs = [MTL4LibraryFunctionDescriptor new];
    vs.name = @"display_vs";
    vs.library = library;
    MTL4LibraryFunctionDescriptor* fs = [MTL4LibraryFunctionDescriptor new];
    fs.name = @"display_fs";
    fs.library = library;
    MTL4RenderPipelineDescriptor* desc = [MTL4RenderPipelineDescriptor new];
    desc.vertexFunctionDescriptor = vs;
    desc.fragmentFunctionDescriptor = fs;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    id<MTLRenderPipelineState> pipeline =
        [context.compiler() newRenderPipelineStateWithDescriptor:desc compilerTaskOptions:nil error:&nsError];
    context.addCompileMs(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    if (!pipeline) {
        *error = "display pipeline failed: " + describe(nsError);
        return nil;
    }
    return pipeline;
}

void cameraChanged(Viewer& v)
{
    v.renderer->resetAccumulation();
}

// Orbit and zoom around the target; position is recomputed from spherical coordinates
// only when the user moves the camera, so the initial frames use the scene camera exactly.
void orbitCamera(Viewer& v, float dYaw, float dPitch, float distanceScale)
{
    CameraDesc& cam = v.camera;
    simd_float3 offset = cam.position - cam.target;
    float distance = std::max(simd_length(offset), 1e-3f);
    float pitch = std::asin(std::clamp(offset.y / distance, -1.0f, 1.0f)) + dPitch;
    float yaw = std::atan2(offset.x, offset.z) + dYaw;
    pitch = std::clamp(pitch, -kMaxPitch, kMaxPitch);
    distance = std::max(distance * distanceScale, kMinDistance);
    cam.position = cam.target + distance * simd_make_float3(std::cos(pitch) * std::sin(yaw), std::sin(pitch),
                                                            std::cos(pitch) * std::cos(yaw));
    cameraChanged(v);
}

// Moves target and position together: `right`/`up` in the view plane, `vertical` along world +Y.
void panCamera(Viewer& v, float right, float up, float vertical)
{
    CameraDesc& cam = v.camera;
    simd_float3 toTarget = cam.target - cam.position;
    simd_float3 forward = simd_normalize(toTarget);
    simd_float3 r = simd_normalize(simd_cross(forward, simd_make_float3(0.0f, 1.0f, 0.0f)));
    simd_float3 u = simd_cross(r, forward);
    simd_float3 delta = kPanFraction * simd_length(toTarget) *
                        (right * r + up * u + vertical * simd_make_float3(0.0f, 1.0f, 0.0f));
    cam.position += delta;
    cam.target += delta;
    cameraChanged(v);
}

void switchRenderer(Viewer& v, const char* name)
{
    if (v.config.renderer == name) return;
    RenderConfig next = v.config;
    next.renderer = name;
    v.context.waitIdle();
    std::string err;
    std::unique_ptr<Renderer> renderer = makeRenderer(v.context, v.scene, next, *v.volume, &err);
    if (!renderer) {
        fprintf(stderr, "cloudnr view: cannot switch to renderer '%s': %s\n", name, err.c_str());
        return;
    }
    v.renderer = std::move(renderer);  // old renderer destroyed after waitIdle
    v.config = next;
    v.stats->gpuMs.store(0.0);
}

bool handleKey(Viewer& v, unichar key)
{
    switch (key) {
        case 'w': panCamera(v, 0, 1, 0); return true;
        case 's': panCamera(v, 0, -1, 0); return true;
        case 'a': panCamera(v, -1, 0, 0); return true;
        case 'd': panCamera(v, 1, 0, 0); return true;
        case 'q': panCamera(v, 0, 0, -1); return true;
        case 'e': panCamera(v, 0, 0, 1); return true;
        case '[': v.camera.exposure_ev -= kExposureStep; return true;
        case ']': v.camera.exposure_ev += kExposureStep; return true;
        case 'r':
            v.camera = v.initialCamera;
            cameraChanged(v);
            return true;
        case '1': case '2': case '3': case '4':
            switchRenderer(v, kRendererKeys[key - '1']);
            return true;
        default: return false;
    }
}

// Input handlers shared by NSView events and the autotest script.
void onDrag(Viewer& v, float dx, float dy)
{
    orbitCamera(v, -dx * kOrbitRadPerPoint, dy * kOrbitRadPerPoint, 1.0f);
}

void onScroll(Viewer& v, float deltaY, bool precise)
{
    orbitCamera(v, 0.0f, 0.0f, std::exp(-deltaY * (precise ? kScrollPerPreciseUnit : kScrollPerNotch)));
}

void onMagnify(Viewer& v, float magnification)
{
    orbitCamera(v, 0.0f, 0.0f, 1.0f / std::max(1.0f + magnification, 0.1f));
}

// Step k runs its action at presented frame kAutotestStart + k*kAutotestPeriod and captures
// the frame kAutotestPeriod - 1 later (lets progressive renderers accumulate a few samples).
// After the last step the window closes through performClose: (windowWillClose path).
constexpr uint64_t kAutotestStart = 10;
constexpr uint64_t kAutotestPeriod = 10;
struct AutotestStep {
    const char* name;
    void (*action)(Viewer&);
};
const AutotestStep kAutotestSteps[] = {
    {"initial", [](Viewer&) {}},
    {"orbit", [](Viewer& v) { onDrag(v, 200.0f, 0.0f); }},
    {"zoom", [](Viewer& v) { onScroll(v, 3.0f, false); }},  // three wheel notches toward the target
    {"pan", [](Viewer& v) { for (int i = 0; i < 5; ++i) handleKey(v, 'd'); }},
    {"exposure", [](Viewer& v) { handleKey(v, ']'); handleKey(v, ']'); }},
    {"reset", [](Viewer& v) { handleKey(v, 'r'); }},
};
constexpr uint64_t kAutotestStepCount = sizeof(kAutotestSteps) / sizeof(kAutotestSteps[0]);

// Runs the script for the frame about to be encoded. Returns the capture name for this
// frame (or nullptr); sets *close when the script has finished.
const char* runAutotest(Viewer& v, bool* close)
{
    *close = false;
    const uint64_t f = v.presentedFrames;
    if (v.autotestPrefix.empty() || f < kAutotestStart) return nullptr;
    const uint64_t step = (f - kAutotestStart) / kAutotestPeriod;
    const uint64_t phase = (f - kAutotestStart) % kAutotestPeriod;
    if (step >= kAutotestStepCount) {
        v.autotestFinished = true;
        *close = true;
        return nullptr;
    }
    if (phase == 0) kAutotestSteps[step].action(v);
    return phase == kAutotestPeriod - 1 ? kAutotestSteps[step].name : nullptr;
}

// Converts the completed BGRA8 capture to display RGB, writes <prefix>_<name>.png, prints the camera.
void writeCapture(Viewer& v, const char* name)
{
    Image image;
    image.width = v.scene.width;
    image.height = v.scene.height;
    image.rgb.resize(size_t(image.width) * image.height * 3);
    const uint8_t* bgra = static_cast<const uint8_t*>(v.captureBuffer.contents);
    for (size_t i = 0, n = size_t(image.width) * image.height; i < n; ++i) {
        image.rgb[i * 3 + 0] = bgra[i * 4 + 2] / 255.0f;
        image.rgb[i * 3 + 1] = bgra[i * 4 + 1] / 255.0f;
        image.rgb[i * 3 + 2] = bgra[i * 4 + 0] / 255.0f;
    }
    std::string err;
    const std::string path = v.autotestPrefix + "_" + name + ".png";
    if (!writeDisplayPNG(image, path, &err) && v.autotestError.empty())
        v.autotestError = "autotest: cannot write " + path + ": " + err;
    const CameraDesc& c = v.camera;
    printf("autotest %s camera=%g,%g,%g target=%g,%g,%g ev=%g\n", name, c.position.x, c.position.y,
           c.position.z, c.target.x, c.target.y, c.target.z, c.exposure_ev);
    fflush(stdout);
}

void drawFrame(Viewer& v)
{
    // Render resolution stays fixed; the layer scales the drawable to the window.
    CGSize renderSize = CGSizeMake(v.scene.width, v.scene.height);
    if (!CGSizeEqualToSize(v.layer.drawableSize, renderSize)) v.layer.drawableSize = renderSize;

    id<CAMetalDrawable> drawable = [v.layer nextDrawable];
    if (!drawable) return;

    bool close = false;
    const char* captureName = runAutotest(v, &close);
    if (close) {
        [v.window performClose:nil];
        return;
    }

    id<MTL4CommandBuffer> commandBuffer = v.context.beginFrame();
    uint32_t slot = v.context.currentSlot();
    auto* exposure = reinterpret_cast<float*>(static_cast<uint8_t*>(v.exposureBuffer.contents) +
                                              slot * kExposureStride);
    *exposure = v.camera.exposure_ev;

    id<MTL4ComputeCommandEncoder> compute = [commandBuffer computeCommandEncoder];
    // The previous frame's display pass may still be reading the renderer output.
    [compute barrierAfterQueueStages:MTLStageFragment
                        beforeStages:MTLStageDispatch
                   visibilityOptions:MTL4VisibilityOptionNone];
    v.renderer->encode(compute, FrameInputs{makeSceneGPU(v.scene, v.camera), slot});
    [compute endEncoding];

    v.displayPass.colorAttachments[0].texture = drawable.texture;
    id<MTL4RenderCommandEncoder> render = [commandBuffer renderCommandEncoderWithDescriptor:v.displayPass];
    v.displayPass.colorAttachments[0].texture = nil;
    // Fragment reads of the output wait for this frame's compute writes.
    [render barrierAfterQueueStages:MTLStageDispatch
                       beforeStages:MTLStageFragment
                  visibilityOptions:MTL4VisibilityOptionDevice];
    [v.displayTable setTexture:v.renderer->output().gpuResourceID atIndex:0];
    [v.displayTable setAddress:v.exposureBuffer.gpuAddress + slot * kExposureStride atIndex:0];
    [render setRenderPipelineState:v.displayPipeline];
    [render setArgumentTable:v.displayTable atStages:MTLRenderStageFragment];
    [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [render endEncoding];

    if (captureName) {
        // Copy the presented image (framebufferOnly is NO in autotest mode) after the display pass.
        id<MTL4ComputeCommandEncoder> copy = [commandBuffer computeCommandEncoder];
        [copy barrierAfterQueueStages:MTLStageFragment
                         beforeStages:MTLStageBlit
                    visibilityOptions:MTL4VisibilityOptionDevice];
        const NSUInteger w = v.scene.width, h = v.scene.height;
        [copy copyFromTexture:drawable.texture
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:MTLSizeMake(w, h, 1)
                            toBuffer:v.captureBuffer
                   destinationOffset:0
              destinationBytesPerRow:w * 4
            destinationBytesPerImage:w * h * 4];
        [copy endEncoding];
    }

    // MTL4CommandQueue.h: waitForDrawable before committing work that targets the drawable,
    // signalDrawable after that commit and before present.
    id<MTL4CommandQueue> queue = v.context.queue();
    [queue waitForDrawable:drawable];
    std::shared_ptr<FrameStats> stats = v.stats;
    const uint64_t frameIndex = v.context.endFrame(commandBuffer, [stats](double gpuMs, NSError* error) {
        if (error) {
            if (!stats->errorReported.exchange(true))
                fprintf(stderr, "cloudnr view: GPU error: %s\n", error.localizedDescription.UTF8String);
            return;
        }
        stats->gpuMs.store(gpuMs, std::memory_order_relaxed);
    });
    [queue signalDrawable:drawable];
    [drawable present];
    if (captureName) {
        v.context.waitForFrame(frameIndex);
        writeCapture(v, captureName);
    }
    ++v.presentedFrames;

    CFTimeInterval now = CACurrentMediaTime();
    if (now - v.lastTitleTime >= kTitleInterval) {
        v.lastTitleTime = now;
        char title[256];
        snprintf(title, sizeof title, "cloudnr \u2014 %s \u2014 %.2f ms \u2014 %u spp", v.renderer->name(),
                 v.stats->gpuMs.load(std::memory_order_relaxed), v.renderer->accumulatedSamples());
        v.window.title = [NSString stringWithUTF8String:title];
    }
}

}  // namespace

@interface CnrMetalView : NSView
@property(nonatomic) Viewer* viewer;
@end

@implementation CnrMetalView

- (CALayer*)makeBackingLayer
{
    return [CAMetalLayer layer];
}

- (BOOL)acceptsFirstResponder
{
    return YES;
}

- (BOOL)acceptsFirstMouse:(NSEvent*)event
{
    return YES;
}

- (void)mouseDragged:(NSEvent*)event
{
    onDrag(*_viewer, float(event.deltaX), float(event.deltaY));
}

- (void)scrollWheel:(NSEvent*)event
{
    onScroll(*_viewer, float(event.scrollingDeltaY), event.hasPreciseScrollingDeltas);
}

- (void)magnifyWithEvent:(NSEvent*)event
{
    onMagnify(*_viewer, float(event.magnification));
}

- (void)keyDown:(NSEvent*)event
{
    NSString* chars = event.charactersIgnoringModifiers;
    unichar key = chars.length ? [chars characterAtIndex:0] : 0;
    if (key >= 'A' && key <= 'Z') key = unichar(key - 'A' + 'a');
    if (!handleKey(*_viewer, key)) [super keyDown:event];
}

@end

@interface CnrViewerDelegate : NSObject <NSWindowDelegate>
@property(nonatomic) Viewer* viewer;
- (void)onDisplayLink:(CADisplayLink*)link;
@end

@implementation CnrViewerDelegate

- (void)onDisplayLink:(CADisplayLink*)link
{
    @autoreleasepool {
        drawFrame(*_viewer);
    }
}

- (void)windowWillClose:(NSNotification*)notification
{
    [_viewer->displayLink invalidate];
    _viewer->displayLink = nil;
    [NSApp stop:nil];
    // -stop: takes effect after the next event; post one so -run returns now.
    NSEvent* wake = [NSEvent otherEventWithType:NSEventTypeApplicationDefined
                                       location:NSZeroPoint
                                  modifierFlags:0
                                      timestamp:0
                                   windowNumber:0
                                        context:nil
                                        subtype:0
                                          data1:0
                                          data2:0];
    [NSApp postEvent:wake atStart:YES];
}

@end

int runViewer(MetalContext& context, const Scene& scene, const std::string& cameraName,
              const RenderConfig& config, std::string* error, const std::string& autotestPrefix)
{
    auto fail = [error](const std::string& message) {
        if (error) *error = message;
        return 1;
    };
    const CameraDesc* camera = scene.camera(cameraName);
    if (!camera) return fail("unknown camera '" + cameraName + "'");

    @autoreleasepool {
        const uint64_t errorsAtStart = context.gpuErrorCount();
        Viewer v(context, scene, *camera, config);
        v.autotestPrefix = autotestPrefix;
        const bool autotest = !autotestPrefix.empty();
        std::string err;
        v.volume = DensityVolume::bake(context, scene, &err);
        if (!v.volume) return fail("density bake failed: " + err);
        v.renderer = makeRenderer(context, scene, config, *v.volume, &err);
        if (!v.renderer) return fail(err);
        v.displayPipeline = makeDisplayPipeline(context, &err);
        if (!v.displayPipeline) return fail(err);
        v.displayTable = context.makeArgumentTable(1, 1, &err);
        if (!v.displayTable) return fail(err);
        v.displayPass = [MTL4RenderPassDescriptor new];
        v.displayPass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        v.displayPass.colorAttachments[0].storeAction = MTLStoreActionStore;

        NSApplication* app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        NSRect frame = NSMakeRect(0, 0, scene.width, scene.height);
        NSWindow* window = [[NSWindow alloc]
            initWithContentRect:frame
                      styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable
                        backing:NSBackingStoreBuffered
                          defer:NO];
        window.releasedWhenClosed = NO;
        CnrMetalView* view = [[CnrMetalView alloc] initWithFrame:frame];
        view.viewer = &v;
        view.wantsLayer = YES;
        if (![view.layer isKindOfClass:[CAMetalLayer class]]) return fail("view is not backed by a CAMetalLayer");
        CAMetalLayer* layer = (CAMetalLayer*)view.layer;
        layer.device = context.device();
        layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        layer.framebufferOnly = !autotest;  // autotest copies the drawable texture
        layer.drawableSize = CGSizeMake(scene.width, scene.height);
        // AppKit owns the backing layer's contentsGravity; aspect-fit the fixed-size drawable.
        view.layerContentsPlacement = NSViewLayerContentsPlacementScaleProportionallyToFit;
        v.layer = layer;
        v.window = window;
        id<MTLResidencySet> layerResidency = layer.residencySet;
        if (!layerResidency) return fail("CAMetalLayer provides no residency set");

        v.exposureBuffer = [context.device() newBufferWithLength:kExposureStride * MetalContext::kFramesInFlight
                                                         options:MTLResourceStorageModeShared];
        if (!v.exposureBuffer) return fail("cannot allocate the exposure buffer");
        context.addResident(v.exposureBuffer);
        if (autotest) {
            v.captureBuffer = [context.device() newBufferWithLength:NSUInteger(scene.width) * scene.height * 4
                                                            options:MTLResourceStorageModeShared];
            if (!v.captureBuffer) return fail("cannot allocate the autotest capture buffer");
            context.addResident(v.captureBuffer);
        }
        context.commitResidency();
        [context.queue() addResidencySet:layerResidency];

        CnrViewerDelegate* delegate = [CnrViewerDelegate new];
        delegate.viewer = &v;
        window.delegate = delegate;
        window.contentView = view;
        window.title = @"cloudnr";
        [window center];
        [window makeFirstResponder:view];
        [window makeKeyAndOrderFront:nil];

        v.displayLink = [view displayLinkWithTarget:delegate selector:@selector(onDisplayLink:)];
        [v.displayLink addToRunLoop:[NSRunLoop currentRunLoop] forMode:NSRunLoopCommonModes];
        [app activate];
        [app run];

        context.waitIdle();
        window.delegate = nil;
        [context.queue() removeResidencySet:layerResidency];
        context.removeResident(v.exposureBuffer);
        if (v.captureBuffer) context.removeResident(v.captureBuffer);
        context.commitResidency();

        const uint64_t gpuErrors = context.gpuErrorCount() - errorsAtStart;
        if (autotest) {
            printf("autotest done gpu_errors=%llu\n", static_cast<unsigned long long>(gpuErrors));
            fflush(stdout);
        }
        if (gpuErrors) return fail(std::to_string(gpuErrors) + " GPU error(s) reported during viewing");
        if (autotest && !v.autotestError.empty()) return fail(v.autotestError);
        if (autotest && !v.autotestFinished) return fail("autotest: window closed before the script finished");
        return 0;
    }
}
