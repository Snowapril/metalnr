#include "neural_renderer.h"

#include "density_volume.h"
#include "metal_context.h"
#include "neural_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

namespace {

// Per frame slot: SceneGPU at offset 0, NeuralParams at kParamsOffset.
constexpr NSUInteger kParamsOffset = 256;
constexpr NSUInteger kSlotStride = 512;
static_assert(sizeof(SceneGPU) <= kParamsOffset, "SceneGPU must fit before the neural params");
static_assert(kParamsOffset + sizeof(NeuralParams) <= kSlotStride, "uniform slot too small");

class NeuralRenderer final : public Renderer {
public:
    ~NeuralRenderer() override {
        if (!context_) return;
        context_->waitIdle();
        for (id<MTLAllocation> allocation : resident_) context_->removeResident(allocation);
        context_->commitResidency();
    }

    bool init(MetalContext& context, const Scene& scene, const RenderConfig& config,
              const DensityVolume& density, std::string* error) {
        if (config.primary_steps == 0 || config.shadow_steps == 0) {
            if (error) *error = "neural renderer: primary_steps and shadow_steps must be > 0";
            return false;
        }
        id<MTLDevice> device = context.device();

        // Model load (file read, validation, hash, weight upload) is an excluded cost.
        const auto loadStart = std::chrono::steady_clock::now();
        std::optional<NeuralModel> model = loadNeuralModel(config.model, error);
        if (!model) return false;
        if (!validateNeuralModelForScene(*model, scene, error)) return false;
        weights_ = [device newBufferWithBytes:model->weights.data()
                                       length:model->weights.size() * sizeof(uint16_t)
                                      options:MTLResourceStorageModeShared];
        modelLoadMs_ =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loadStart).count();
        weightCount_ = model->weights.size();

        pipeline_ = context.makeComputePipeline(neuralShaderFiles("src/shaders/neural.metal"), "neural_render",
                                                neuralShaderDefines(model->dims), error);
        if (!pipeline_) return false;
        table_ = context.makeArgumentTable(3, 2, error);
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
        if (!weights_ || !uniforms_ || !output_) {
            if (error) *error = "neural renderer: allocation failed";
            return false;
        }
        weights_.label = @"neural weights";
        uniforms_.label = @"neural uniforms";
        output_.label = @"neural output";

        params_.primary_steps = config.primary_steps;
        params_.shadow_steps = config.shadow_steps;
        [table_ setAddress:weights_.gpuAddress atIndex:2];
        [table_ setTexture:density.texture().gpuResourceID atIndex:0];
        [table_ setTexture:output_.gpuResourceID atIndex:1];

        context_ = &context;
        resident_ = {weights_, uniforms_, output_};
        for (id<MTLAllocation> allocation : resident_) context.addResident(allocation);
        context.commitResidency();
        return true;
    }

    const char* name() const override { return "neural"; }

    void encode(id<MTL4ComputeCommandEncoder> encoder, const FrameInputs& in) override {
        const NSUInteger slotOffset = in.slot * kSlotStride;
        uint8_t* slot = static_cast<uint8_t*>(uniforms_.contents) + slotOffset;
        std::memcpy(slot, &in.scene, sizeof(SceneGPU));
        std::memcpy(slot + kParamsOffset, &params_, sizeof(NeuralParams));

        // The argument table is snapshotted at dispatch, so rebinding per frame is safe.
        [table_ setAddress:uniforms_.gpuAddress + slotOffset atIndex:0];
        [table_ setAddress:uniforms_.gpuAddress + slotOffset + kParamsOffset atIndex:1];
        [encoder setComputePipelineState:pipeline_];
        [encoder setArgumentTable:table_];
        [encoder dispatchThreads:MTLSizeMake(output_.width, output_.height, 1)
           threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    }

    id<MTLTexture> output() const override { return output_; }

    uint64_t memoryBytes() const override {
        return weights_.allocatedSize + uniforms_.allocatedSize + output_.allocatedSize;
    }

    std::vector<std::pair<std::string, double>> stats() const override {
        return {{"model_load_ms", modelLoadMs_}, {"model_weights", double(weightCount_)}};
    }

private:
    MetalContext* context_ = nullptr;
    id<MTLComputePipelineState> pipeline_ = nil;
    id<MTL4ArgumentTable> table_ = nil;
    id<MTLBuffer> weights_ = nil;
    id<MTLBuffer> uniforms_ = nil;
    id<MTLTexture> output_ = nil;
    std::vector<id<MTLAllocation>> resident_;
    NeuralParams params_{};
    double modelLoadMs_ = 0.0;
    size_t weightCount_ = 0;
};

std::string jsonEscape(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        if (static_cast<unsigned char>(c) < 0x20) continue;
        out += c;
    }
    return out + "\"";
}

}  // namespace

std::unique_ptr<Renderer> makeNeuralRenderer(MetalContext& context, const Scene& scene,
                                             const RenderConfig& config, const DensityVolume& density,
                                             std::string* error) {
    auto renderer = std::make_unique<NeuralRenderer>();
    if (!renderer->init(context, scene, config, density, error)) return nullptr;
    return renderer;
}

bool evalNeuralBatch(MetalContext& context, const NeuralModel& model, const std::vector<float>& inputs,
                     std::vector<float>* outputs, std::string* error) {
    const uint32_t nIn = model.dims.n_in;
    if (nIn == 0 || inputs.size() % nIn != 0) {
        if (error) *error = "evalNeuralBatch: input size is not a multiple of n_in";
        return false;
    }
    const size_t count = inputs.size() / nIn;
    outputs->assign(count * 3, 0.0f);
    if (count == 0) return true;

    id<MTLComputePipelineState> pipeline = context.makeComputePipeline(
        neuralShaderFiles("src/shaders/neural.metal"), "neural_eval_batch", neuralShaderDefines(model.dims), error);
    if (!pipeline) return false;
    id<MTL4ArgumentTable> table = context.makeArgumentTable(4, 1, error);
    if (!table) return false;

    id<MTLDevice> device = context.device();
    NeuralEvalParams params{};
    params.count = static_cast<uint32_t>(count);
    id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params
                                                     length:sizeof(params)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> weightBuffer = [device newBufferWithBytes:model.weights.data()
                                                     length:model.weights.size() * sizeof(uint16_t)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:inputs.data()
                                                    length:inputs.size() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:count * 3 * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    if (!paramsBuffer || !weightBuffer || !inputBuffer || !outputBuffer) {
        if (error) *error = "evalNeuralBatch: allocation failed";
        return false;
    }
    const std::vector<id<MTLAllocation>> resident = {paramsBuffer, weightBuffer, inputBuffer, outputBuffer};
    for (id<MTLAllocation> allocation : resident) context.addResident(allocation);
    context.commitResidency();

    [table setAddress:paramsBuffer.gpuAddress atIndex:0];
    [table setAddress:weightBuffer.gpuAddress atIndex:1];
    [table setAddress:inputBuffer.gpuAddress atIndex:2];
    [table setAddress:outputBuffer.gpuAddress atIndex:3];
    const bool ok = context.runSync(
        [&](id<MTL4CommandBuffer> commandBuffer) {
            id<MTL4ComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setArgumentTable:table];
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [encoder endEncoding];
        },
        nullptr, error);

    for (id<MTLAllocation> allocation : resident) context.removeResident(allocation);
    context.commitResidency();
    if (!ok) return false;
    std::memcpy(outputs->data(), outputBuffer.contents, count * 3 * sizeof(float));
    return true;
}

int cmdNeuralCheck(int argc, char** argv) {
    std::string modelPath;
    std::string checkPath;
    bool validate = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--validate") {
            validate = true;
        } else if ((arg == "--model" || arg == "--check") && i + 1 < argc) {
            (arg == "--model" ? modelPath : checkPath) = argv[++i];
        } else {
            std::fprintf(stderr, "cloudnr: neural-check: unknown or incomplete argument '%s'\n", arg.c_str());
            return 1;
        }
    }
    if (modelPath.empty() || checkPath.empty()) {
        std::fprintf(stderr, "cloudnr: neural-check: --model and --check are required\n");
        return 1;
    }
    if (validate) ensureValidationEnvironment(argc, argv);

    std::string error;
    std::optional<NeuralModel> model = loadNeuralModel(modelPath, &error);
    if (!model) {
        std::fprintf(stderr, "cloudnr: %s\n", error.c_str());
        return 1;
    }

    // Check file (tools/train.py): "CNRC", u32 count, u32 n_in, u32 n_out, then
    // float32 inputs[count][n_in] and torch outputs f[count][n_out].
    std::ifstream in(checkPath, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cloudnr: cannot open %s\n", checkPath.c_str());
        return 1;
    }
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    uint32_t header[3] = {};
    if (bytes.size() < 16 || std::memcmp(bytes.data(), "CNRC", 4) != 0) {
        std::fprintf(stderr, "cloudnr: %s: not a CNRC check file\n", checkPath.c_str());
        return 1;
    }
    std::memcpy(header, bytes.data() + 4, sizeof(header));
    const uint64_t count = header[0];
    const uint32_t nIn = header[1];
    const uint32_t nOut = header[2];
    if (nIn != model->dims.n_in || nOut != 3 ||
        bytes.size() != 16 + count * (uint64_t(nIn) + nOut) * sizeof(float)) {
        std::fprintf(stderr, "cloudnr: %s: dimensions or size do not match the model\n", checkPath.c_str());
        return 1;
    }
    std::vector<float> inputs(count * nIn);
    std::vector<float> expected(count * nOut);
    std::memcpy(inputs.data(), bytes.data() + 16, inputs.size() * sizeof(float));
    std::memcpy(expected.data(), bytes.data() + 16 + inputs.size() * sizeof(float), expected.size() * sizeof(float));

    auto context = MetalContext::create(&error);
    if (!context) {
        std::fprintf(stderr, "cloudnr: %s\n", error.c_str());
        return 1;
    }
    std::vector<float> outputs;
    if (!evalNeuralBatch(*context, *model, inputs, &outputs, &error)) {
        std::fprintf(stderr, "cloudnr: %s\n", error.c_str());
        return 1;
    }

    double maxRel = 0.0;
    double sumRel = 0.0;
    double maxAbs = 0.0;
    bool finite = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double a = outputs[i];
        const double b = expected[i];
        finite = finite && std::isfinite(a);
        const double rel = std::fabs(a - b) / (std::fabs(b) + 1e-3);
        maxRel = std::max(maxRel, rel);
        maxAbs = std::max(maxAbs, std::fabs(a - b));
        sumRel += rel;
    }
    const double meanRel = expected.empty() ? 0.0 : sumRel / double(expected.size());
    const bool pass = finite && maxRel <= 1e-2 && context->gpuErrorCount() == 0;
    std::printf("{\"model\":%s,\"sha256\":%s,\"count\":%llu,\"max_rel\":%.9g,\"mean_rel\":%.9g,\"max_abs\":%.9g,"
                "\"finite\":%s,\"gpu_errors\":%llu,\"pass\":%s}\n",
                jsonEscape(modelPath).c_str(), jsonEscape(model->sha256).c_str(),
                static_cast<unsigned long long>(count), maxRel, meanRel, maxAbs, finite ? "true" : "false",
                static_cast<unsigned long long>(context->gpuErrorCount()), pass ? "true" : "false");
    return pass ? 0 : 1;
}
