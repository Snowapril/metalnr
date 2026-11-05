#include "neural_renderer.h"

#include "density_volume.h"
#include "metal_context.h"
#include "neural_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

bool fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

bool parseInference(const std::string& name, NeuralInference* out, std::string* error) {
    if (name == "scalar") {
        *out = NeuralInference::Scalar;
        return true;
    }
    if (name == "shader_ml") {
        *out = NeuralInference::ShaderML;
        return true;
    }
    if (name == "ml_encoder") return fail(error, "ml_encoder inference is provided by the ML encoder module");
    return fail(error, "neural renderer: unknown inference '" + name + "' (expected scalar, shader_ml or ml_encoder)");
}

// The weight buffers of one variant: fp32 (always; biases for shader_ml) and fp16 matmul2d matrices.
struct WeightBuffers {
    id<MTLBuffer> weights = nil;
    id<MTLBuffer> weights16 = nil;  // shader_ml only
};

bool makeWeightBuffers(id<MTLDevice> device, const NeuralModel& model, NeuralInference inference,
                       WeightBuffers* out, std::string* error) {
    out->weights = [device newBufferWithBytes:model.weights.data()
                                       length:model.weights.size() * sizeof(float)
                                      options:MTLResourceStorageModeShared];
    if (!out->weights) return fail(error, "neural: weight buffer allocation failed");
    out->weights.label = @"neural weights fp32";
    if (inference == NeuralInference::ShaderML) {
        out->weights16 = [device newBufferWithBytes:model.weights_ml.data()
                                             length:model.weights_ml.size() * sizeof(uint16_t)
                                            options:MTLResourceStorageModeShared];
        if (!out->weights16) return fail(error, "neural: fp16 weight buffer allocation failed");
        out->weights16.label = @"neural weights fp16 [K][N]";
    }
    return true;
}

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
        if (!parseInference(config.inference, &inference_, error)) return false;
        if (!(config.step_scale > 0.0f) || config.max_steps == 0)
            return fail(error, "neural renderer: step_scale and max_steps must be > 0");
        id<MTLDevice> device = context.device();

        // Model load (file read, validation, hash, weight upload) is an excluded cost.
        const auto loadStart = std::chrono::steady_clock::now();
        std::optional<NeuralModel> model = loadNeuralModel(config.model, error);
        if (!model) return false;
        if (!validateNeuralModelForScene(*model, scene, error)) return false;
        if (!makeWeightBuffers(device, *model, inference_, &weights_, error)) return false;
        modelLoadMs_ =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loadStart).count();
        weightCount_ = model->weights.size();

        pipeline_ = context.makeComputePipeline(neuralShaderFiles("src/shaders/neural.metal"), "neural_render",
                                                neuralShaderDefines(model->dims, inference_), error);
        if (!pipeline_) return false;
        table_ = context.makeArgumentTable(4, 2, error);
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
        if (!uniforms_ || !output_) return fail(error, "neural renderer: allocation failed");
        uniforms_.label = @"neural uniforms";
        output_.label = @"neural output";

        params_.march = makeMarchParams(config);
        params_.norm = neuralStandardization(*model);
        [table_ setAddress:weights_.weights.gpuAddress atIndex:2];
        if (weights_.weights16) [table_ setAddress:weights_.weights16.gpuAddress atIndex:3];
        [table_ setTexture:density.texture().gpuResourceID atIndex:0];
        [table_ setTexture:output_.gpuResourceID atIndex:1];

        context_ = &context;
        resident_ = {weights_.weights, uniforms_, output_};
        if (weights_.weights16) resident_.push_back(weights_.weights16);
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
        uint64_t bytes = weights_.weights.allocatedSize + uniforms_.allocatedSize + output_.allocatedSize;
        if (weights_.weights16) bytes += weights_.weights16.allocatedSize;
        return bytes;
    }

    std::vector<std::pair<std::string, double>> stats() const override {
        return {{"model_load_ms", modelLoadMs_},
                {"model_weights", double(weightCount_)},
                {"shader_ml", inference_ == NeuralInference::ShaderML ? 1.0 : 0.0}};
    }

private:
    MetalContext* context_ = nullptr;
    NeuralInference inference_ = NeuralInference::Scalar;
    id<MTLComputePipelineState> pipeline_ = nil;
    id<MTL4ArgumentTable> table_ = nil;
    WeightBuffers weights_;
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

struct ParityStats {
    double maxRel = 0.0;
    double meanRel = 0.0;
    double maxAbs = 0.0;
    bool finite = true;
};

ParityStats compareOutputs(const std::vector<float>& actual, const std::vector<float>& expected) {
    ParityStats st;
    double sumRel = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double a = actual[i];
        const double b = expected[i];
        st.finite = st.finite && std::isfinite(a);
        const double rel = std::fabs(a - b) / (std::fabs(b) + 1e-3);
        st.maxRel = std::max(st.maxRel, rel);
        st.maxAbs = std::max(st.maxAbs, std::fabs(a - b));
        sumRel += rel;
    }
    st.meanRel = expected.empty() ? 0.0 : sumRel / double(expected.size());
    return st;
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
                     std::vector<float>* outputs, NeuralInference inference, std::string* error) {
    const uint32_t nIn = model.dims.n_in;
    if (nIn == 0 || inputs.size() % nIn != 0)
        return fail(error, "evalNeuralBatch: input size is not a multiple of n_in");
    const size_t count = inputs.size() / nIn;
    if (count > UINT32_MAX) return fail(error, "evalNeuralBatch: too many inputs");
    outputs->assign(count * 3, 0.0f);
    if (count == 0) return true;

    id<MTLComputePipelineState> pipeline =
        context.makeComputePipeline(neuralShaderFiles("src/shaders/neural.metal"), "neural_eval_batch",
                                    neuralShaderDefines(model.dims, inference), error);
    if (!pipeline) return false;
    id<MTL4ArgumentTable> table = context.makeArgumentTable(5, 1, error);
    if (!table) return false;

    id<MTLDevice> device = context.device();
    NeuralEvalParams params{};
    params.count = static_cast<uint32_t>(count);
    params.norm = neuralStandardization(model);
    WeightBuffers weights;
    if (!makeWeightBuffers(device, model, inference, &weights, error)) return false;
    id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params
                                                     length:sizeof(params)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:inputs.data()
                                                    length:inputs.size() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:count * 3 * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    if (!paramsBuffer || !inputBuffer || !outputBuffer) return fail(error, "evalNeuralBatch: allocation failed");
    std::vector<id<MTLAllocation>> resident = {paramsBuffer, inputBuffer, outputBuffer, weights.weights};
    if (weights.weights16) resident.push_back(weights.weights16);
    for (id<MTLAllocation> allocation : resident) context.addResident(allocation);
    context.commitResidency();

    [table setAddress:paramsBuffer.gpuAddress atIndex:0];
    [table setAddress:inputBuffer.gpuAddress atIndex:1];
    [table setAddress:outputBuffer.gpuAddress atIndex:2];
    [table setAddress:weights.weights.gpuAddress atIndex:3];
    if (weights.weights16) [table setAddress:weights.weights16.gpuAddress atIndex:4];
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

    // CNRC v2 (tools/train.py): CnrcHeader, float32 inputs[count][n_in], torch fp32 outputs f[count][n_out].
    std::ifstream in(checkPath, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cloudnr: cannot open %s\n", checkPath.c_str());
        return 1;
    }
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CnrcHeader header{};
    if (bytes.size() < sizeof(header)) {
        std::fprintf(stderr, "cloudnr: %s: not a CNRC check file\n", checkPath.c_str());
        return 1;
    }
    std::memcpy(&header, bytes.data(), sizeof(header));
    if (std::memcmp(header.magic, "CNRC", 4) != 0 || header.version != 2) {
        std::fprintf(stderr, "cloudnr: %s: not a CNRC version 2 check file\n", checkPath.c_str());
        return 1;
    }
    const uint64_t count = header.count;
    const uint32_t nIn = header.n_in;
    const uint32_t nOut = header.n_out;
    if (nIn != model->dims.n_in || nOut != model->dims.n_out ||
        bytes.size() != sizeof(header) + count * (uint64_t(nIn) + nOut) * sizeof(float)) {
        std::fprintf(stderr, "cloudnr: %s: dimensions or size do not match the model\n", checkPath.c_str());
        return 1;
    }
    std::vector<float> inputs(count * nIn);
    std::vector<float> expected(count * nOut);
    std::memcpy(inputs.data(), bytes.data() + sizeof(header), inputs.size() * sizeof(float));
    std::memcpy(expected.data(), bytes.data() + sizeof(header) + inputs.size() * sizeof(float),
                expected.size() * sizeof(float));

    auto context = MetalContext::create(&error);
    if (!context) {
        std::fprintf(stderr, "cloudnr: %s\n", error.c_str());
        return 1;
    }

    constexpr double kTolerance = 1e-2;
    bool pass = true;
    std::string variants;
    for (NeuralInference inference : {NeuralInference::Scalar, NeuralInference::ShaderML}) {
        std::vector<float> outputs;
        if (!evalNeuralBatch(*context, *model, inputs, &outputs, inference, &error)) {
            std::fprintf(stderr, "cloudnr: %s inference: %s\n", neuralInferenceName(inference), error.c_str());
            return 1;
        }
        const ParityStats st = compareOutputs(outputs, expected);
        const bool ok = st.finite && st.maxRel <= kTolerance;
        pass = pass && ok;
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "\"%s\":{\"max_rel\":%.9g,\"mean_rel\":%.9g,\"max_abs\":%.9g,\"finite\":%s,\"pass\":%s}",
                      neuralInferenceName(inference), st.maxRel, st.meanRel, st.maxAbs, st.finite ? "true" : "false",
                      ok ? "true" : "false");
        variants += "," + std::string(buf);
    }
    context->waitIdle();
    pass = pass && context->gpuErrorCount() == 0;
    std::printf("{\"model\":%s,\"sha256\":%s,\"count\":%llu,\"tolerance\":%.9g%s,\"gpu_errors\":%llu,\"pass\":%s}\n",
                jsonEscape(modelPath).c_str(), jsonEscape(model->sha256).c_str(),
                static_cast<unsigned long long>(count), kTolerance, variants.c_str(),
                static_cast<unsigned long long>(context->gpuErrorCount()), pass ? "true" : "false");
    return pass ? 0 : 1;
}
