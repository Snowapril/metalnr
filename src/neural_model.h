// Neural shader model artifact (.cnrm format_version 2, the article's 13 -> 16 -> 16 -> 3 network):
// loading, validation and the GPU weight layouts of both inference variants.
#pragma once

#include "neural_types.h"
#include "scene.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Descriptor radii (m); the model header must carry exactly these.
inline constexpr std::array<float, CNR_NN_RADII> kNeuralRadii = {6.0f, 12.0f, 24.0f, 48.0f, 96.0f, 192.0f};

// RenderConfig.inference values with an in-module implementation.
enum class NeuralInference {
    Scalar,    // "scalar": hand-written MSL loops, fp32 weights
    ShaderML,  // "shader_ml": Metal 4 Shader ML matmul2d, fp16 weights and activations
};
const char* neuralInferenceName(NeuralInference inference);

struct NeuralModel {
    std::string path;
    uint32_t format_version = 0;
    NeuralDims dims;
    float hg_g = 0.0f;
    float albedo = 0.0f;
    std::array<float, CNR_NN_RADII> radii{};
    std::array<float, CNR_NN_IN> mean{};
    std::array<float, CNR_NN_IN> scale{};
    std::array<float, CNR_NN_OUT> clamp{};
    // fp32 weights in file order (CNR_NN_W0 .. CNR_NN_B2): the scalar layout, and the biases of
    // the Shader ML variant.
    std::vector<float> weights;
    // fp16 bit patterns, CNR_NN_ML_WEIGHT_COUNT: each matrix transposed to [K][N] (N contiguous)
    // with zero padding to K = 16 inputs and N = 4 outputs (matmul2d layout).
    std::vector<uint16_t> weights_ml;
    std::string sha256;  // lowercase hex of the whole .cnrm file
    // Inclusive shape-seed range of the training set, from the sidecar <name>.json
    // ("train_seed_range"); false if the sidecar is missing or has no such key.
    bool has_train_seed_range = false;
    uint32_t train_seed_first = 0;
    uint32_t train_seed_last = 0;
};

// Number of fp32 weights and biases in a .cnrm (547).
uint64_t neuralWeightCount(const NeuralDims& dims);

// Reads and validates a .cnrm: magic "CNRM", format_version 2, dimensions (13, 16, 2, 3), the
// descriptor radii, the exact file size, finite hg_g/albedo/mean/scale/weights, positive clamp,
// weights representable in fp16. Logs `neural model <path> sha256=<hex> train_seeds=<a>-<b>`
// to stderr.
std::optional<NeuralModel> loadNeuralModel(const std::string& path, std::string* error);

// The model is valid only for the medium it was trained on: rejects a scene whose
// medium.hg_g or medium.albedo differs from the model's by more than 1e-6.
bool validateNeuralModelForScene(const NeuralModel& model, const Scene& scene, std::string* error);

// Standardization and clamp constants for the shaders.
NeuralStandardization neuralStandardization(const NeuralModel& model);

// `#define NN_*` lines for neural_common.metal (dimensions and NN_SHADER_ML).
std::string neuralShaderDefines(const NeuralDims& dims, NeuralInference inference);

// Shader files (after shared_types.h and common.metal) for neural kernels.
std::vector<std::string> neuralShaderFiles(const std::string& kernelFile);
