// Neural L_rest model artifact (.cnrm, docs/architecture.md §8): loading and validation.
#pragma once

#include "neural_types.h"
#include "scene.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct NeuralModel {
    std::string path;
    uint32_t format_version = 0;
    NeuralDims dims;
    float hg_g = 0.0f;
    float albedo = 0.0f;
    std::vector<uint16_t> weights;  // fp16 bit patterns, layout of CnrmHeader
    std::string sha256;             // lowercase hex of the whole .cnrm file
    // Inclusive density-seed range of the training set, from the sidecar <name>.json
    // ("train_seed_range"); false if the sidecar is missing or has no such key.
    bool has_train_seed_range = false;
    uint32_t train_seed_first = 0;
    uint32_t train_seed_last = 0;
};

// Number of fp16 values (weights and biases) of an MLP with these dimensions.
uint64_t neuralWeightCount(const NeuralDims& dims);

// Reads and validates a .cnrm: magic "CNRM", format_version 1, dimensions equal to the
// fixed architecture NeuralDims{} and the exact file size. Logs
// `neural model <path> sha256=<hex> train_seeds=<a>-<b>` to stderr.
std::optional<NeuralModel> loadNeuralModel(const std::string& path, std::string* error);

// The model is valid only for the medium it was trained on: rejects a scene whose
// medium.hg_g or medium.albedo differs from the model's by more than 1e-6.
bool validateNeuralModelForScene(const NeuralModel& model, const Scene& scene, std::string* error);

// `#define NN_*` lines for neural_common.metal.
std::string neuralShaderDefines(const NeuralDims& dims);

// Shader files (after shared_types.h and common.metal) for neural kernels.
std::vector<std::string> neuralShaderFiles(const std::string& kernelFile);
