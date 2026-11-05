// Neural renderer (the article's neural shader): the shared march with the in-scattered radiance
// from the 13 -> 16 -> 16 -> 3 MLP, inference via hand-written MSL or Metal 4 Shader ML.
#pragma once

#include "renderer.h"

#include <string>
#include <vector>

struct NeuralModel;
enum class NeuralInference;

// config.renderer == "neural". Loads config.model (.cnrm v2), rejects a scene whose hg_g/albedo
// differ from the model's. config.inference: "scalar" | "shader_ml" ("ml_encoder" is provided by
// the ML encoder module and rejected here).
std::unique_ptr<Renderer> makeNeuralRenderer(MetalContext& context, const Scene& scene,
                                             const RenderConfig& config, const DensityVolume& density,
                                             std::string* error);

// Runs kernel neural_eval_batch (the given variant) on the GPU: inputs is count * 13 raw descriptors,
// outputs receives count * 3 floats f = (f_sun, f_sky, f_gnd).
bool evalNeuralBatch(MetalContext& context, const NeuralModel& model, const std::vector<float>& inputs,
                     std::vector<float>* outputs, NeuralInference inference, std::string* error);

// `cloudnr neural-check --model M.cnrm --check M.check.bin [--validate]`: GPU vs PyTorch fp32 parity
// for BOTH inference variants on the CNRC v2 check file written by tools/train.py. Prints per-variant
// max/mean relative error |a-b|/(|b|+1e-3); exit code 1 if either max exceeds 1e-2.
int cmdNeuralCheck(int argc, char** argv);
