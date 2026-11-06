// Neural renderer (the article's neural shader): the shared march with the in-scattered radiance
// from the 13 -> 16 -> 16 -> 3 MLP, inference via hand-written MSL, Metal 4 Shader ML, or the Metal 4
// machine-learning encoder (wavefront march, neural_wave.metal + NeuralMLEncoder).
#pragma once

#include "renderer.h"

#include <string>
#include <vector>

struct NeuralModel;
enum class NeuralInference;

// config.renderer == "neural". Loads config.model (.cnrm v2), rejects a scene whose hg_g/albedo
// differ from the model's. config.inference: "scalar" | "shader_ml" | "ml_encoder". "ml_encoder"
// needs <stem>.mtlpackage exported from that .cnrm (tools/export_mtlpackage.py); its renderer encodes
// its own passes (Renderer::encodesOwnPasses) and must be driven through encodeFrame().
std::unique_ptr<Renderer> makeNeuralRenderer(MetalContext& context, const Scene& scene,
                                             const RenderConfig& config, const DensityVolume& density,
                                             std::string* error);

// Evaluates the network on the GPU with the given variant: inputs is count * 13 raw descriptors,
// outputs receives count * 3 floats f = (f_sun, f_sky, f_gnd). scalar / shader_ml run kernel
// neural_eval_batch; ml_encoder runs one machine-learning encoder dispatch over the whole batch.
// gpuMs (optional) receives the GPU time of the command buffer.
bool evalNeuralBatch(MetalContext& context, const NeuralModel& model, const std::vector<float>& inputs,
                     std::vector<float>* outputs, NeuralInference inference, std::string* error,
                     double* gpuMs = nullptr);

// `cloudnr neural-check --model M.cnrm --check M.check.bin [--validate]`: GPU vs PyTorch fp32 parity
// on the CNRC v2 check file written by tools/train.py for scalar, shader_ml and ml_encoder (reported as
// "unavailable (no .mtlpackage)" without failing when M.mtlpackage is absent). Prints per-variant
// max/mean relative error |a-b|/(|b|+1e-3) and the GPU ms of the batch; exit code 1 if any evaluated
// variant's max exceeds 1e-2.
int cmdNeuralCheck(int argc, char** argv);
