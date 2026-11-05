// Neural renderer (docs/architecture.md §8): analytic march with an MLP L_rest.
#pragma once

#include "renderer.h"

#include <string>
#include <vector>

struct NeuralModel;

// config.renderer == "neural". Loads config.model (.cnrm) and rejects a scene whose
// hg_g/albedo differ from the model's.
std::unique_ptr<Renderer> makeNeuralRenderer(MetalContext& context, const Scene& scene,
                                             const RenderConfig& config, const DensityVolume& density,
                                             std::string* error);

// Runs mlp_eval (kernel neural_eval_batch) on the GPU: inputs is count * n_in floats,
// outputs receives count * 3 floats f = (f_sun, f_sky, f_gnd).
bool evalNeuralBatch(MetalContext& context, const NeuralModel& model, const std::vector<float>& inputs,
                     std::vector<float>* outputs, std::string* error);

// `cloudnr neural-check --model M.cnrm --check M.check.bin [--validate]`: GPU vs PyTorch
// parity on the check file written by tools/train.py. Exit code 1 if max rel error > 1e-2.
int cmdNeuralCheck(int argc, char** argv);
