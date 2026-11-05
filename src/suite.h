// Full comparison experiment (docs/experiment-protocol.md): references with both acceptance gates,
// steady-state benchmarks of every method on every scene/camera, equal-quality spp, and exported
// reports (records/*.json, summary.csv, report.md).
//
//   cloudnr suite --experiment experiments/eval.json [--out results/<name>] [--quick]
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Experiment {
    std::string name;
    std::vector<std::string> scenes;   // scene JSON paths
    std::vector<std::string> cameras;  // camera names, each must exist in every scene
    std::vector<std::string> methods;  // RenderConfig JSON paths
    std::string reference_config;      // pt config for the references
    uint32_t reference_initial_spp = 1024;
    uint32_t reference_max_spp = 16384;
    double abs_gate = 1e-4;
    double rel_gate = 0.1;
    uint32_t warmup = 30;
    uint32_t frames = 300;
    uint32_t runs = 3;
    uint32_t equal_quality_seed = 3;
    uint32_t equal_quality_max_spp = 4096;
};

// Every key is required; unknown keys are an error.
bool parseExperiment(const std::string& path, Experiment* experiment, std::string* error);

// Smallest power of two >= max(2 * spp, spp * estimate / target), capped at maxSpp. Uses the 1/spp
// variance scaling of the reference error estimate.
uint32_t nextReferenceSpp(uint32_t spp, double estimate, double target, uint32_t maxSpp);

int cmdSuite(int argc, char** argv);
