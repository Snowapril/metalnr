// Neural training-data generation (docs/architecture.md §8, protocol §1).
#pragma once

// `cloudnr dump-training --out FILE.cnrd --seed-begin A --seed-end B --scenes K
//  --records-per-scene R [--paths 1024] [--shadow-steps 16] [--template scenes/eval_medium.json]
//  [--validate]`
// Scene k uses density_seed A + (k mod (B - A)) (B exclusive, A >= 1000 because seeds
// 1-999 are reserved for evaluation), sun elevation uniform in [5, 90] deg, azimuth uniform,
// sigma_t_max log-uniform in [0.005, 0.15], unit lights; everything else from the template.
// Writes a .cnrd file (CnrdHeader in neural_types.h) and prints a JSON summary.
int cmdDumpTraining(int argc, char** argv);
