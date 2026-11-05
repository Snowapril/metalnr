// Neural training-data generation (the article's dataset: random cumulus shapes, camera-style
// sampling, full in-scattered radiance targets).
#pragma once

// `cloudnr dump-training --out FILE.cnrd --shape-begin A --shape-end B --shapes K
//  --records-per-shape R [--paths 256] [--template scenes/cumulus.json] [--validate]`
// Shape k uses cloud = makeRandomCumulus(A + (k mod (B - A)), template volume) (B exclusive,
// A >= 1000 because seeds 1-999 are reserved for evaluation), sun elevation uniform in
// [-10, 90] deg, azimuth uniform, sigma_t_max log-uniform in [0.02, 0.1], albedo/g from the
// template, unit lights. Writes a .cnrd v2 file (CnrdHeader in neural_types.h), logs progress
// per shape to stderr and prints a JSON summary.
int cmdDumpTraining(int argc, char** argv);
