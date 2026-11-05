// Neural renderer POD structs shared by host C++ and MSL (docs/architecture.md §8).
// Shader sources list this file right after common.metal, so it relies on the
// cnr_* typedefs of shared_types.h. File-format headers are host-only.
#pragma once

#ifndef __METAL_VERSION__
#include "shared_types.h"
#endif

// Per-frame parameters of kernel `neural_render`.
struct NeuralParams {
    cnr_uint primary_steps;
    cnr_uint shadow_steps;
    cnr_uint _pad[2];
};
CNR_STATIC_ASSERT(sizeof(NeuralParams) == 16, "NeuralParams layout");

// Kernel `neural_eval_batch`: count descriptors in, count f triples out.
struct NeuralEvalParams {
    cnr_uint count;
    cnr_uint _pad[3];
};
CNR_STATIC_ASSERT(sizeof(NeuralEvalParams) == 16, "NeuralEvalParams layout");

// Kernel `neural_dump` (training-target generation). One dispatch covers records
// [record_offset, record_offset + record_count) of one scene; each record uses
// `chunks` threads that trace paths_per_thread of its `paths` paths each.
struct DumpParams {
    cnr_uint seed;              // per-scene dump seed
    cnr_uint record_offset;     // index of the first record of this dispatch within the scene
    cnr_uint record_count;
    cnr_uint paths;             // paths per record
    cnr_uint paths_per_thread;
    cnr_uint chunks;            // ceil(paths / paths_per_thread)
    cnr_uint shadow_steps;      // sun march for the T_sun descriptor input
    cnr_uint max_tries;         // rejection-sampling attempts for a point with density > 0.01
};
CNR_STATIC_ASSERT(sizeof(DumpParams) == 32, "DumpParams layout");

// Dump kernel output strides (floats). records: [descriptor NN_IN][valid flag] per record.
// partials: [sun, sky, ground, bounce-cap hits] per thread.
#define CNR_DUMP_PARTIAL_STRIDE 4u

#ifndef __METAL_VERSION__
#include <cstdint>

// The only architecture the descriptor and the loader support.
struct NeuralDims {
    uint32_t n_in = 40;           // 7 * levels + 5
    uint32_t n_hidden = 32;
    uint32_t n_layers = 3;        // hidden layers
    uint32_t n_out = 3;           // (f_sun, f_sky, f_gnd)
    uint32_t levels = 5;          // descriptor mip levels
};

// .cnrm model header (little-endian), followed by fp16 weights: per layer W[out][in]
// row-major, then bias[out]; layers 40->32, 32->32 (n_layers - 1 times), 32->3.
struct CnrmHeader {
    char magic[4];                // "CNRM"
    uint32_t format_version;      // 1
    uint32_t n_in;
    uint32_t n_hidden;
    uint32_t n_layers;
    uint32_t n_out;
    float hg_g;
    float albedo;
    uint32_t levels;
};
static_assert(sizeof(CnrmHeader) == 36, "CnrmHeader layout");

// .cnrd training-data header (little-endian, 64 bytes), followed by n_records
// records of float32 [n_in descriptor][n_out targets f].
struct CnrdHeader {
    char magic[4];                // "CNRD"
    uint32_t version;             // 1
    uint32_t n_in;
    uint32_t n_out;
    uint64_t n_records;
    uint32_t seed_begin;          // density seeds [seed_begin, seed_end)
    uint32_t seed_end;
    uint32_t paths;               // paths per target
    float hg_g;
    float albedo;
    uint32_t shadow_steps;        // sun march used for the T_sun input
    uint32_t n_scenes;
    uint32_t records_per_scene;   // requested; dropped records are not written
    uint32_t _pad[2];
};
static_assert(sizeof(CnrdHeader) == 64, "CnrdHeader layout");
#endif
