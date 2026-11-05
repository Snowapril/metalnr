// Neural renderer POD structs and constants shared by host C++ and MSL (the article's network:
// 13 inputs -> 16 -> 16 -> 3, squared ReLU, squared output). Shader sources list this file right
// after common.metal, so it relies on the cnr_* typedefs of shared_types.h. File-format headers
// are host-only.
#pragma once

#ifndef __METAL_VERSION__
#include "shared_types.h"
#endif

// Fixed architecture (the loader rejects anything else).
#define CNR_NN_IN 13             // descriptor: 6 local, 6 toward-sun, cos(view, sun)
#define CNR_NN_HIDDEN 16
#define CNR_NN_HIDDEN_LAYERS 2
#define CNR_NN_OUT 3             // f = (f_sun, f_sky, f_gnd)
#define CNR_NN_RADII 6           // descriptor radii, m: 6, 12, 24, 48, 96, 192
#define CNR_NN_IN_PAD 16         // matmul2d: K must be a multiple of 16
#define CNR_NN_OUT_PAD 4         // matmul2d execution_thread: N must be 1, 2, 4 or a multiple of 8

// fp32 weights in .cnrm order (scalar inference): W0[16][13] row-major (out, in), b0[16],
// W1[16][16], b1[16], W2[3][16], b2[3]. Offsets in floats.
#define CNR_NN_W0 0
#define CNR_NN_B0 (CNR_NN_W0 + CNR_NN_HIDDEN * CNR_NN_IN)
#define CNR_NN_W1 (CNR_NN_B0 + CNR_NN_HIDDEN)
#define CNR_NN_B1 (CNR_NN_W1 + CNR_NN_HIDDEN * CNR_NN_HIDDEN)
#define CNR_NN_W2 (CNR_NN_B1 + CNR_NN_HIDDEN)
#define CNR_NN_B2 (CNR_NN_W2 + CNR_NN_OUT * CNR_NN_HIDDEN)
#define CNR_NN_WEIGHT_COUNT (CNR_NN_B2 + CNR_NN_OUT)

// fp16 matrices for Shader ML (matmul2d), each stored [K][N] row-major with N contiguous and zero
// padding: M0[16 (13 inputs + 3 zero rows)][16], M1[16][16], M2[16][4 (3 outputs + 1 zero column)].
// Offsets in halves. Biases stay fp32 and are read from the scalar buffer (CNR_NN_B*).
#define CNR_NN_ML_W0 0
#define CNR_NN_ML_W1 (CNR_NN_ML_W0 + CNR_NN_IN_PAD * CNR_NN_HIDDEN)
#define CNR_NN_ML_W2 (CNR_NN_ML_W1 + CNR_NN_HIDDEN * CNR_NN_HIDDEN)
#define CNR_NN_ML_WEIGHT_COUNT (CNR_NN_ML_W2 + CNR_NN_HIDDEN * CNR_NN_OUT_PAD)

// Input standardization x = (desc - mean) * scale and output clamp f = min(y, clamp), from the model.
// Entries past CNR_NN_IN / CNR_NN_OUT are 0.
struct NeuralStandardization {
    float mean[CNR_NN_IN_PAD];
    float scale[CNR_NN_IN_PAD];
    float clamp[CNR_NN_OUT_PAD];
};
CNR_STATIC_ASSERT(sizeof(NeuralStandardization) == 144, "NeuralStandardization layout");

// Per-frame parameters of kernel `neural_render`.
struct NeuralParams {
    MarchParams march;
    NeuralStandardization norm;
};
CNR_STATIC_ASSERT(sizeof(NeuralParams) == 160, "NeuralParams layout");

// Kernel `neural_eval_batch`: count descriptors in, count f triples out.
struct NeuralEvalParams {
    cnr_uint count;
    cnr_uint _pad[3];
    NeuralStandardization norm;
};
CNR_STATIC_ASSERT(sizeof(NeuralEvalParams) == 160, "NeuralEvalParams layout");

// Kernel `neural_dump` (training-target generation). One dispatch covers records
// [record_offset, record_offset + record_count) of one shape; each record uses `chunks` threads
// that trace paths_per_thread of its `paths` paths each.
struct DumpParams {
    cnr_uint seed;              // per-shape dump seed
    cnr_uint record_offset;     // index of the first record of this dispatch within the shape
    cnr_uint record_count;
    cnr_uint paths;             // paths per record
    cnr_uint paths_per_thread;
    cnr_uint chunks;            // ceil(paths / paths_per_thread)
    cnr_uint max_tries;         // camera rays per record before it is dropped (no collision)
    cnr_uint _pad;
};
CNR_STATIC_ASSERT(sizeof(DumpParams) == 32, "DumpParams layout");

// Dump kernel output strides (floats). records: [descriptor CNR_NN_IN][valid flag] per record.
// partials: [sun, sky, ground, bounce-cap hits] per thread.
#define CNR_DUMP_RECORD_STRIDE 14u
#define CNR_DUMP_PARTIAL_STRIDE 4u

#ifndef __METAL_VERSION__
#include <cstdint>

static_assert(CNR_DUMP_RECORD_STRIDE == CNR_NN_IN + 1, "dump record stride");
static_assert(CNR_NN_WEIGHT_COUNT == 547, "weight count");
static_assert(CNR_NN_ML_WEIGHT_COUNT == 576, "shader ML weight count");

// The only architecture the descriptor, the shaders and the loader support.
struct NeuralDims {
    uint32_t n_in = CNR_NN_IN;
    uint32_t n_hidden = CNR_NN_HIDDEN;
    uint32_t n_hidden_layers = CNR_NN_HIDDEN_LAYERS;
    uint32_t n_out = CNR_NN_OUT;
};

// .cnrm model file, format_version 2 (little-endian): this header, then CNR_NN_WEIGHT_COUNT fp32
// weights in the order of CNR_NN_W0..CNR_NN_B2.
struct CnrmHeader {
    char magic[4];                // "CNRM"
    uint32_t format_version;      // 2
    uint32_t n_in;                // 13
    uint32_t n_hidden;            // 16
    uint32_t n_hidden_layers;     // 2
    uint32_t n_out;               // 3
    float hg_g;
    float albedo;
    float radii[CNR_NN_RADII];    // must equal the descriptor radii
    float mean[CNR_NN_IN];
    float scale[CNR_NN_IN];
    float clamp[CNR_NN_OUT];
};
static_assert(sizeof(CnrmHeader) == 172, "CnrmHeader layout");

// .cnrd training data, version 2 (little-endian, 64-byte header), followed by n_records records of
// float32 [n_in descriptor][n_out targets f].
struct CnrdHeader {
    char magic[4];                // "CNRD"
    uint32_t version;             // 2
    uint32_t n_in;                // 13
    uint32_t n_out;               // 3
    uint64_t n_records;
    uint32_t shape_begin;         // makeRandomCumulus seeds [shape_begin, shape_end)
    uint32_t shape_end;
    uint32_t n_shapes;
    uint32_t records_per_shape;   // requested; records without a collision are not written
    uint32_t paths;               // paths per target
    float hg_g;
    float albedo;
    float sigma_t_min;            // log-uniform sigma_t_max range of the shapes
    float sigma_t_max;
    uint32_t fixed_shape;         // 1: every record uses the --template cloud and sigma (sun varies only)
};
static_assert(sizeof(CnrdHeader) == 64, "CnrdHeader layout");

// Parity check file, version 2, written by tools/train.py: this header, then float32
// inputs[count][n_in] and the PyTorch fp32 outputs f[count][n_out].
struct CnrcHeader {
    char magic[4];                // "CNRC"
    uint32_t version;             // 2
    uint32_t count;
    uint32_t n_in;
    uint32_t n_out;
};
static_assert(sizeof(CnrcHeader) == 20, "CnrcHeader layout");
#endif
