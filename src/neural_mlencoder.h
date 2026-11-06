// The cloud network on the Metal 4 machine-learning encoder: models/<name>.mtlpackage (built from
// models/<name>.cnrm by tools/export_mtlpackage.py: fp32 Core ML program, input "x" [N, 13] raw
// descriptors, output "f" [N, 3] = min(y, clamp); standardization is inside the network) dispatched with
// MTL4MachineLearningCommandEncoder on tensors that alias shared buffers.
#pragma once

#import <Metal/Metal.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class MetalContext;
struct NeuralModel;

// Floats per tensor row: ML-usage tensors need a row stride that is a multiple of 64 bytes, so input rows
// are 13 values + 3 pad and output rows 3 values + 13 pad.
inline constexpr uint32_t kMLEncoderRowFloats = 16;
// Largest batch the exported flexible input accepts (tools/export_mtlpackage.py MAX_BATCH).
inline constexpr uint64_t kMLEncoderMaxBatch = uint64_t(1) << 24;

// <stem>.mtlpackage for <stem>.cnrm.
std::string mlPackagePath(const std::string& cnrmPath);
// True if the .mtlpackage of cnrmPath exists.
bool mlPackageExists(const std::string& cnrmPath);

class NeuralMLEncoder {
public:
    // Loads mlPackagePath(model.path) (its sidecar <stem>.mtlpackage.json must name model.sha256 as
    // "source_sha256"), builds the ML pipeline for exactly `batch` rows, validates its reflection ("x" at
    // buffer 0 [13, batch] fp32, "f" at buffer 1 [3, batch] fp32), allocates the padded shared x/f buffers,
    // their tensors and the argument table, and makes them resident. The intermediates heap is the context's
    // shared MetalContext::machineLearningHeap (grown to this pipeline's size if needed). Library load and
    // pipeline build wall time is added to context.addCompileMs.
    static std::unique_ptr<NeuralMLEncoder> create(MetalContext& context, const NeuralModel& model, uint64_t batch,
                                                   std::string* error);
    ~NeuralMLEncoder();  // waits for the GPU, then removes its allocations from the residency set
    NeuralMLEncoder(const NeuralMLEncoder&) = delete;
    NeuralMLEncoder& operator=(const NeuralMLEncoder&) = delete;

    // Encodes one ML encoder pass (f = network(x) for all rows) into commandBuffer. The pass starts with a
    // queue barrier after earlier dispatch and machine-learning stages, so compute writes of x (and earlier
    // reads of f) are ordered before it. Consumers of f must issue a queue barrier after
    // MTLStageMachineLearning.
    void encode(id<MTL4CommandBuffer> commandBuffer) const;

    id<MTLBuffer> inputs() const { return x_; }   // batch rows of kMLEncoderRowFloats: raw descriptor + pad
    id<MTLBuffer> outputs() const { return f_; }  // batch rows of kMLEncoderRowFloats: (f_sun, f_sky, f_gnd) + pad
    uint64_t batch() const { return batch_; }
    // The x/f buffers (the tensors alias them); the shared intermediates heap is context-owned.
    uint64_t memoryBytes() const;

private:
    explicit NeuralMLEncoder(MetalContext& context) : context_(context) {}

    MetalContext& context_;
    uint64_t batch_ = 0;
    id<MTL4MachineLearningPipelineState> pipeline_ = nil;
    id<MTLBuffer> x_ = nil;
    id<MTLBuffer> f_ = nil;
    id<MTLTensor> xTensor_ = nil;
    id<MTLTensor> fTensor_ = nil;
    id<MTLHeap> heap_ = nil;
    id<MTL4ArgumentTable> table_ = nil;
    std::vector<id<MTLAllocation>> resident_;  // empty until create() succeeded
};
