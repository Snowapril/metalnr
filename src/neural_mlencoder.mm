#include "neural_mlencoder.h"

#import <Foundation/Foundation.h>

#include "metal_context.h"
#include "neural_model.h"

#include <algorithm>
#include <chrono>

namespace {

bool fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

std::string describe(NSError* error) {
    return error ? std::string(error.localizedDescription.UTF8String) : std::string("unknown error");
}

std::string stem(const std::string& cnrmPath) {
    const std::string ext = ".cnrm";
    if (cnrmPath.size() > ext.size() && cnrmPath.compare(cnrmPath.size() - ext.size(), ext.size(), ext) == 0)
        return cnrmPath.substr(0, cnrmPath.size() - ext.size());
    return cnrmPath;
}

NSString* ns(const std::string& s) { return [NSString stringWithUTF8String:s.c_str()]; }

// "source_sha256" of the export sidecar, empty if missing or malformed.
std::string sidecarSourceSha(const std::string& sidecarPath) {
    NSData* data = [NSData dataWithContentsOfFile:ns(sidecarPath)];
    if (!data) return {};
    id root = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    if (![root isKindOfClass:[NSDictionary class]]) return {};
    id sha = ((NSDictionary*)root)[@"source_sha256"];
    return [sha isKindOfClass:[NSString class]] ? std::string([(NSString*)sha UTF8String]) : std::string();
}

MTLTensorExtents* extents2(NSInteger inner, NSInteger outer) {
    const NSInteger values[2] = {inner, outer};
    return [[MTLTensorExtents alloc] initWithRank:2 values:values];
}

// The reflected tensor binding `name` must sit at `index` with fp32 data and dimensions [features, batch].
bool checkBinding(MTL4MachineLearningPipelineReflection* reflection, NSString* name, NSUInteger index,
                  NSInteger features, NSInteger batch, std::string* error) {
    for (id<MTLBinding> binding in reflection.bindings) {
        if (![binding.name isEqualToString:name]) continue;
        const std::string what = "ml_encoder: model binding '" + std::string(name.UTF8String) + "'";
        if (![binding conformsToProtocol:@protocol(MTLTensorBinding)]) return fail(error, what + " is not a tensor");
        id<MTLTensorBinding> tensor = (id<MTLTensorBinding>)binding;
        if (binding.index != index)
            return fail(error, what + " is at buffer index " + std::to_string(binding.index) + ", expected " +
                                   std::to_string(index));
        if (tensor.tensorDataType != MTLTensorDataTypeFloat32) return fail(error, what + " is not float32");
        MTLTensorExtents* dims = tensor.dimensions;
        if (!dims || dims.rank != 2 || [dims extentAtDimensionIndex:0] != features ||
            [dims extentAtDimensionIndex:1] != batch)
            return fail(error, what + " does not have dimensions [" + std::to_string(features) + ", " +
                                   std::to_string(batch) + "] (re-export with tools/export_mtlpackage.py)");
        return true;
    }
    return fail(error, "ml_encoder: model has no binding '" + std::string(name.UTF8String) + "'");
}

}  // namespace

std::string mlPackagePath(const std::string& cnrmPath) { return stem(cnrmPath) + ".mtlpackage"; }

bool mlPackageExists(const std::string& cnrmPath) {
    return [[NSFileManager defaultManager] fileExistsAtPath:ns(mlPackagePath(cnrmPath))];
}

std::unique_ptr<NeuralMLEncoder> NeuralMLEncoder::create(MetalContext& context, const NeuralModel& model,
                                                         uint64_t batch, std::string* error) {
    const std::string packagePath = mlPackagePath(model.path);
    if (batch == 0 || batch > kMLEncoderMaxBatch) {
        fail(error, "ml_encoder: batch " + std::to_string(batch) + " outside [1, " +
                        std::to_string(kMLEncoderMaxBatch) + "] (the exported input range)");
        return nullptr;
    }
    if (!mlPackageExists(model.path)) {
        fail(error, "ml_encoder: no " + packagePath + " (run .venv/bin/python tools/export_mtlpackage.py --model " +
                        model.path + ")");
        return nullptr;
    }
    const std::string sidecar = packagePath + ".json";
    const std::string sourceSha = sidecarSourceSha(sidecar);
    if (sourceSha != model.sha256) {
        fail(error, "ml_encoder: " + packagePath + " was not exported from this " + model.path + " (" + sidecar +
                        " source_sha256 '" + sourceSha + "', model sha256 " + model.sha256 +
                        "); re-run tools/export_mtlpackage.py");
        return nullptr;
    }

    id<MTLDevice> device = context.device();
    std::unique_ptr<NeuralMLEncoder> enc(new NeuralMLEncoder(context));
    enc->batch_ = batch;
    const NSInteger rows = static_cast<NSInteger>(batch);

    const auto compileStart = std::chrono::steady_clock::now();
    auto addCompileTime = [&] {
        context.addCompileMs(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - compileStart).count());
    };
    NSError* nsError = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:ns(packagePath)] error:&nsError];
    if (!library) {
        addCompileTime();
        fail(error, "ml_encoder: cannot load " + packagePath + ": " + describe(nsError));
        return nullptr;
    }
    MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
    function.name = @"main";
    function.library = library;
    MTL4MachineLearningPipelineDescriptor* desc = [MTL4MachineLearningPipelineDescriptor new];
    desc.label = @"cloudnr mlp";
    desc.machineLearningFunctionDescriptor = function;
    MTL4PipelineOptions* options = [MTL4PipelineOptions new];
    // Without reflection the pipeline's bindings are empty.
    options.shaderReflection = MTL4ShaderReflectionBindingInfo | MTL4ShaderReflectionBufferTypeInfo;
    desc.options = options;
    [desc setInputDimensions:extents2(CNR_NN_IN, rows) atBufferIndex:0];
    enc->pipeline_ = [context.compiler() newMachineLearningPipelineStateWithDescriptor:desc error:&nsError];
    addCompileTime();
    if (!enc->pipeline_) {
        fail(error, "ml_encoder: pipeline creation failed for " + packagePath + ": " + describe(nsError));
        return nullptr;
    }
    MTL4MachineLearningPipelineReflection* reflection = enc->pipeline_.reflection;
    if (!reflection) {
        fail(error, "ml_encoder: pipeline has no reflection");
        return nullptr;
    }
    if (!checkBinding(reflection, @"x", 0, CNR_NN_IN, rows, error) ||
        !checkBinding(reflection, @"f", 1, CNR_NN_OUT, rows, error))
        return nullptr;

    const NSUInteger bytes = NSUInteger(batch) * kMLEncoderRowFloats * sizeof(float);
    enc->x_ = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    enc->f_ = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!enc->x_ || !enc->f_) {
        fail(error, "ml_encoder: allocation of 2 x " + std::to_string(bytes) + " bytes failed");
        return nullptr;
    }
    enc->x_.label = @"ml_encoder x rows";
    enc->f_.label = @"ml_encoder f rows";

    auto makeTensor = [&](id<MTLBuffer> buffer, NSInteger features) -> id<MTLTensor> {
        MTLTensorDescriptor* td = [MTLTensorDescriptor new];
        td.dimensions = extents2(features, rows);
        td.strides = extents2(1, kMLEncoderRowFloats);
        td.dataType = MTLTensorDataTypeFloat32;
        td.usage = MTLTensorUsageMachineLearning;
        NSError* tensorError = nil;
        id<MTLTensor> tensor = [buffer newTensorWithDescriptor:td offset:0 error:&tensorError];
        if (!tensor) fail(error, "ml_encoder: newTensorWithDescriptor failed: " + describe(tensorError));
        return tensor;
    };
    enc->xTensor_ = makeTensor(enc->x_, CNR_NN_IN);
    if (!enc->xTensor_) return nullptr;
    enc->fTensor_ = makeTensor(enc->f_, CNR_NN_OUT);
    if (!enc->fTensor_) return nullptr;

    // Context-owned and shared: releasing a dispatched ML heap crashes the GPU validation layer (metal_context.h).
    enc->heap_ = context.machineLearningHeap(enc->pipeline_.intermediatesHeapSize, error);
    if (!enc->heap_) return nullptr;

    enc->table_ = context.makeArgumentTable(2, 0, error);
    if (!enc->table_) return nullptr;
    [enc->table_ setResource:enc->xTensor_.gpuResourceID atBufferIndex:0];
    [enc->table_ setResource:enc->fTensor_.gpuResourceID atBufferIndex:1];

    enc->resident_ = {enc->x_, enc->f_, enc->xTensor_, enc->fTensor_};
    for (id<MTLAllocation> allocation : enc->resident_) context.addResident(allocation);
    context.commitResidency();
    return enc;
}

NeuralMLEncoder::~NeuralMLEncoder() {
    if (resident_.empty()) return;
    context_.waitIdle();
    for (id<MTLAllocation> allocation : resident_) context_.removeResident(allocation);
    context_.commitResidency();
}

void NeuralMLEncoder::encode(id<MTL4CommandBuffer> commandBuffer) const {
    id<MTL4MachineLearningCommandEncoder> encoder = [commandBuffer machineLearningCommandEncoder];
    // x was written by earlier dispatches; f and the heap may still be in use by earlier dispatches / ML passes.
    [encoder barrierAfterQueueStages:MTLStageDispatch | MTLStageMachineLearning
                        beforeStages:MTLStageMachineLearning
                   visibilityOptions:MTL4VisibilityOptionDevice];
    [encoder setPipelineState:pipeline_];
    [encoder setArgumentTable:table_];
    [encoder dispatchNetworkWithIntermediatesHeap:heap_];
    [encoder endEncoding];
}

uint64_t NeuralMLEncoder::memoryBytes() const {
    // The intermediates heap is shared and context-owned, so it is not counted here.
    return uint64_t(x_.allocatedSize) + f_.allocatedSize;
}
