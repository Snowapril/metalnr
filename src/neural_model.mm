#include "neural_model.h"

#import <Foundation/Foundation.h>

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

namespace {

bool fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

std::string sha256Hex(const std::vector<uint8_t>& bytes) {
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    // CC_LONG is 32-bit: feed large inputs in chunks.
    constexpr size_t kChunk = size_t(1) << 30;
    for (size_t offset = 0; offset < bytes.size(); offset += kChunk) {
        const size_t n = std::min(kChunk, bytes.size() - offset);
        CC_SHA256_Update(&ctx, bytes.data() + offset, static_cast<CC_LONG>(n));
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &ctx);
    std::string hex;
    char buf[3];
    for (unsigned char byte : digest) {
        std::snprintf(buf, sizeof(buf), "%02x", byte);
        hex += buf;
    }
    return hex;
}

std::string sidecarPath(const std::string& path) {
    const std::string ext = ".cnrm";
    if (path.size() > ext.size() && path.compare(path.size() - ext.size(), ext.size(), ext) == 0)
        return path.substr(0, path.size() - ext.size()) + ".json";
    return path + ".json";
}

void readSidecar(const std::string& path, NeuralModel* model) {
    NSData* data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:sidecarPath(path).c_str()]];
    if (!data) return;
    id root = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    if (![root isKindOfClass:[NSDictionary class]]) return;
    id range = ((NSDictionary*)root)[@"train_seed_range"];
    if (![range isKindOfClass:[NSArray class]] || [(NSArray*)range count] != 2) return;
    id first = ((NSArray*)range)[0];
    id last = ((NSArray*)range)[1];
    if (![first isKindOfClass:[NSNumber class]] || ![last isKindOfClass:[NSNumber class]]) return;
    model->has_train_seed_range = true;
    model->train_seed_first = [(NSNumber*)first unsignedIntValue];
    model->train_seed_last = [(NSNumber*)last unsignedIntValue];
}

uint16_t halfBits(float value) {
    const __fp16 h = static_cast<__fp16>(value);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

bool allFinite(const float* values, size_t count) {
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(values[i])) return false;
    return true;
}

// Transposes W[n][k] (row-major, nOut x nIn) into the matmul2d [K][N] layout padded to kPad x nPad.
void packMatrix(const float* w, uint32_t nOut, uint32_t nIn, uint32_t nPad, uint32_t kPad, uint16_t* out) {
    for (uint32_t k = 0; k < kPad; ++k)
        for (uint32_t n = 0; n < nPad; ++n)
            out[size_t(k) * nPad + n] = (k < nIn && n < nOut) ? halfBits(w[size_t(n) * nIn + k]) : halfBits(0.0f);
}

}  // namespace

const char* neuralInferenceName(NeuralInference inference) {
    return inference == NeuralInference::ShaderML ? "shader_ml" : "scalar";
}

uint64_t neuralWeightCount(const NeuralDims& dims) {
    const uint64_t in = dims.n_in, hidden = dims.n_hidden, out = dims.n_out;
    if (dims.n_hidden_layers == 0) return 0;
    return (in * hidden + hidden) + uint64_t(dims.n_hidden_layers - 1) * (hidden * hidden + hidden) +
           (hidden * out + out);
}

std::optional<NeuralModel> loadNeuralModel(const std::string& path, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail(error, "cannot open neural model " + path);
        return std::nullopt;
    }
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.size() < sizeof(CnrmHeader)) {
        fail(error, path + ": file too small for a .cnrm header");
        return std::nullopt;
    }
    CnrmHeader header;
    std::memcpy(&header, bytes.data(), sizeof(header));
    if (std::memcmp(header.magic, "CNRM", 4) != 0) {
        fail(error, path + ": bad magic (expected CNRM)");
        return std::nullopt;
    }
    if (header.format_version != 2) {
        fail(error, path + ": unsupported format_version " + std::to_string(header.format_version) + " (expected 2)");
        return std::nullopt;
    }
    const NeuralDims expected;
    if (header.n_in != expected.n_in || header.n_hidden != expected.n_hidden ||
        header.n_hidden_layers != expected.n_hidden_layers || header.n_out != expected.n_out) {
        fail(error, path + ": dimensions (n_in " + std::to_string(header.n_in) + ", n_hidden " +
                        std::to_string(header.n_hidden) + ", n_hidden_layers " + std::to_string(header.n_hidden_layers) +
                        ", n_out " + std::to_string(header.n_out) + ") differ from the supported (13, 16, 2, 3)");
        return std::nullopt;
    }
    const uint64_t count = neuralWeightCount(expected);
    const uint64_t expectedSize = sizeof(CnrmHeader) + count * sizeof(float);
    if (bytes.size() != expectedSize) {
        fail(error, path + ": size " + std::to_string(bytes.size()) + " bytes, expected " +
                        std::to_string(expectedSize));
        return std::nullopt;
    }
    if (!std::isfinite(header.hg_g) || !std::isfinite(header.albedo)) {
        fail(error, path + ": non-finite hg_g or albedo");
        return std::nullopt;
    }
    for (int i = 0; i < CNR_NN_RADII; ++i) {
        if (std::fabs(header.radii[i] - kNeuralRadii[i]) > 1e-6f * kNeuralRadii[i]) {
            fail(error, path + ": descriptor radius " + std::to_string(i) + " is " + std::to_string(header.radii[i]) +
                            " m, the shaders use " + std::to_string(kNeuralRadii[i]) + " m");
            return std::nullopt;
        }
    }
    if (!allFinite(header.mean, CNR_NN_IN) || !allFinite(header.scale, CNR_NN_IN)) {
        fail(error, path + ": non-finite standardization mean or scale");
        return std::nullopt;
    }
    for (float c : header.clamp) {
        if (!std::isfinite(c) || c <= 0.0f) {
            fail(error, path + ": output clamp must be finite and > 0");
            return std::nullopt;
        }
    }

    NeuralModel model;
    model.path = path;
    model.format_version = header.format_version;
    model.dims = expected;
    model.hg_g = header.hg_g;
    model.albedo = header.albedo;
    std::copy(std::begin(header.radii), std::end(header.radii), model.radii.begin());
    std::copy(std::begin(header.mean), std::end(header.mean), model.mean.begin());
    std::copy(std::begin(header.scale), std::end(header.scale), model.scale.begin());
    std::copy(std::begin(header.clamp), std::end(header.clamp), model.clamp.begin());
    model.weights.resize(count);
    std::memcpy(model.weights.data(), bytes.data() + sizeof(CnrmHeader), count * sizeof(float));
    for (float w : model.weights) {
        if (!std::isfinite(w)) {
            fail(error, path + ": non-finite weight");
            return std::nullopt;
        }
        if (std::fabs(w) > 65504.0f) {
            fail(error, path + ": weight " + std::to_string(w) + " exceeds the fp16 range (shader_ml inference)");
            return std::nullopt;
        }
    }

    model.weights_ml.resize(CNR_NN_ML_WEIGHT_COUNT);
    const float* w = model.weights.data();
    packMatrix(w + CNR_NN_W0, CNR_NN_HIDDEN, CNR_NN_IN, CNR_NN_HIDDEN, CNR_NN_IN_PAD,
               model.weights_ml.data() + CNR_NN_ML_W0);
    packMatrix(w + CNR_NN_W1, CNR_NN_HIDDEN, CNR_NN_HIDDEN, CNR_NN_HIDDEN, CNR_NN_HIDDEN,
               model.weights_ml.data() + CNR_NN_ML_W1);
    packMatrix(w + CNR_NN_W2, CNR_NN_OUT, CNR_NN_HIDDEN, CNR_NN_OUT_PAD, CNR_NN_HIDDEN,
               model.weights_ml.data() + CNR_NN_ML_W2);

    model.sha256 = sha256Hex(bytes);
    readSidecar(path, &model);

    std::string seeds = "unknown";
    if (model.has_train_seed_range)
        seeds = std::to_string(model.train_seed_first) + "-" + std::to_string(model.train_seed_last);
    std::fprintf(stderr, "neural model %s sha256=%s train_seeds=%s\n", path.c_str(), model.sha256.c_str(),
                 seeds.c_str());
    return model;
}

bool validateNeuralModelForScene(const NeuralModel& model, const Scene& scene, std::string* error) {
    if (std::fabs(scene.medium.hg_g - model.hg_g) > 1e-6f)
        return fail(error, model.path + ": model trained for hg_g " + std::to_string(model.hg_g) + ", scene has " +
                               std::to_string(scene.medium.hg_g));
    if (std::fabs(scene.medium.albedo - model.albedo) > 1e-6f)
        return fail(error, model.path + ": model trained for albedo " + std::to_string(model.albedo) +
                               ", scene has " + std::to_string(scene.medium.albedo));
    return true;
}

NeuralStandardization neuralStandardization(const NeuralModel& model) {
    NeuralStandardization n{};
    std::copy(model.mean.begin(), model.mean.end(), n.mean);
    std::copy(model.scale.begin(), model.scale.end(), n.scale);
    std::copy(model.clamp.begin(), model.clamp.end(), n.clamp);
    return n;
}

std::string neuralShaderDefines(const NeuralDims& dims, NeuralInference inference) {
    return "#define NN_IN " + std::to_string(dims.n_in) + "\n#define NN_HIDDEN " + std::to_string(dims.n_hidden) +
           "\n#define NN_HIDDEN_LAYERS " + std::to_string(dims.n_hidden_layers) + "\n#define NN_OUT " +
           std::to_string(dims.n_out) + "\n#define NN_SHADER_ML " +
           (inference == NeuralInference::ShaderML ? "1" : "0") + "\n";
}

std::vector<std::string> neuralShaderFiles(const std::string& kernelFile) {
    return {"src/neural_types.h", "src/shaders/neural_common.metal", kernelFile};
}
