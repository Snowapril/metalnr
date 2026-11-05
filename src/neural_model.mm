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

}  // namespace

uint64_t neuralWeightCount(const NeuralDims& dims) {
    const uint64_t in = dims.n_in, hidden = dims.n_hidden, out = dims.n_out;
    if (dims.n_layers == 0) return 0;
    return (in * hidden + hidden) + uint64_t(dims.n_layers - 1) * (hidden * hidden + hidden) + (hidden * out + out);
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
    if (header.format_version != 1) {
        fail(error, path + ": unsupported format_version " + std::to_string(header.format_version) + " (expected 1)");
        return std::nullopt;
    }
    const NeuralDims expected;
    if (header.n_in != expected.n_in || header.n_hidden != expected.n_hidden || header.n_layers != expected.n_layers ||
        header.n_out != expected.n_out || header.levels != expected.levels) {
        fail(error, path + ": dimensions (n_in " + std::to_string(header.n_in) + ", n_hidden " +
                        std::to_string(header.n_hidden) + ", n_layers " + std::to_string(header.n_layers) +
                        ", n_out " + std::to_string(header.n_out) + ", levels " + std::to_string(header.levels) +
                        ") differ from the supported (40, 32, 3, 3, 5)");
        return std::nullopt;
    }
    if (!std::isfinite(header.hg_g) || !std::isfinite(header.albedo)) {
        fail(error, path + ": non-finite hg_g or albedo");
        return std::nullopt;
    }

    NeuralModel model;
    model.path = path;
    model.format_version = header.format_version;
    model.dims.n_in = header.n_in;
    model.dims.n_hidden = header.n_hidden;
    model.dims.n_layers = header.n_layers;
    model.dims.n_out = header.n_out;
    model.dims.levels = header.levels;
    model.hg_g = header.hg_g;
    model.albedo = header.albedo;

    const uint64_t count = neuralWeightCount(model.dims);
    const uint64_t expectedSize = sizeof(CnrmHeader) + count * sizeof(uint16_t);
    if (bytes.size() != expectedSize) {
        fail(error, path + ": size " + std::to_string(bytes.size()) + " bytes, expected " +
                        std::to_string(expectedSize));
        return std::nullopt;
    }
    model.weights.resize(count);
    std::memcpy(model.weights.data(), bytes.data() + sizeof(CnrmHeader), count * sizeof(uint16_t));
    for (uint16_t bits : model.weights) {
        if ((bits & 0x7C00u) == 0x7C00u) {  // fp16 inf or NaN
            fail(error, path + ": non-finite weight");
            return std::nullopt;
        }
    }
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

std::string neuralShaderDefines(const NeuralDims& dims) {
    return "#define NN_IN " + std::to_string(dims.n_in) + "\n#define NN_HIDDEN " + std::to_string(dims.n_hidden) +
           "\n#define NN_HIDDEN_LAYERS " + std::to_string(dims.n_layers) + "\n#define NN_OUT " +
           std::to_string(dims.n_out) + "\n#define NN_LEVELS " + std::to_string(dims.levels) + "\n";
}

std::vector<std::string> neuralShaderFiles(const std::string& kernelFile) {
    return {"src/neural_types.h", "src/shaders/neural_common.metal", kernelFile};
}
