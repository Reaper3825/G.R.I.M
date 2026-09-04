#include "LoRATrainingCheckpoint.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include <flatbuffers/flatbuffers.h>

#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "grim_lora_training_checkpoint_generated.h"

namespace GRIM::Checkpoint {
namespace {

namespace fs = std::filesystem;

constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint64_t kMaximumCheckpointBytes = 16ull * 1024ull * 1024ull * 1024ull;

std::uint64_t rotateLeft(std::uint64_t value, int bits) {
    return (value << bits) | (value >> (64 - bits));
}

std::uint64_t readU64LE(const std::uint8_t* data) {
    std::uint64_t value = 0;
    for (int byte = 7; byte >= 0; --byte) {
        value = (value << 8) | data[byte];
    }
    return value;
}

std::uint32_t readU32LE(const std::uint8_t* data) {
    std::uint32_t value = 0;
    for (int byte = 3; byte >= 0; --byte) {
        value = (value << 8) | data[byte];
    }
    return value;
}

std::uint64_t xxhash64(const std::uint8_t* data, std::size_t size) {
    constexpr std::uint64_t p1 = 11400714785074694791ull;
    constexpr std::uint64_t p2 = 14029467366897019727ull;
    constexpr std::uint64_t p3 = 1609587929392839161ull;
    constexpr std::uint64_t p4 = 9650029242287828579ull;
    constexpr std::uint64_t p5 = 2870177450012600261ull;
    const std::uint8_t* cursor = data;
    const std::uint8_t* const end = data + size;
    std::uint64_t hash;
    if (size >= 32) {
        std::uint64_t v1 = p1 + p2;
        std::uint64_t v2 = p2;
        std::uint64_t v3 = 0;
        std::uint64_t v4 = 0 - p1;
        const std::uint8_t* const limit = end - 32;
        do {
            v1 = rotateLeft(v1 + readU64LE(cursor) * p2, 31) * p1; cursor += 8;
            v2 = rotateLeft(v2 + readU64LE(cursor) * p2, 31) * p1; cursor += 8;
            v3 = rotateLeft(v3 + readU64LE(cursor) * p2, 31) * p1; cursor += 8;
            v4 = rotateLeft(v4 + readU64LE(cursor) * p2, 31) * p1; cursor += 8;
        } while (cursor <= limit);
        hash = rotateLeft(v1, 1) + rotateLeft(v2, 7) + rotateLeft(v3, 12) + rotateLeft(v4, 18);
        for (const std::uint64_t value : {v1, v2, v3, v4}) {
            const std::uint64_t mixed = rotateLeft(value * p2, 31) * p1;
            hash = (hash ^ mixed) * p1 + p4;
        }
    } else {
        hash = p5;
    }
    hash += size;
    while (cursor + 8 <= end) {
        const std::uint64_t mixed = rotateLeft(readU64LE(cursor) * p2, 31) * p1;
        hash = rotateLeft(hash ^ mixed, 27) * p1 + p4;
        cursor += 8;
    }
    if (cursor + 4 <= end) {
        hash = rotateLeft(hash ^ (static_cast<std::uint64_t>(readU32LE(cursor)) * p1), 23) * p2 + p3;
        cursor += 4;
    }
    while (cursor < end) {
        hash = rotateLeft(hash ^ (static_cast<std::uint64_t>(*cursor++) * p5), 11) * p1;
    }
    hash ^= hash >> 33; hash *= p2; hash ^= hash >> 29; hash *= p3; hash ^= hash >> 32;
    return hash;
}

std::uint64_t checkedElementCount(
    const flatbuffers::Vector<std::uint32_t>* shape,
    const std::string& name)
{
    if (!shape || shape->size() != 2) {
        throw std::runtime_error(name + ": tensor shape must contain exactly two dimensions");
    }
    std::uint64_t count = 1;
    for (const std::uint32_t dimension : *shape) {
        if (dimension == 0) {
            throw std::runtime_error(name + ": tensor shape contains a zero dimension");
        }
        if (count > std::numeric_limits<std::uint64_t>::max() / dimension) {
            throw std::runtime_error(name + ": tensor element count overflows uint64");
        }
        count *= dimension;
    }
    return count;
}

LoRAHostTensorState decodeTensor(
    const GRIMLoRACheckpoint::TensorState* tensor,
    const std::vector<std::uint32_t>& expected_shape,
    const std::string& name)
{
    if (!tensor || tensor->storage_type() != GRIMLoRACheckpoint::TensorStorageType_FP32) {
        throw std::runtime_error(name + ": missing tensor or unsupported storage type");
    }
    const auto* shape = tensor->shape();
    const std::uint64_t element_count = checkedElementCount(shape, name);
    const std::vector<std::uint32_t> decoded_shape(shape->begin(), shape->end());
    if (decoded_shape != expected_shape) {
        throw std::runtime_error(name + ": tensor shape does not match its matrix-class contract");
    }
    if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::runtime_error(name + ": tensor byte count exceeds host capacity");
    }
    const std::size_t expected_bytes = static_cast<std::size_t>(element_count) * sizeof(float);
    const auto* data = tensor->data();
    if (!data || data->size() != expected_bytes) {
        throw std::runtime_error(name + ": FP32 payload size does not match tensor shape");
    }
    if (xxhash64(data->data(), data->size()) != tensor->data_xxhash64()) {
        throw std::runtime_error(name + ": tensor payload checksum mismatch");
    }
    LoRAHostTensorState decoded;
    decoded.shape = decoded_shape;
    decoded.values.resize(static_cast<std::size_t>(element_count));
    std::memcpy(decoded.values.data(), data->data(), expected_bytes);
    return decoded;
}

std::string requiredString(const flatbuffers::String* value, const char* name) {
    if (!value || value->size() == 0) {
        throw std::runtime_error(std::string("LoRA training checkpoint missing required string: ") + name);
    }
    return value->str();
}

std::array<std::uint8_t, 32> requiredSha256(
    const flatbuffers::Vector<std::uint8_t>* value,
    const char* name)
{
    if (!value || value->size() != 32) {
        throw std::runtime_error(std::string("LoRA training checkpoint requires 32-byte SHA-256: ") + name);
    }
    std::array<std::uint8_t, 32> result{};
    std::copy(value->begin(), value->end(), result.begin());
    return result;
}

bool isZeroDigest(const std::array<std::uint8_t, 32>& digest) {
    return std::all_of(digest.begin(), digest.end(), [](std::uint8_t value) {
        return value == 0;
    });
}

std::vector<std::uint32_t> expectedBaseShape(
    LoRAMatrixClass matrix_class,
    const LoRAArchitectureMetadata& architecture)
{
    switch (matrix_class) {
        case LoRAMatrixClass::QKV: return {architecture.qkv_dim, architecture.d_model};
        case LoRAMatrixClass::ATTENTION_OUTPUT: return {architecture.d_model, architecture.d_model};
        case LoRAMatrixClass::FFN_GATE:
        case LoRAMatrixClass::FFN_UP: return {architecture.d_model, architecture.d_ff};
        case LoRAMatrixClass::FFN_DOWN: return {architecture.d_ff, architecture.d_model};
    }
    throw std::runtime_error("LoRA training checkpoint contains an unknown matrix class");
}

LoRAMatrixClass decodeMatrixClass(GRIMLoRACheckpoint::MatrixClass value) {
    if (value < GRIMLoRACheckpoint::MatrixClass_QKV ||
        value > GRIMLoRACheckpoint::MatrixClass_FFN_DOWN) {
        throw std::runtime_error("LoRA training checkpoint contains an unknown matrix class");
    }
    return static_cast<LoRAMatrixClass>(value);
}

LoRAMatrixOrientation expectedOrientation(LoRAMatrixClass matrix_class) {
    if (matrix_class == LoRAMatrixClass::QKV ||
        matrix_class == LoRAMatrixClass::ATTENTION_OUTPUT) {
        return LoRAMatrixOrientation::TRANSPOSED_WEIGHT;
    }
    return LoRAMatrixOrientation::DIRECT_WEIGHT;
}

std::vector<std::uint8_t> readCheckpointBytes(const fs::path& path) {
    std::error_code error;
    const std::uintmax_t size = fs::file_size(path, error);
    if (error) {
        throw std::runtime_error("Cannot determine LoRA training checkpoint size: " + path.string() + ": " + error.message());
    }
    if (size == 0 || size > kMaximumCheckpointBytes || size > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("LoRA training checkpoint size is outside the supported range: " + path.string());
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot open LoRA training checkpoint: " + path.string());
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        throw std::runtime_error("Failed reading LoRA training checkpoint: " + path.string());
    }
    return bytes;
}

} // namespace

fs::path resolveLoRATrainingCheckpointPath(
    const Config::AiConfigSnapshot& config,
    const std::string& checkpoint_name)
{
    if (!config.model_config) {
        throw std::runtime_error("resolveLoRATrainingCheckpointPath: selected model.grimcfg is unavailable");
    }
    const fs::path requested(checkpoint_name);
    if (checkpoint_name.empty() || requested.is_absolute() || requested.has_parent_path() ||
        requested.filename() != requested || requested.extension() != ".grimlorackpt") {
        throw std::runtime_error(
            "resolveLoRATrainingCheckpointPath: checkpoint name must be a bare .grimlorackpt filename");
    }
    const fs::path model_directory = config.model_config->source_path.parent_path().lexically_normal();
    if (model_directory.empty()) {
        throw std::runtime_error("resolveLoRATrainingCheckpointPath: selected model directory is empty");
    }
    return (model_directory / "lora_checkpoints" / requested).lexically_normal();
}

LoRATrainingCheckpointSnapshot loadLoRATrainingCheckpoint(
    const Config::AiConfigSnapshot& config,
    const std::string& checkpoint_name,
    const std::string& expected_base_checkpoint_identity,
    const std::array<std::uint8_t, 32>& expected_base_checkpoint_sha256,
    const std::array<std::uint8_t, 32>& expected_training_config_sha256)
{
    if (expected_base_checkpoint_identity.empty()) {
        throw std::runtime_error("loadLoRATrainingCheckpoint: expected base checkpoint identity is empty");
    }
    if (isZeroDigest(expected_base_checkpoint_sha256) ||
        isZeroDigest(expected_training_config_sha256)) {
        throw std::runtime_error("loadLoRATrainingCheckpoint: expected SHA-256 digests must not be all zero");
    }
    const fs::path path = resolveLoRATrainingCheckpointPath(config, checkpoint_name);
    std::vector<std::uint8_t> bytes = readCheckpointBytes(path);
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    if (!GRIMLoRACheckpoint::VerifyLoRATrainingCheckpointBuffer(verifier) ||
        !GRIMLoRACheckpoint::LoRATrainingCheckpointBufferHasIdentifier(bytes.data())) {
        throw std::runtime_error("Invalid GLTC LoRA training checkpoint: " + path.string());
    }
    const auto* root = GRIMLoRACheckpoint::GetLoRATrainingCheckpoint(bytes.data());
    if (!root || root->format_version() != kFormatVersion) {
        throw std::runtime_error("Unsupported LoRA training checkpoint format version: " + path.string());
    }

    LoRATrainingCheckpointSnapshot result;
    result.source_path = fs::absolute(path).lexically_normal();
    result.checkpoint_id = requiredString(root->checkpoint_id(), "checkpoint_id");
    result.creation_timestamp_ms = root->creation_timestamp_ms();
    result.adapter_id = requiredString(root->adapter_id(), "adapter_id");
    if (root->has_parent_adapter_revision()) {
        result.parent_adapter_revision = root->parent_adapter_revision();
    }
    result.output_adapter_revision = root->output_adapter_revision();
    if ((result.parent_adapter_revision && *result.parent_adapter_revision == 0) ||
        result.output_adapter_revision == 0 ||
        (result.parent_adapter_revision && result.output_adapter_revision <= *result.parent_adapter_revision)) {
        throw std::runtime_error("LoRA training checkpoint has an invalid output adapter revision");
    }
    result.base_checkpoint_identity = requiredString(
        root->base_checkpoint_identity(), "base_checkpoint_identity");
    result.base_checkpoint_sha256 = requiredSha256(
        root->base_checkpoint_sha256(), "base_checkpoint_sha256");
    if (isZeroDigest(result.base_checkpoint_sha256) ||
        result.base_checkpoint_identity != expected_base_checkpoint_identity ||
        result.base_checkpoint_sha256 != expected_base_checkpoint_sha256) {
        throw std::runtime_error("LoRA training checkpoint does not match the currently loaded base checkpoint");
    }
    result.training_config_canonical = requiredString(
        root->training_config_canonical(), "training_config_canonical");
    result.training_config_sha256 = requiredSha256(
        root->training_config_sha256(), "training_config_sha256");
    if (isZeroDigest(result.training_config_sha256) ||
        result.training_config_sha256 != expected_training_config_sha256) {
        throw std::runtime_error("LoRA training checkpoint does not match the authoritative LoRA training configuration");
    }

    const auto* architecture = root->architecture();
    if (!config.model_config || !architecture) {
        throw std::runtime_error("LoRA training checkpoint architecture or selected model config is unavailable");
    }
    const auto& compiled = *config.model_config;
    result.architecture = {
        architecture->d_model(), architecture->num_layers(), architecture->num_heads(),
        architecture->num_kv_heads(), architecture->qkv_dim(), architecture->d_ff()};
    const LoRAArchitectureMetadata expected_architecture = {
        compiled.architecture.d_model, compiled.architecture.num_layers,
        compiled.architecture.num_heads, compiled.architecture.num_kv_heads,
        compiled.derived_architecture.qkv_dim, compiled.architecture.d_ff};
    if (result.architecture.d_model != expected_architecture.d_model ||
        result.architecture.num_layers != expected_architecture.num_layers ||
        result.architecture.num_heads != expected_architecture.num_heads ||
        result.architecture.num_kv_heads != expected_architecture.num_kv_heads ||
        result.architecture.qkv_dim != expected_architecture.qkv_dim ||
        result.architecture.d_ff != expected_architecture.d_ff) {
        throw std::runtime_error("LoRA training checkpoint architecture does not match selected model.grimcfg");
    }

    const auto* progress = root->progress();
    const auto* optimizer = root->optimizer();
    const auto* rng = root->rng();
    if (!progress || !optimizer || !rng || !root->targets() || root->targets()->size() == 0) {
        throw std::runtime_error("LoRA training checkpoint is missing required durable training state");
    }
    result.progress = {
        progress->global_step(), progress->epochs_completed(), progress->batch_cursor(),
        progress->accumulation_cursor(), progress->best_validation_loss(), {}};
    if (!std::isfinite(result.progress.best_validation_loss) ||
        !progress->data_order() || progress->data_order()->size() == 0) {
        throw std::runtime_error("LoRA training checkpoint has invalid progress or data-order state");
    }
    result.progress.data_order.assign(progress->data_order()->begin(), progress->data_order()->end());

    result.optimizer.family = requiredString(optimizer->family(), "optimizer.family");
    result.optimizer.canonical_config = requiredString(
        optimizer->canonical_config(), "optimizer.canonical_config");
    result.optimizer.optimizer_step = optimizer->optimizer_step();
    result.optimizer.learning_rate_lora = optimizer->learning_rate_lora();
    if (!std::isfinite(result.optimizer.learning_rate_lora) || result.optimizer.learning_rate_lora <= 0.0f ||
        !optimizer->lr_scheduler_state() || optimizer->lr_scheduler_state()->size() == 0 ||
        !optimizer->soft_restart_state() || optimizer->soft_restart_state()->size() == 0) {
        throw std::runtime_error("LoRA training checkpoint has incomplete optimizer or scheduler state");
    }
    result.optimizer.lr_scheduler_state.assign(
        optimizer->lr_scheduler_state()->begin(), optimizer->lr_scheduler_state()->end());
    result.optimizer.soft_restart_state.assign(
        optimizer->soft_restart_state()->begin(), optimizer->soft_restart_state()->end());

    result.rng = {rng->base_seed(), rng->data_seed(), rng->init_seed(), rng->cuda_seed(),
                  requiredString(rng->data_rng_state(), "rng.data_rng_state"), {}};
    if (!rng->cuda_rng_state() || rng->cuda_rng_state()->size() == 0) {
        throw std::runtime_error("LoRA training checkpoint is missing CUDA RNG state");
    }
    result.rng.cuda_rng_state.assign(rng->cuda_rng_state()->begin(), rng->cuda_rng_state()->end());

    std::unordered_set<std::string> identities;
    std::array<std::vector<bool>, 5> class_layers;
    std::array<std::optional<std::uint32_t>, 5> class_ranks;
    std::array<std::optional<float>, 5> class_alphas;
    for (auto& layers : class_layers) {
        layers.assign(result.architecture.num_layers, false);
    }
    result.targets.reserve(root->targets()->size());
    for (const auto* target : *root->targets()) {
        if (!target || target->layer_index() >= result.architecture.num_layers) {
            throw std::runtime_error("LoRA training checkpoint contains a null target or out-of-range layer");
        }
        LoRATargetTrainingState decoded;
        decoded.target_identity = requiredString(target->target_identity(), "target_identity");
        if (!identities.insert(decoded.target_identity).second) {
            throw std::runtime_error("LoRA training checkpoint contains duplicate target identity: " + decoded.target_identity);
        }
        decoded.layer_index = target->layer_index();
        decoded.matrix_class = decodeMatrixClass(target->matrix_class());
        auto& seen_layer = class_layers[static_cast<std::size_t>(decoded.matrix_class)][decoded.layer_index];
        if (seen_layer) {
            throw std::runtime_error("LoRA training checkpoint contains a duplicate layer/matrix-class target");
        }
        seen_layer = true;
        if (target->orientation() < GRIMLoRACheckpoint::MatrixOrientation_TRANSPOSED_WEIGHT ||
            target->orientation() > GRIMLoRACheckpoint::MatrixOrientation_DIRECT_WEIGHT) {
            throw std::runtime_error("LoRA training checkpoint target has an unknown matrix orientation");
        }
        decoded.orientation = static_cast<LoRAMatrixOrientation>(target->orientation());
        if (decoded.orientation != expectedOrientation(decoded.matrix_class)) {
            throw std::runtime_error("LoRA training checkpoint target has the wrong matrix orientation");
        }
        decoded.rank = target->rank();
        decoded.alpha = target->alpha();
        if (decoded.rank == 0 || !std::isfinite(decoded.alpha) || decoded.alpha <= 0.0f) {
            throw std::runtime_error("LoRA training checkpoint target has invalid rank or alpha");
        }
        const std::size_t class_index = static_cast<std::size_t>(decoded.matrix_class);
        if (!class_ranks[class_index]) {
            class_ranks[class_index] = decoded.rank;
            class_alphas[class_index] = decoded.alpha;
        } else if (*class_ranks[class_index] != decoded.rank ||
                   *class_alphas[class_index] != decoded.alpha) {
            throw std::runtime_error(
                "LoRA training checkpoint matrix class has inconsistent rank or alpha across layers");
        }
        const auto base_shape = expectedBaseShape(decoded.matrix_class, result.architecture);
        if (!target->base_shape()) {
            throw std::runtime_error("LoRA training checkpoint target is missing its base shape");
        }
        decoded.base_shape.assign(target->base_shape()->begin(), target->base_shape()->end());
        if (decoded.base_shape != base_shape || decoded.rank > std::min(base_shape[0], base_shape[1])) {
            throw std::runtime_error("LoRA training checkpoint target has invalid base shape or rank");
        }
        const std::vector<std::uint32_t> a_shape = {decoded.rank, base_shape[1]};
        const std::vector<std::uint32_t> b_shape = {base_shape[0], decoded.rank};
        const std::string prefix = "target " + decoded.target_identity;
        decoded.A = decodeTensor(target->a(), a_shape, prefix + ".A");
        decoded.B = decodeTensor(target->b(), b_shape, prefix + ".B");
        decoded.A_first_moment = decodeTensor(target->a_first_moment(), a_shape, prefix + ".A_first_moment");
        decoded.A_second_moment = decodeTensor(target->a_second_moment(), a_shape, prefix + ".A_second_moment");
        decoded.B_first_moment = decodeTensor(target->b_first_moment(), b_shape, prefix + ".B_first_moment");
        decoded.B_second_moment = decodeTensor(target->b_second_moment(), b_shape, prefix + ".B_second_moment");
        const bool gradients_required = result.progress.accumulation_cursor != 0;
        if ((target->a_gradient() != nullptr) != gradients_required ||
            (target->b_gradient() != nullptr) != gradients_required) {
            throw std::runtime_error(prefix + ": gradient presence does not match accumulation cursor");
        }
        if (gradients_required) {
            decoded.A_gradient = decodeTensor(target->a_gradient(), a_shape, prefix + ".A_gradient");
            decoded.B_gradient = decodeTensor(target->b_gradient(), b_shape, prefix + ".B_gradient");
        }
        result.targets.push_back(std::move(decoded));
    }
    for (std::size_t matrix_class = 0; matrix_class < class_layers.size(); ++matrix_class) {
        const auto& layers = class_layers[matrix_class];
        const bool enabled = std::any_of(layers.begin(), layers.end(), [](bool value) { return value; });
        if (enabled && !std::all_of(layers.begin(), layers.end(), [](bool value) { return value; })) {
            throw std::runtime_error("LoRA training checkpoint contains a partially populated matrix class");
        }
    }
    return result;
}

} // namespace GRIM::Checkpoint