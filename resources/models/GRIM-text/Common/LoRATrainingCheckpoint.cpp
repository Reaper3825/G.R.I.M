#include "LoRATrainingCheckpoint.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
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
        case LoRAMatrixClass::FFN_UP: return {architecture.d_ff, architecture.d_model};
        case LoRAMatrixClass::FFN_DOWN: return {architecture.d_model, architecture.d_ff};
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

flatbuffers::Offset<GRIMLoRACheckpoint::TensorState> encodeTensor(
    flatbuffers::FlatBufferBuilder& builder,
    const LoRAHostTensorState& tensor,
    const std::string& name)
{
    if (tensor.shape.size() != 2 || tensor.shape[0] == 0 || tensor.shape[1] == 0) {
        throw std::runtime_error(name + ": tensor shape must contain exactly two positive dimensions");
    }
    const std::uint64_t element_count =
        static_cast<std::uint64_t>(tensor.shape[0]) * tensor.shape[1];
    if (element_count != tensor.values.size()) {
        throw std::runtime_error(name + ": tensor value count does not match shape");
    }
    if (!std::all_of(tensor.values.begin(), tensor.values.end(), [](float value) {
            return std::isfinite(value);
        })) {
        throw std::runtime_error(name + ": tensor contains a non-finite value");
    }
    std::vector<std::uint8_t> bytes(tensor.values.size() * sizeof(float));
    std::memcpy(bytes.data(), tensor.values.data(), bytes.size());
    return GRIMLoRACheckpoint::CreateTensorStateDirect(
        builder,
        &tensor.shape,
        GRIMLoRACheckpoint::TensorStorageType_FP32,
        &bytes,
        xxhash64(bytes.data(), bytes.size()));
}

void writeCheckpointBytes(const fs::path& path, const std::uint8_t* data, std::size_t size) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Cannot create LoRA training checkpoint: " + path.string());
    }
    stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    stream.flush();
    if (!stream) {
        throw std::runtime_error("Failed writing LoRA training checkpoint: " + path.string());
    }
    stream.close();
    if (!stream) {
        throw std::runtime_error("Failed closing LoRA training checkpoint: " + path.string());
    }
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

std::optional<std::string> findLatestLoRATrainingCheckpointName(
    const Config::AiConfigSnapshot& config)
{
    if (!config.model_config) {
        throw std::runtime_error("findLatestLoRATrainingCheckpointName: selected model.grimcfg is unavailable");
    }
    const fs::path directory =
        config.model_config->source_path.parent_path().lexically_normal() /
        "lora_checkpoints";
    std::error_code error;
    if (!fs::exists(directory, error)) {
        if (error) {
            throw std::runtime_error(
                "findLatestLoRATrainingCheckpointName: cannot inspect model store: " +
                error.message());
        }
        return std::nullopt;
    }
    if (!fs::is_directory(directory, error) || error) {
        throw std::runtime_error(
            "findLatestLoRATrainingCheckpointName: model checkpoint store is not a directory: " +
            directory.string());
    }

    std::optional<fs::directory_entry> latest;
    for (const auto& entry : fs::directory_iterator(directory)) {
        const std::string filename = entry.path().filename().string();
        constexpr std::string_view temporary_suffix = ".writing.grimlorackpt";
        const bool is_temporary = filename.size() >= temporary_suffix.size() &&
            filename.compare(filename.size() - temporary_suffix.size(), temporary_suffix.size(), temporary_suffix) == 0;
        if (!entry.is_regular_file() || entry.path().extension() != ".grimlorackpt" || is_temporary) {
            continue;
        }
        if (entry.file_size() == 0) {
            throw std::runtime_error(
                "findLatestLoRATrainingCheckpointName: empty checkpoint in model store: " +
                entry.path().string());
        }
        if (!latest || entry.last_write_time() > latest->last_write_time() ||
            (entry.last_write_time() == latest->last_write_time() &&
             entry.path().filename().string() > latest->path().filename().string())) {
            latest = entry;
        }
    }
    if (!latest) {
        return std::nullopt;
    }
    return latest->path().filename().string();
}

LoRATrainingCheckpointSnapshot loadLoRATrainingCheckpoint(
    const Config::AiConfigSnapshot& config,
    const std::string& checkpoint_name,
    const std::string& expected_base_checkpoint_identity,
    const std::array<std::uint8_t, 32>& expected_base_checkpoint_sha256,
    const std::string& expected_training_config_canonical,
    const std::array<std::uint8_t, 32>& expected_training_config_sha256)
{
    if (expected_base_checkpoint_identity.empty()) {
        throw std::runtime_error("loadLoRATrainingCheckpoint: expected base checkpoint identity is empty");
    }
    if (isZeroDigest(expected_base_checkpoint_sha256) ||
        isZeroDigest(expected_training_config_sha256)) {
        throw std::runtime_error("loadLoRATrainingCheckpoint: expected SHA-256 digests must not be all zero");
    }
    if (expected_training_config_canonical.empty()) {
        throw std::runtime_error("loadLoRATrainingCheckpoint: expected canonical training config is empty");
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
    if (result.creation_timestamp_ms == 0) {
        throw std::runtime_error("LoRA training checkpoint creation timestamp is missing");
    }
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
    if (result.training_config_canonical != expected_training_config_canonical ||
        isZeroDigest(result.training_config_sha256) ||
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
    std::array<std::vector<std::uint8_t>, 5> class_layers;
    std::array<std::optional<std::uint32_t>, 5> class_ranks;
    std::array<std::optional<float>, 5> class_alphas;
    for (auto& layers : class_layers) {
        layers.assign(result.architecture.num_layers, 0);
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
        seen_layer = 1;
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

fs::path saveLoRATrainingCheckpoint(
    const Config::AiConfigSnapshot& config,
    const std::string& checkpoint_name,
    const LoRATrainingCheckpointSnapshot& snapshot)
{
    const fs::path destination = resolveLoRATrainingCheckpointPath(config, checkpoint_name);
    if (snapshot.checkpoint_id.empty() || snapshot.creation_timestamp_ms == 0 ||
        snapshot.adapter_id.empty() || snapshot.output_adapter_revision == 0 ||
        snapshot.base_checkpoint_identity.empty() || snapshot.training_config_canonical.empty()) {
        throw std::runtime_error("saveLoRATrainingCheckpoint: required checkpoint metadata is missing");
    }
    if ((snapshot.parent_adapter_revision && *snapshot.parent_adapter_revision == 0) ||
        (snapshot.parent_adapter_revision &&
         snapshot.output_adapter_revision <= *snapshot.parent_adapter_revision)) {
        throw std::runtime_error("saveLoRATrainingCheckpoint: invalid adapter revision lineage");
    }
    if (isZeroDigest(snapshot.base_checkpoint_sha256) ||
        isZeroDigest(snapshot.training_config_sha256)) {
        throw std::runtime_error("saveLoRATrainingCheckpoint: SHA-256 digests must not be all zero");
    }
    if (snapshot.targets.empty()) {
        throw std::runtime_error("saveLoRATrainingCheckpoint: target inventory is empty");
    }

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<GRIMLoRACheckpoint::AdapterTargetState>> targets;
    targets.reserve(snapshot.targets.size());
    for (const auto& target : snapshot.targets) {
        const std::string prefix = "target " + target.target_identity;
        if (target.target_identity.empty()) {
            throw std::runtime_error("saveLoRATrainingCheckpoint: target identity is empty");
        }
        const bool gradients_required = snapshot.progress.accumulation_cursor != 0;
        if (target.A_gradient.has_value() != gradients_required ||
            target.B_gradient.has_value() != gradients_required) {
            throw std::runtime_error(prefix + ": gradient presence does not match accumulation cursor");
        }
        const auto a = encodeTensor(builder, target.A, prefix + ".A");
        const auto b = encodeTensor(builder, target.B, prefix + ".B");
        flatbuffers::Offset<GRIMLoRACheckpoint::TensorState> a_gradient;
        flatbuffers::Offset<GRIMLoRACheckpoint::TensorState> b_gradient;
        if (gradients_required) {
            a_gradient = encodeTensor(builder, *target.A_gradient, prefix + ".A_gradient");
            b_gradient = encodeTensor(builder, *target.B_gradient, prefix + ".B_gradient");
        }
        const auto a_first_moment = encodeTensor(
            builder, target.A_first_moment, prefix + ".A_first_moment");
        const auto a_second_moment = encodeTensor(
            builder, target.A_second_moment, prefix + ".A_second_moment");
        const auto b_first_moment = encodeTensor(
            builder, target.B_first_moment, prefix + ".B_first_moment");
        const auto b_second_moment = encodeTensor(
            builder, target.B_second_moment, prefix + ".B_second_moment");
        targets.push_back(GRIMLoRACheckpoint::CreateAdapterTargetStateDirect(
            builder,
            target.target_identity.c_str(),
            target.layer_index,
            static_cast<GRIMLoRACheckpoint::MatrixClass>(target.matrix_class),
            &target.base_shape,
            static_cast<GRIMLoRACheckpoint::MatrixOrientation>(target.orientation),
            target.rank,
            target.alpha,
            a,
            b,
            a_gradient,
            b_gradient,
            a_first_moment,
            a_second_moment,
            b_first_moment,
            b_second_moment));
    }

    const auto architecture = GRIMLoRACheckpoint::CreateArchitectureMetadata(
        builder,
        snapshot.architecture.d_model,
        snapshot.architecture.num_layers,
        snapshot.architecture.num_heads,
        snapshot.architecture.num_kv_heads,
        snapshot.architecture.qkv_dim,
        snapshot.architecture.d_ff);
    const auto optimizer = GRIMLoRACheckpoint::CreateOptimizerStateDirect(
        builder,
        snapshot.optimizer.family.c_str(),
        snapshot.optimizer.canonical_config.c_str(),
        snapshot.optimizer.optimizer_step,
        snapshot.optimizer.learning_rate_lora,
        &snapshot.optimizer.lr_scheduler_state,
        &snapshot.optimizer.soft_restart_state);
    const auto progress = GRIMLoRACheckpoint::CreateTrainingProgressStateDirect(
        builder,
        snapshot.progress.global_step,
        snapshot.progress.epochs_completed,
        snapshot.progress.batch_cursor,
        snapshot.progress.accumulation_cursor,
        snapshot.progress.best_validation_loss,
        &snapshot.progress.data_order);
    const auto rng = GRIMLoRACheckpoint::CreateRngStateDirect(
        builder,
        snapshot.rng.base_seed,
        snapshot.rng.data_seed,
        snapshot.rng.init_seed,
        snapshot.rng.cuda_seed,
        snapshot.rng.data_rng_state.c_str(),
        &snapshot.rng.cuda_rng_state);
    const std::vector<std::uint8_t> base_digest(
        snapshot.base_checkpoint_sha256.begin(), snapshot.base_checkpoint_sha256.end());
    const std::vector<std::uint8_t> config_digest(
        snapshot.training_config_sha256.begin(), snapshot.training_config_sha256.end());
    const auto root = GRIMLoRACheckpoint::CreateLoRATrainingCheckpointDirect(
        builder,
        kFormatVersion,
        snapshot.checkpoint_id.c_str(),
        snapshot.creation_timestamp_ms,
        snapshot.adapter_id.c_str(),
        snapshot.parent_adapter_revision.has_value(),
        snapshot.parent_adapter_revision.value_or(0),
        snapshot.output_adapter_revision,
        snapshot.base_checkpoint_identity.c_str(),
        &base_digest,
        architecture,
        snapshot.training_config_canonical.c_str(),
        &config_digest,
        &targets,
        optimizer,
        progress,
        rng);
    GRIMLoRACheckpoint::FinishLoRATrainingCheckpointBuffer(builder, root);
    if (builder.GetSize() == 0 || builder.GetSize() > kMaximumCheckpointBytes) {
        throw std::runtime_error("saveLoRATrainingCheckpoint: encoded checkpoint size is outside the supported range");
    }

    std::error_code error;
    if (fs::exists(destination, error) || error) {
        throw std::runtime_error(
            "saveLoRATrainingCheckpoint: immutable destination already exists or cannot be inspected: " +
            destination.string() + (error ? ": " + error.message() : ""));
    }
    fs::create_directories(destination.parent_path(), error);
    if (error) {
        throw std::runtime_error(
            "saveLoRATrainingCheckpoint: cannot create checkpoint directory: " + error.message());
    }
    const std::string temporary_name = checkpoint_name + ".writing.grimlorackpt";
    const fs::path temporary = resolveLoRATrainingCheckpointPath(config, temporary_name);
    if (fs::exists(temporary, error) || error) {
        throw std::runtime_error(
            "saveLoRATrainingCheckpoint: temporary checkpoint already exists or cannot be inspected: " +
            temporary.string() + (error ? ": " + error.message() : ""));
    }

    try {
        writeCheckpointBytes(temporary, builder.GetBufferPointer(), builder.GetSize());
        (void)loadLoRATrainingCheckpoint(
            config,
            temporary_name,
            snapshot.base_checkpoint_identity,
            snapshot.base_checkpoint_sha256,
            snapshot.training_config_canonical,
            snapshot.training_config_sha256);
        fs::rename(temporary, destination, error);
        if (error) {
            throw std::runtime_error(
                "saveLoRATrainingCheckpoint: atomic publish failed: " + error.message());
        }
    } catch (...) {
        std::error_code cleanup_error;
        fs::remove(temporary, cleanup_error);
        throw;
    }
    return fs::absolute(destination).lexically_normal();
}

} // namespace GRIM::Checkpoint