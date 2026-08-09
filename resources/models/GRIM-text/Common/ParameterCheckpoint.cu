#include "ParameterCheckpoint.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <flatbuffers/flatbuffers.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include "../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../Shared/LogRecorder/LogRecorder.hpp"
#include "../training/Phases/Startup/Model/ParameterRegistry.hpp"
#include "grim_parameter_checkpoint_generated.h"

namespace GRIM::Checkpoint {
namespace {

namespace fs = std::filesystem;
using Logging::EmitModuleError;
using Logging::EmitModuleInfo;
using Logging::ModuleId;

constexpr std::uint32_t kParameterCheckpointVersion = 1;
constexpr std::size_t kStoredElementBytes = sizeof(float);
static_assert(sizeof(float) == 4, "GRIM checkpoint FP32 storage requires 4-byte float");

struct CompatibilityFactValue {
    std::string name;
    std::string value;
};

struct HostParameterEntry {
    std::string name;
    GRIMCheckpoint::ParameterGroupType group_type;
    GRIMCheckpoint::ParameterStatsBucket stats_bucket;
    std::int32_t layer_index = -1;
    GRIMCheckpoint::TensorLayout layout;
    std::vector<std::uint32_t> shape;
    GRIMCheckpoint::TensorStorageType storage_type =
        GRIMCheckpoint::TensorStorageType_FP32;
    GRIMCheckpoint::ParameterComputePrecision compute_precision;
    std::uint64_t element_count = 0;
    std::uint64_t payload_offset = 0;
    std::uint64_t payload_size = 0;
    std::uint64_t payload_xxhash64 = 0;
};

std::uint64_t checkedAdd(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* context)
{
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::runtime_error(
            std::string("ParameterCheckpoint byte-count overflow while computing ") +
            context);
    }
    return lhs + rhs;
}

std::uint64_t checkedMultiply(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* context)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::runtime_error(
            std::string("ParameterCheckpoint byte-count overflow while computing ") +
            context);
    }
    return lhs * rhs;
}

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

std::uint64_t xxhash64(
    const std::uint8_t* data,
    std::size_t size,
    std::uint64_t seed = 0)
{
    constexpr std::uint64_t prime1 = 11400714785074694791ULL;
    constexpr std::uint64_t prime2 = 14029467366897019727ULL;
    constexpr std::uint64_t prime3 = 1609587929392839161ULL;
    constexpr std::uint64_t prime4 = 9650029242287828579ULL;
    constexpr std::uint64_t prime5 = 2870177450012600261ULL;

    auto round = [=](std::uint64_t accumulator, std::uint64_t input) {
        accumulator += input * prime2;
        accumulator = rotateLeft(accumulator, 31);
        return accumulator * prime1;
    };
    auto mergeRound = [&](std::uint64_t accumulator, std::uint64_t value) {
        accumulator ^= round(0, value);
        return accumulator * prime1 + prime4;
    };

    const std::uint8_t* cursor = data;
    const std::uint8_t* const end = data + size;
    std::uint64_t hash = 0;

    if (size >= 32) {
        std::uint64_t v1 = seed + prime1 + prime2;
        std::uint64_t v2 = seed + prime2;
        std::uint64_t v3 = seed;
        std::uint64_t v4 = seed - prime1;
        const std::uint8_t* const block_end = end - 32;
        do {
            v1 = round(v1, readU64LE(cursor)); cursor += 8;
            v2 = round(v2, readU64LE(cursor)); cursor += 8;
            v3 = round(v3, readU64LE(cursor)); cursor += 8;
            v4 = round(v4, readU64LE(cursor)); cursor += 8;
        } while (cursor <= block_end);
        hash = rotateLeft(v1, 1) + rotateLeft(v2, 7) +
               rotateLeft(v3, 12) + rotateLeft(v4, 18);
        hash = mergeRound(hash, v1);
        hash = mergeRound(hash, v2);
        hash = mergeRound(hash, v3);
        hash = mergeRound(hash, v4);
    } else {
        hash = seed + prime5;
    }

    hash += static_cast<std::uint64_t>(size);
    while (static_cast<std::size_t>(end - cursor) >= 8) {
        const std::uint64_t lane = round(0, readU64LE(cursor));
        hash ^= lane;
        hash = rotateLeft(hash, 27) * prime1 + prime4;
        cursor += 8;
    }
    if (static_cast<std::size_t>(end - cursor) >= 4) {
        hash ^= static_cast<std::uint64_t>(readU32LE(cursor)) * prime1;
        hash = rotateLeft(hash, 23) * prime2 + prime3;
        cursor += 4;
    }
    while (cursor < end) {
        hash ^= static_cast<std::uint64_t>(*cursor) * prime5;
        hash = rotateLeft(hash, 11) * prime1;
        ++cursor;
    }

    hash ^= hash >> 33;
    hash *= prime2;
    hash ^= hash >> 29;
    hash *= prime3;
    hash ^= hash >> 32;
    return hash;
}

std::uint64_t xxhash64(const std::vector<std::uint8_t>& bytes) {
    return xxhash64(bytes.data(), bytes.size());
}

template <typename UIntT>
void appendUnsignedLE(std::vector<std::uint8_t>& bytes, UIntT value) {
    static_assert(std::is_unsigned<UIntT>::value, "UIntT must be unsigned");
    for (std::size_t byte = 0; byte < sizeof(UIntT); ++byte) {
        bytes.push_back(static_cast<std::uint8_t>(value & 0xFFu));
        value >>= 8;
    }
}

void appendString(std::vector<std::uint8_t>& bytes, const std::string& value) {
    if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("ParameterCheckpoint canonical string exceeds uint32 length");
    }
    appendUnsignedLE(bytes, static_cast<std::uint32_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
}

std::string canonicalFloat(float value) {
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    return stream.str();
}

std::vector<CompatibilityFactValue> buildCompatibilityFacts(
    const Config::AiConfigSnapshot& config)
{
    using HyperParameters::snapshotTrainingConfigField;
    std::vector<CompatibilityFactValue> facts;
    auto addBool = [&](const char* name) {
        facts.push_back({name, snapshotTrainingConfigField<bool>(config, name) ? "true" : "false"});
    };
    auto addInt = [&](const char* name) {
        facts.push_back({name, std::to_string(snapshotTrainingConfigField<int>(config, name))});
    };
    auto addFloat = [&](const char* name) {
        facts.push_back({name, canonicalFloat(snapshotTrainingConfigField<float>(config, name))});
    };

    addInt("vocab_size");
    addInt("d_model");
    addInt("num_layers");
    addInt("num_heads");
    addInt("num_kv_heads");
    addInt("d_ff");
    addInt("max_seq_len");
    addInt("rope_base_seq_len");
    addInt("alibi_min_locality_distance");
    addInt("execution_block_layer");
    addInt("execution_block_num_ops");
    addInt("execution_block_num_slots");
    addInt("execution_block_num_scratch_slots");
    addInt("execution_block_num_steps");
    addInt("execution_block_result_slot_mode");
    addInt("execution_block_result_slot_index");
    addInt("number_encoder_max_digit_slots");
    addInt("number_encoder_max_abs_pow10");

    addFloat("embedding_scale");
    addFloat("rms_epsilon");
    addFloat("rope_theta");
    addFloat("rope_scaling");
    addFloat("alibi_slope_exponent");
    addFloat("alibi_max_bias");
    addFloat("lm_head_mlp_alpha");

    addBool("tie_embeddings");
    addBool("causal_mask");
    addBool("use_pre_norm");
    addBool("fuse_qkv");
    addBool("qk_norm_enabled");
    addBool("attention_off_by_one");
    addBool("attention_residual_gate_enabled");
    addBool("use_layer_scale");
    addBool("use_atom_data");
    addBool("execution_block_enabled");
    addBool("number_encoder_enabled");
    addBool("selector_enabled");
    addBool("slot_seed_encoder_enabled");
    addBool("lm_head_center_hidden_states");
    addBool("project_out_pc1");
    addBool("center_logits");
    addBool("center_encoder_residuals");
    addBool("lm_head_mlp_enabled");

    const auto positional_encoding =
        snapshotTrainingConfigField<HyperParameters::PositionalEncodingType>(
            config,
            "positional_encoding");
    facts.push_back({
        "positional_encoding",
        HyperParameters::positionalEncodingTypeToJsonString(positional_encoding)});

    std::sort(
        facts.begin(),
        facts.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.name < rhs.name; });
    for (std::size_t i = 1; i < facts.size(); ++i) {
        if (facts[i - 1].name == facts[i].name) {
            throw std::runtime_error(
                "Duplicate checkpoint compatibility fact: " + facts[i].name);
        }
    }
    return facts;
}

std::uint64_t compatibilityChecksum(
    const std::vector<CompatibilityFactValue>& facts)
{
    std::vector<std::uint8_t> canonical;
    for (const auto& fact : facts) {
        appendString(canonical, fact.name);
        appendString(canonical, fact.value);
    }
    return xxhash64(canonical);
}

GRIMCheckpoint::ParameterGroupType checkpointGroupType(ParamGroupType type) {
    switch (type) {
        case ParamGroupType::EMBEDDING: return GRIMCheckpoint::ParameterGroupType_EMBEDDING;
        case ParamGroupType::LM_HEAD: return GRIMCheckpoint::ParameterGroupType_LM_HEAD;
        case ParamGroupType::ATTENTION: return GRIMCheckpoint::ParameterGroupType_ATTENTION;
        case ParamGroupType::FFN: return GRIMCheckpoint::ParameterGroupType_FFN;
        case ParamGroupType::RMSNORM: return GRIMCheckpoint::ParameterGroupType_RMSNORM;
        case ParamGroupType::EXECUTION_BLOCK: return GRIMCheckpoint::ParameterGroupType_EXECUTION_BLOCK;
        case ParamGroupType::NUMBER_ENCODER: return GRIMCheckpoint::ParameterGroupType_NUMBER_ENCODER;
        case ParamGroupType::ARG_SELECTOR: return GRIMCheckpoint::ParameterGroupType_ARG_SELECTOR;
        case ParamGroupType::SLOT_SEED_ENCODER: return GRIMCheckpoint::ParameterGroupType_SLOT_SEED_ENCODER;
        case ParamGroupType::COUNT: break;
    }
    throw std::runtime_error("ParameterCheckpoint encountered ParamGroupType::COUNT");
}

GRIMCheckpoint::ParameterStatsBucket checkpointStatsBucket(ParamStatsBucket bucket) {
    switch (bucket) {
        case ParamStatsBucket::EMBEDDING: return GRIMCheckpoint::ParameterStatsBucket_EMBEDDING;
        case ParamStatsBucket::ENCODER: return GRIMCheckpoint::ParameterStatsBucket_ENCODER;
        case ParamStatsBucket::LM_HEAD: return GRIMCheckpoint::ParameterStatsBucket_LM_HEAD;
        case ParamStatsBucket::COUNT: break;
    }
    throw std::runtime_error("ParameterCheckpoint encountered ParamStatsBucket::COUNT");
}

GRIMCheckpoint::TensorLayout checkpointLayout(TensorContract::Layout layout) {
    switch (layout) {
        case TensorContract::Layout::BSM: return GRIMCheckpoint::TensorLayout_BSM;
        case TensorContract::Layout::QKV_FUSED: return GRIMCheckpoint::TensorLayout_QKV_FUSED;
        case TensorContract::Layout::LOGITS: return GRIMCheckpoint::TensorLayout_LOGITS;
        case TensorContract::Layout::BHSD: return GRIMCheckpoint::TensorLayout_BHSD;
        case TensorContract::Layout::BSHD: return GRIMCheckpoint::TensorLayout_BSHD;
        case TensorContract::Layout::UNKNOWN: break;
    }
    throw std::runtime_error("ParameterCheckpoint encountered TensorContract::Layout::UNKNOWN");
}

GRIMCheckpoint::ParameterComputePrecision checkpointComputePrecision(
    HyperParameters::ParameterGroupPrecision precision)
{
    switch (precision) {
        case HyperParameters::ParameterGroupPrecision::FP32:
            return GRIMCheckpoint::ParameterComputePrecision_FP32;
        case HyperParameters::ParameterGroupPrecision::BF16_COMPUTE:
            return GRIMCheckpoint::ParameterComputePrecision_BF16_COMPUTE;
        case HyperParameters::ParameterGroupPrecision::UNSPECIFIED:
            break;
    }
    throw std::runtime_error(
        "ParameterCheckpoint encountered unspecified parameter compute precision");
}

std::vector<std::uint32_t> checkpointShape(const Tensor& tensor) {
    tensor.shape.require("ParameterCheckpoint tensor shape");
    if (tensor.shape.is_2d_layout()) {
        const auto dims = tensor.shape.as_2d();
        return {
            static_cast<std::uint32_t>(dims.rows),
            static_cast<std::uint32_t>(dims.cols)};
    }
    if (tensor.shape.is_4d()) {
        const auto dims = tensor.shape.as_4d();
        return {
            static_cast<std::uint32_t>(dims.batch),
            static_cast<std::uint32_t>(dims.heads),
            static_cast<std::uint32_t>(dims.seq),
            static_cast<std::uint32_t>(dims.head_dim)};
    }
    throw std::runtime_error("ParameterCheckpoint tensor has unsupported shape layout");
}

HostParameterEntry makeHostEntry(
    const ParameterGroup& group,
    std::uint64_t payload_offset)
{
    if (group.name.empty()) {
        throw std::runtime_error("ParameterCheckpoint encountered an empty parameter group name");
    }
    if (!group.tensor) {
        throw std::runtime_error(
            "ParameterCheckpoint parameter group has NULL tensor: " + group.name);
    }
    const Tensor& tensor = group.tensor->require("ParameterCheckpoint parameter tensor");

    HostParameterEntry entry;
    entry.name = group.name;
    entry.group_type = checkpointGroupType(group.type);
    entry.stats_bucket = checkpointStatsBucket(group.stats_bucket);
    entry.layer_index = group.layer_index;
    entry.layout = checkpointLayout(tensor.layout());
    entry.shape = checkpointShape(tensor);
    entry.compute_precision = checkpointComputePrecision(group.parameter_precision);
    entry.element_count = static_cast<std::uint64_t>(tensor.numel());
    entry.payload_offset = payload_offset;
    entry.payload_size = checkedMultiply(
        entry.element_count,
        kStoredElementBytes,
        "parameter payload");
    return entry;
}

std::uint64_t manifestChecksum(const std::vector<HostParameterEntry>& entries) {
    std::vector<std::uint8_t> canonical;
    for (const auto& entry : entries) {
        appendString(canonical, entry.name);
        appendUnsignedLE(canonical, static_cast<std::uint8_t>(entry.group_type));
        appendUnsignedLE(canonical, static_cast<std::uint8_t>(entry.stats_bucket));
        appendUnsignedLE(
            canonical,
            static_cast<std::uint32_t>(entry.layer_index));
        appendUnsignedLE(canonical, static_cast<std::uint8_t>(entry.layout));
        appendUnsignedLE(canonical, static_cast<std::uint8_t>(entry.shape.size()));
        for (const std::uint32_t dimension : entry.shape) {
            appendUnsignedLE(canonical, dimension);
        }
        appendUnsignedLE(canonical, static_cast<std::uint8_t>(entry.storage_type));
        appendUnsignedLE(canonical, static_cast<std::uint8_t>(entry.compute_precision));
        appendUnsignedLE(canonical, entry.element_count);
    }
    return xxhash64(canonical);
}

std::string makeCheckpointId() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const std::uint64_t timestamp = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
    std::random_device random;
    const std::uint64_t random_bits =
        (static_cast<std::uint64_t>(random()) << 32) ^ random();
    std::ostringstream id;
    id << std::hex << std::setfill('0')
       << std::setw(16) << timestamp << '-'
       << std::setw(16) << random_bits;
    return id.str();
}

void requireCheckpointExtension(const std::string& path, const char* caller) {
    if (path.empty()) {
        throw std::runtime_error(std::string(caller) + ": checkpoint path is empty");
    }
    if (fs::path(path).extension() != ".grimckpt") {
        throw std::runtime_error(
            std::string(caller) +
            ": registry checkpoints must use the .grimckpt extension: " + path);
    }
}

std::vector<std::uint8_t> readFile(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Failed to open parameter checkpoint: " + path);
    }
    const std::streampos raw_size = input.tellg();
    if (raw_size <= 0) {
        throw std::runtime_error("Parameter checkpoint is empty or unreadable: " + path);
    }
    const auto size = static_cast<std::uint64_t>(
        static_cast<std::streamoff>(raw_size));
    if (size > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("Parameter checkpoint exceeds host address range: " + path);
    }
    if (size > static_cast<std::uint64_t>(
            std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Parameter checkpoint exceeds stream read range: " + path);
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    input.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!input || static_cast<std::size_t>(input.gcount()) != bytes.size()) {
        throw std::runtime_error("Failed to read complete parameter checkpoint: " + path);
    }
    return bytes;
}

void flushFileToDisk(const fs::path& path) {
#ifdef _WIN32
    const HANDLE handle = CreateFileW(
        path.c_str(),
        GENERIC_WRITE,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error(
            "Failed to reopen temporary checkpoint for flush: " + path.string());
    }
    const BOOL flushed = FlushFileBuffers(handle);
    CloseHandle(handle);
    if (!flushed) {
        throw std::runtime_error(
            "FlushFileBuffers failed for temporary checkpoint: " + path.string());
    }
#else
    const int descriptor = open(path.c_str(), O_RDWR);
    if (descriptor < 0) {
        throw std::runtime_error(
            "Failed to reopen temporary checkpoint for fsync: " + path.string());
    }
    const int result = fsync(descriptor);
    close(descriptor);
    if (result != 0) {
        throw std::runtime_error(
            "fsync failed for temporary checkpoint: " + path.string());
    }
#endif
}

void replaceFileAtomically(const fs::path& temporary, const fs::path& final_path) {
#ifdef _WIN32
    if (!MoveFileExW(
            temporary.c_str(),
            final_path.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        throw std::runtime_error(
            "Atomic checkpoint rename failed from " + temporary.string() +
            " to " + final_path.string());
    }
#else
    if (std::rename(temporary.c_str(), final_path.c_str()) != 0) {
        throw std::runtime_error(
            "Atomic checkpoint rename failed from " + temporary.string() +
            " to " + final_path.string());
    }
#endif
}

void writeAtomically(const std::string& path, const std::uint8_t* data, std::size_t size) {
    if (size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error(
            "Parameter checkpoint exceeds std::streamsize write range");
    }
    const fs::path final_path(path);
    const fs::path parent = final_path.parent_path();
    if (!parent.empty() && !fs::is_directory(parent)) {
        throw std::runtime_error(
            "Parameter checkpoint parent directory does not exist: " + parent.string());
    }
    const fs::path temporary =
        final_path.string() + "." + makeCheckpointId() + ".tmp";
    try {
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output) {
                throw std::runtime_error(
                    "Failed to open temporary parameter checkpoint: " + temporary.string());
            }
            output.write(
                reinterpret_cast<const char*>(data),
                static_cast<std::streamsize>(size));
            output.flush();
            if (!output) {
                throw std::runtime_error(
                    "Failed to write complete temporary parameter checkpoint: " +
                    temporary.string());
            }
        }
        flushFileToDisk(temporary);
        const std::vector<std::uint8_t> reread = readFile(temporary.string());
        if (reread.size() != size ||
            xxhash64(reread) != xxhash64(data, size)) {
            throw std::runtime_error(
                "Temporary parameter checkpoint failed post-write verification: " +
                temporary.string());
        }
        replaceFileAtomically(temporary, final_path);
    } catch (...) {
        std::error_code cleanup_error;
        fs::remove(temporary, cleanup_error);
        throw;
    }
}

std::vector<HostParameterEntry> buildHostManifest(
    const std::vector<ParameterGroup>& groups)
{
    std::vector<HostParameterEntry> entries;
    entries.reserve(groups.size());
    std::unordered_set<std::string> names;
    std::uint64_t offset = 0;
    for (const auto& group : groups) {
        HostParameterEntry entry = makeHostEntry(group, offset);
        if (!names.insert(entry.name).second) {
            throw std::runtime_error(
                "Duplicate parameter group name in registry checkpoint: " + entry.name);
        }
        offset = checkedAdd(offset, entry.payload_size, "checkpoint payload size");
        entries.push_back(std::move(entry));
    }
    if (offset > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("Parameter checkpoint payload exceeds host address range");
    }
    return entries;
}

std::uint64_t totalPayloadBytes(const std::vector<HostParameterEntry>& entries) {
    if (entries.empty()) return 0;
    return checkedAdd(
        entries.back().payload_offset,
        entries.back().payload_size,
        "checkpoint payload size");
}

bool compatibilityFactsMatch(
    const flatbuffers::Vector<flatbuffers::Offset<GRIMCheckpoint::CompatibilityFact>>* stored,
    const std::vector<CompatibilityFactValue>& expected,
    std::string& mismatch)
{
    if (!stored || stored->size() != expected.size()) {
        mismatch = "compatibility fact count mismatch: checkpoint=" +
            std::to_string(stored ? stored->size() : 0) +
            " current=" + std::to_string(expected.size());
        return false;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto* fact = stored->Get(static_cast<flatbuffers::uoffset_t>(i));
        if (!fact || !fact->name() || !fact->canonical_value()) {
            mismatch = "invalid compatibility fact at index " + std::to_string(i);
            return false;
        }
        const std::string stored_name = fact->name()->str();
        const std::string stored_value = fact->canonical_value()->str();
        if (stored_name != expected[i].name || stored_value != expected[i].value) {
            mismatch = "compatibility fact mismatch at index " + std::to_string(i) +
                ": checkpoint=" + stored_name + "=" + stored_value +
                " current=" + expected[i].name + "=" + expected[i].value;
            return false;
        }
    }
    return true;
}

bool shapeMatches(
    const flatbuffers::Vector<std::uint32_t>* stored,
    const std::vector<std::uint32_t>& expected)
{
    if (!stored || stored->size() != expected.size()) return false;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (stored->Get(static_cast<flatbuffers::uoffset_t>(i)) != expected[i]) {
            return false;
        }
    }
    return true;
}

} // namespace

bool saveParameterCheckpoint(
    const Config::AiConfigSnapshot& config,
    const ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    cudaStream_t stream,
    const std::string& path)
{
    try {
        static std::mutex save_mutex;
        const std::lock_guard<std::mutex> save_lock(save_mutex);
        requireCheckpointExtension(path, "saveParameterCheckpoint");
        if (!stream) {
            throw std::runtime_error("saveParameterCheckpoint: stream is NULL");
        }
        const auto& groups =
            parameter_registry.requireParameterGroups("saveParameterCheckpoint");
        std::vector<HostParameterEntry> entries = buildHostManifest(groups);
        const auto facts = buildCompatibilityFacts(config);

        const std::uint64_t payload_bytes = totalPayloadBytes(entries);
        std::vector<std::uint8_t> payload(static_cast<std::size_t>(payload_bytes));
        const cudaError_t sync_error = cudaStreamSynchronize(stream);
        if (sync_error != cudaSuccess) {
            throw std::runtime_error(
                "saveParameterCheckpoint: pre-save stream synchronization failed: " +
                std::string(cudaGetErrorString(sync_error)));
        }

        for (std::size_t i = 0; i < groups.size(); ++i) {
            auto& entry = entries[i];
            std::uint8_t* destination =
                payload.data() + static_cast<std::size_t>(entry.payload_offset);
            const cudaError_t copy_error = cudaMemcpy(
                destination,
                groups[i].tensor->data,
                static_cast<std::size_t>(entry.payload_size),
                cudaMemcpyDeviceToHost);
            if (copy_error != cudaSuccess) {
                throw std::runtime_error(
                    "saveParameterCheckpoint: D2H copy failed for " + entry.name +
                    ": " + cudaGetErrorString(copy_error));
            }
            entry.payload_xxhash64 = xxhash64(
                destination,
                static_cast<std::size_t>(entry.payload_size));
        }

        flatbuffers::FlatBufferBuilder builder;
        std::vector<flatbuffers::Offset<GRIMCheckpoint::CompatibilityFact>> fact_offsets;
        fact_offsets.reserve(facts.size());
        for (const auto& fact : facts) {
            fact_offsets.push_back(GRIMCheckpoint::CreateCompatibilityFact(
                builder,
                builder.CreateString(fact.name),
                builder.CreateString(fact.value)));
        }

        std::vector<flatbuffers::Offset<GRIMCheckpoint::ParameterEntry>> entry_offsets;
        entry_offsets.reserve(entries.size());
        for (const auto& entry : entries) {
            entry_offsets.push_back(GRIMCheckpoint::CreateParameterEntry(
                builder,
                builder.CreateString(entry.name),
                entry.group_type,
                entry.stats_bucket,
                entry.layer_index,
                entry.layout,
                builder.CreateVector(entry.shape),
                entry.storage_type,
                entry.compute_precision,
                entry.element_count,
                entry.payload_offset,
                entry.payload_size,
                entry.payload_xxhash64));
        }

        const std::uint64_t creation_timestamp_ms = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        const auto root = GRIMCheckpoint::CreateParameterCheckpoint(
            builder,
            kParameterCheckpointVersion,
            builder.CreateString(makeCheckpointId()),
            creation_timestamp_ms,
            builder.CreateVector(fact_offsets),
            compatibilityChecksum(facts),
            builder.CreateVector(entry_offsets),
            manifestChecksum(entries),
            builder.CreateVector(payload),
            xxhash64(payload));
        builder.Finish(root, "GRCP");

        flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
        if (!GRIMCheckpoint::VerifyParameterCheckpointBuffer(verifier)) {
            throw std::runtime_error(
                "saveParameterCheckpoint: generated checkpoint failed FlatBuffer verification");
        }

        writeAtomically(path, builder.GetBufferPointer(), builder.GetSize());
        EmitModuleInfo(
            ModuleId::Checkpoint,
            "Registry parameter checkpoint saved: " + path +
                " groups=" + std::to_string(groups.size()) +
                " payload_bytes=" + std::to_string(payload.size()));
        return true;
    } catch (const std::exception& error) {
        EmitModuleError(
            ModuleId::Checkpoint,
            std::string("saveParameterCheckpoint failed: ") + error.what());
        return false;
    }
}

bool loadParameterCheckpoint(
    const Config::AiConfigSnapshot& config,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    cudaStream_t stream,
    const std::string& path)
{
    try {
        requireCheckpointExtension(path, "loadParameterCheckpoint");
        if (!stream) {
            throw std::runtime_error("loadParameterCheckpoint: stream is NULL");
        }
        const std::vector<std::uint8_t> file = readFile(path);
        if (file.size() < 8) {
            throw std::runtime_error(
                "checkpoint is too small to contain a FlatBuffer identifier");
        }
        if (!flatbuffers::BufferHasIdentifier(file.data(), "GRCP")) {
            throw std::runtime_error(
                "checkpoint does not have the GRCP registry-checkpoint identifier");
        }
        flatbuffers::Verifier verifier(file.data(), file.size());
        if (!GRIMCheckpoint::VerifyParameterCheckpointBuffer(verifier)) {
            throw std::runtime_error("checkpoint failed FlatBuffer verification");
        }
        const auto* checkpoint = GRIMCheckpoint::GetParameterCheckpoint(file.data());
        if (!checkpoint || checkpoint->format_version() != kParameterCheckpointVersion) {
            throw std::runtime_error(
                "unsupported registry checkpoint version: checkpoint=" +
                std::to_string(checkpoint ? checkpoint->format_version() : 0) +
                " expected=" + std::to_string(kParameterCheckpointVersion));
        }
        if (!checkpoint->checkpoint_id() || checkpoint->checkpoint_id()->size() == 0) {
            throw std::runtime_error("checkpoint ID is missing or empty");
        }

        const auto expected_facts = buildCompatibilityFacts(config);
        std::string fact_mismatch;
        if (!compatibilityFactsMatch(
                checkpoint->compatibility_facts(),
                expected_facts,
                fact_mismatch)) {
            throw std::runtime_error(fact_mismatch);
        }
        if (checkpoint->compatibility_xxhash64() !=
            compatibilityChecksum(expected_facts)) {
            throw std::runtime_error("checkpoint compatibility checksum mismatch");
        }

        auto& groups = parameter_registry.requireParameterGroups("loadParameterCheckpoint");
        const auto expected_entries = buildHostManifest(groups);
        const auto* stored_entries = checkpoint->parameters();
        const auto* payload = checkpoint->payload();
        if (!stored_entries || stored_entries->size() != expected_entries.size()) {
            throw std::runtime_error(
                "parameter group count mismatch: checkpoint=" +
                std::to_string(stored_entries ? stored_entries->size() : 0) +
                " current=" + std::to_string(expected_entries.size()));
        }
        if (!payload) {
            throw std::runtime_error("checkpoint payload is missing");
        }
        if (checkpoint->payload_xxhash64() !=
            xxhash64(payload->Data(), payload->size())) {
            throw std::runtime_error("checkpoint payload checksum mismatch");
        }

        std::unordered_set<std::string> names;
        std::vector<HostParameterEntry> stored_host_entries;
        stored_host_entries.reserve(expected_entries.size());
        std::uint64_t expected_offset = 0;
        for (std::size_t i = 0; i < expected_entries.size(); ++i) {
            const auto* stored = stored_entries->Get(static_cast<flatbuffers::uoffset_t>(i));
            const auto& expected = expected_entries[i];
            if (!stored || !stored->name() || !stored->shape()) {
                throw std::runtime_error(
                    "invalid parameter directory entry at index " + std::to_string(i));
            }
            const std::string stored_name = stored->name()->str();
            if (!names.insert(stored_name).second) {
                throw std::runtime_error(
                    "duplicate checkpoint parameter name: " + stored_name);
            }
            if (stored_name != expected.name ||
                stored->group_type() != expected.group_type ||
                stored->stats_bucket() != expected.stats_bucket ||
                stored->layer_index() != expected.layer_index ||
                stored->layout() != expected.layout ||
                !shapeMatches(stored->shape(), expected.shape) ||
                stored->storage_type() != expected.storage_type ||
                stored->compute_precision() != expected.compute_precision ||
                stored->element_count() != expected.element_count ||
                stored->payload_size() != expected.payload_size) {
                throw std::runtime_error(
                    "parameter manifest mismatch at index " + std::to_string(i) +
                    ": checkpoint=" + stored_name + " current=" + expected.name);
            }
            if (stored->payload_offset() != expected_offset ||
                (stored->payload_offset() % kStoredElementBytes) != 0) {
                throw std::runtime_error(
                    "invalid payload offset for parameter " + stored_name);
            }
            const std::uint64_t payload_end = checkedAdd(
                stored->payload_offset(),
                stored->payload_size(),
                "stored parameter payload range");
            if (payload_end > payload->size()) {
                throw std::runtime_error(
                    "payload range exceeds checkpoint payload for parameter " + stored_name);
            }
            const std::uint8_t* parameter_bytes =
                payload->Data() + static_cast<std::size_t>(stored->payload_offset());
            if (stored->payload_xxhash64() !=
                xxhash64(parameter_bytes, static_cast<std::size_t>(stored->payload_size()))) {
                throw std::runtime_error(
                    "payload checksum mismatch for parameter " + stored_name);
            }

            HostParameterEntry stored_host = expected;
            stored_host.payload_offset = stored->payload_offset();
            stored_host.payload_size = stored->payload_size();
            stored_host_entries.push_back(std::move(stored_host));
            expected_offset = payload_end;
        }
        if (expected_offset != payload->size()) {
            throw std::runtime_error(
                "parameter payload is not exactly covered by the registry directory");
        }
        if (checkpoint->manifest_xxhash64() != manifestChecksum(stored_host_entries)) {
            throw std::runtime_error("checkpoint parameter manifest checksum mismatch");
        }

        // All structure, compatibility, manifest, ranges, and checksums have
        // passed. GPU mutation begins only after this transaction preflight.
        for (std::size_t i = 0; i < groups.size(); ++i) {
            const auto& entry = stored_host_entries[i];
            const cudaError_t copy_error = cudaMemcpyAsync(
                groups[i].tensor->data,
                payload->Data() + static_cast<std::size_t>(entry.payload_offset),
                static_cast<std::size_t>(entry.payload_size),
                cudaMemcpyHostToDevice,
                stream);
            if (copy_error != cudaSuccess) {
                throw std::runtime_error(
                    "loadParameterCheckpoint: H2D copy failed for " + entry.name +
                    ": " + cudaGetErrorString(copy_error));
            }
        }
        const cudaError_t sync_error = cudaStreamSynchronize(stream);
        if (sync_error != cudaSuccess) {
            throw std::runtime_error(
                "loadParameterCheckpoint: final stream synchronization failed: " +
                std::string(cudaGetErrorString(sync_error)));
        }

        EmitModuleInfo(
            ModuleId::Checkpoint,
            "Registry parameter checkpoint loaded: " + path +
                " groups=" + std::to_string(groups.size()) +
                " payload_bytes=" + std::to_string(payload->size()));
        return true;
    } catch (const std::exception& error) {
        EmitModuleError(
            ModuleId::Checkpoint,
            std::string("loadParameterCheckpoint failed for '") + path +
                "': " + error.what());
        return false;
    }
}

} // namespace GRIM::Checkpoint
