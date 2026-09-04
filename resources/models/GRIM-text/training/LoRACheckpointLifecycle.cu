#include "LoRACheckpointLifecycle.hpp"

#include "Phases/Phase1_Startup.hpp"
#include "Phases/Startup/Model/ParameterRegistry.hpp"
#include "../Common/LoRATrainingCheckpoint.hpp"
#include "../Shared/HyperParameters/HyperparameterGroupings.hpp"

#include <cuda_runtime.h>
#include <curand.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace GRIMText::Training {
namespace {

namespace fs = std::filesystem;
using GRIM::Checkpoint::LoRAHostTensorState;
using GRIM::Checkpoint::LoRAMatrixOrientation;
using GRIM::Checkpoint::LoRATargetTrainingState;
using GRIM::Checkpoint::LoRATrainingCheckpointSnapshot;

constexpr std::uint64_t kDataOrderMagic = 0x4752494d4c4f5244ULL;
constexpr std::uint64_t kStateEncodingVersion = 1;

class Sha256 {
public:
    void update(const std::uint8_t* data, std::size_t size) {
        total_bytes_ += size;
        while (size != 0) {
            const std::size_t take = std::min(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, data, take);
            block_size_ += take;
            data += take;
            size -= take;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    std::array<std::uint8_t, 32> finish() {
        const std::uint64_t bit_count = static_cast<std::uint64_t>(total_bytes_) * 8u;
        block_[block_size_++] = 0x80;
        if (block_size_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.end(), 0);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.begin() + 56, 0);
        for (int index = 0; index < 8; ++index) {
            block_[63 - index] = static_cast<std::uint8_t>(bit_count >> (index * 8));
        }
        transform(block_.data());
        std::array<std::uint8_t, 32> result{};
        for (std::size_t index = 0; index < state_.size(); ++index) {
            result[index * 4] = static_cast<std::uint8_t>(state_[index] >> 24);
            result[index * 4 + 1] = static_cast<std::uint8_t>(state_[index] >> 16);
            result[index * 4 + 2] = static_cast<std::uint8_t>(state_[index] >> 8);
            result[index * 4 + 3] = static_cast<std::uint8_t>(state_[index]);
        }
        return result;
    }

private:
    static std::uint32_t rotateRight(std::uint32_t value, unsigned count) {
        return (value >> count) | (value << (32u - count));
    }

    static std::uint32_t readBe32(const std::uint8_t* data) {
        return (static_cast<std::uint32_t>(data[0]) << 24) |
               (static_cast<std::uint32_t>(data[1]) << 16) |
               (static_cast<std::uint32_t>(data[2]) << 8) |
               static_cast<std::uint32_t>(data[3]);
    }

    void transform(const std::uint8_t* block) {
        static constexpr std::array<std::uint32_t, 64> constants = {{
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
        }};
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) words[index] = readBe32(block + index * 4);
        for (std::size_t index = 16; index < words.size(); ++index) {
            const std::uint32_t s0 = rotateRight(words[index - 15], 7) ^ rotateRight(words[index - 15], 18) ^ (words[index - 15] >> 3);
            const std::uint32_t s1 = rotateRight(words[index - 2], 17) ^ rotateRight(words[index - 2], 19) ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        std::uint32_t a=state_[0], b=state_[1], c=state_[2], d=state_[3];
        std::uint32_t e=state_[4], f=state_[5], g=state_[6], h=state_[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const std::uint32_t s1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
            const std::uint32_t choice = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + s1 + choice + constants[index] + words[index];
            const std::uint32_t s0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + majority;
            h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
        }
        state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
        state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
    }

    std::array<std::uint32_t, 8> state_{{0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u}};
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_ = 0;
    std::size_t total_bytes_ = 0;
};

std::array<std::uint8_t, 32> sha256String(const std::string& value) {
    Sha256 hash;
    hash.update(reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
    return hash.finish();
}

std::array<std::uint8_t, 32> sha256File(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open base checkpoint for SHA-256: " + path.string());
    Sha256 hash;
    std::array<char, 1024 * 1024> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) hash.update(reinterpret_cast<const std::uint8_t*>(buffer.data()), static_cast<std::size_t>(count));
    }
    if (!input.eof()) throw std::runtime_error("Failed reading base checkpoint for SHA-256: " + path.string());
    return hash.finish();
}

std::string canonicalLoRAConfig(const GRIM::Config::AiConfigSnapshot& config) {
    const auto hp = GRIM::HyperParameters::loraTrainingHP(config);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::boolalpha << std::setprecision(std::numeric_limits<float>::max_digits10);
    const auto write_class = [&](const char* name, const GRIM::HyperParameters::LoRAClassSettingsHP& value) {
        out << name << ".enabled=" << value.enabled << '\n'
            << name << ".rank=" << value.rank << '\n'
            << name << ".alpha=" << value.alpha << '\n'
            << name << ".precision=" << static_cast<int>(value.precision) << '\n';
    };
    write_class("qkv", hp.qkv); write_class("o", hp.o); write_class("gate", hp.gate);
    write_class("w1", hp.w1); write_class("w2", hp.w2);
    out << "learning_rate_lora=" << hp.learning_rate_lora << '\n';
    return out.str();
}

std::string canonicalOptimizerConfig(const GRIM::Config::AiConfigSnapshot& config) {
    const auto hp = GRIM::HyperParameters::optimizerUpdateHP(config);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::boolalpha << std::setprecision(std::numeric_limits<float>::max_digits10)
        << "kind=" << static_cast<int>(hp.kind) << '\n'
        << "weight_decay=" << hp.weight_decay << '\n'
        << "use_depth_aware_upsilon=" << hp.use_depth_aware_upsilon << '\n'
        << "beta1=" << hp.beta1 << '\n' << "beta2=" << hp.beta2 << '\n'
        << "epsilon=" << hp.epsilon << '\n';
    return out.str();
}

std::vector<std::uint8_t> encodeI64(std::int64_t value) {
    std::vector<std::uint8_t> bytes(8);
    const std::uint64_t encoded = static_cast<std::uint64_t>(value);
    for (int index = 0; index < 8; ++index) bytes[index] = static_cast<std::uint8_t>(encoded >> (index * 8));
    return bytes;
}

std::int64_t decodeI64(const std::vector<std::uint8_t>& bytes, const char* name) {
    if (bytes.size() != 8) throw std::runtime_error(std::string(name) + ": unsupported state encoding");
    std::uint64_t value = 0;
    for (int index = 7; index >= 0; --index) value = (value << 8) | bytes[static_cast<std::size_t>(index)];
    return static_cast<std::int64_t>(value);
}

std::vector<std::uint64_t> encodeDataOrder(const std::vector<std::vector<int>>& order) {
    std::vector<std::uint64_t> encoded{kDataOrderMagic, kStateEncodingVersion, static_cast<std::uint64_t>(order.size())};
    for (const auto& epoch : order) {
        encoded.push_back(static_cast<std::uint64_t>(epoch.size()));
        for (int index : epoch) {
            if (index < 0) throw std::runtime_error("LoRA data order contains a negative payload index");
            encoded.push_back(static_cast<std::uint64_t>(index));
        }
    }
    return encoded;
}

void restoreDataOrder(TrainingContext& ctx, const std::vector<std::uint64_t>& encoded) {
    if (encoded.size() < 3 || encoded[0] != kDataOrderMagic || encoded[1] != kStateEncodingVersion ||
        encoded[2] != ctx.epoch_batch_order.size()) {
        throw std::runtime_error("LoRA checkpoint data-order header does not match the current epoch plan");
    }
    std::size_t cursor = 3;
    for (auto& epoch : ctx.epoch_batch_order) {
        if (cursor >= encoded.size() || encoded[cursor++] != epoch.size()) {
            throw std::runtime_error("LoRA checkpoint data-order epoch size does not match the current plan");
        }
        for (auto& index : epoch) {
            if (cursor >= encoded.size() || encoded[cursor] >= ctx.train_payloads.size()) {
                throw std::runtime_error("LoRA checkpoint data-order payload index is invalid");
            }
            index = static_cast<int>(encoded[cursor++]);
        }
    }
    if (cursor != encoded.size()) throw std::runtime_error("LoRA checkpoint data-order state has trailing values");
}

LoRAHostTensorState downloadTensor(const GRIM::Tensor& tensor, const float* data, const std::string& name) {
    tensor.require(name.c_str());
    if (!data || !tensor.shape.is_2d_layout()) throw std::runtime_error(name + ": tensor storage or 2D shape is unavailable");
    const auto shape = tensor.shape.as_2d();
    LoRAHostTensorState result;
    result.shape = {static_cast<std::uint32_t>(shape.rows), static_cast<std::uint32_t>(shape.cols)};
    result.values.resize(tensor.numel());
    const cudaError_t error = cudaMemcpy(result.values.data(), data, tensor.size_bytes(), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) throw std::runtime_error(name + ": D2H copy failed: " + cudaGetErrorString(error));
    return result;
}

void uploadTensor(GRIM::Tensor& tensor, float* destination, const LoRAHostTensorState& source, cudaStream_t stream, const std::string& name) {
    if (!destination || source.values.size() != tensor.numel()) throw std::runtime_error(name + ": restore tensor size mismatch");
    const cudaError_t error = cudaMemcpyAsync(destination, source.values.data(), tensor.size_bytes(), cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) throw std::runtime_error(name + ": H2D copy failed: " + cudaGetErrorString(error));
}

const GRIM::ParameterGroup& groupForTensor(const TrainingContext& ctx, const GRIM::Tensor& tensor, const std::string& name) {
    for (const auto& group : ctx.parameter_registry.requireParameterGroups("LoRA checkpoint bridge")) {
        if (group.tensor == &tensor) {
            if (!group.m_tensor || !group.v_tensor) throw std::runtime_error(name + ": optimizer moments are not bound");
            return group;
        }
    }
    throw std::runtime_error(name + ": tensor is absent from the LoRA optimizer inventory");
}

const char* projectionName(GRIM::LoRAMatrixClass matrix_class) {
    switch (matrix_class) {
        case GRIM::LoRAMatrixClass::QKV: return "qkv";
        case GRIM::LoRAMatrixClass::ATTENTION_OUTPUT: return "wo";
        case GRIM::LoRAMatrixClass::FFN_GATE: return "ffn_w_gate";
        case GRIM::LoRAMatrixClass::FFN_UP: return "ffn_w1";
        case GRIM::LoRAMatrixClass::FFN_DOWN: return "ffn_w2";
    }
    throw std::runtime_error("LoRA checkpoint bridge encountered an unknown matrix class");
}

std::string targetIdentity(std::size_t layer, GRIM::LoRAMatrixClass matrix_class) {
    return "layer" + std::to_string(layer) + "_" + projectionName(matrix_class);
}

std::string sanitizeId(std::string value) {
    for (char& character : value) {
        const bool valid = (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') || character == '-' || character == '_';
        if (!valid) character = '_';
    }
    if (value.empty()) throw std::runtime_error("Selected model store directory cannot produce a LoRA adapter ID");
    return value;
}

std::string digestPrefix(const std::array<std::uint8_t, 32>& digest) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(16);
    for (std::size_t index = 0; index < 8; ++index) {
        result.push_back(digits[digest[index] >> 4]);
        result.push_back(digits[digest[index] & 0x0f]);
    }
    return result;
}

LoRATrainingCheckpointSnapshot captureSnapshot(TrainingContext& ctx, const std::string& boundary, int epochs_completed) {
    auto& lifecycle = ctx.lora_checkpoint;
    if (!lifecycle.active || lifecycle.next_output_adapter_revision == 0) throw std::runtime_error("LoRA checkpoint lifecycle is not initialized");
    if (epochs_completed < 0 || ctx.global_step < 0 || ctx.optimizer.optimizer_step.step < 0) throw std::runtime_error("LoRA checkpoint progress contains a negative counter");
    if (ctx.optimizer.accumulationSlot() != 0) throw std::runtime_error("LoRA checkpoint boundary requires accumulation_slot=0");
    if (!std::isfinite(ctx.best_val_loss)) throw std::runtime_error("LoRA checkpoint boundary requires a finite best validation loss");

    const auto now = std::chrono::system_clock::now();
    const std::uint64_t timestamp = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
    LoRATrainingCheckpointSnapshot snapshot;
    snapshot.creation_timestamp_ms = timestamp;
    snapshot.adapter_id = lifecycle.adapter_id;
    snapshot.parent_adapter_revision = lifecycle.parent_adapter_revision;
    snapshot.output_adapter_revision = lifecycle.next_output_adapter_revision;
    snapshot.checkpoint_id = lifecycle.adapter_id + ".r" + std::to_string(snapshot.output_adapter_revision) + "." + boundary + "." + std::to_string(timestamp);
    snapshot.base_checkpoint_identity = lifecycle.base_checkpoint_identity;
    snapshot.base_checkpoint_sha256 = lifecycle.base_checkpoint_sha256;
    snapshot.training_config_canonical = lifecycle.training_config_canonical;
    snapshot.training_config_sha256 = lifecycle.training_config_sha256;
    const auto& compiled = *ctx.config.model_config;
    snapshot.architecture = {compiled.architecture.d_model, compiled.architecture.num_layers,
        compiled.architecture.num_heads, compiled.architecture.num_kv_heads,
        compiled.derived_architecture.qkv_dim, compiled.architecture.d_ff};

    for (std::size_t layer = 0; layer < ctx.parameter_registry.loraLayerParameterPairs().size(); ++layer) {
        const auto& pairs = ctx.parameter_registry.loraLayerParameterPairs()[layer].pairs;
        for (const auto& pair_owner : pairs) {
            if (!pair_owner) continue;
            const auto& pair = *pair_owner;
            const std::string identity = targetIdentity(layer, pair.matrix_class);
            const auto& a_group = groupForTensor(ctx, pair.A, identity + ".A");
            const auto& b_group = groupForTensor(ctx, pair.B, identity + ".B");
            LoRATargetTrainingState target;
            target.target_identity = identity;
            target.layer_index = static_cast<std::uint32_t>(layer);
            target.matrix_class = pair.matrix_class;
            const auto a_shape = pair.A.shape.as_2d();
            const auto b_shape = pair.B.shape.as_2d();
            target.base_shape = {static_cast<std::uint32_t>(b_shape.rows), static_cast<std::uint32_t>(a_shape.cols)};
            target.orientation = LoRAMatrixOrientation::DIRECT_WEIGHT;
            if (pair.matrix_class == GRIM::LoRAMatrixClass::QKV ||
                pair.matrix_class == GRIM::LoRAMatrixClass::ATTENTION_OUTPUT) {
                target.orientation = LoRAMatrixOrientation::TRANSPOSED_WEIGHT;
            }
            target.rank = pair.rank;
            target.alpha = pair.alpha;
            target.A = downloadTensor(pair.A, pair.A.data, identity + ".A");
            target.B = downloadTensor(pair.B, pair.B.data, identity + ".B");
            target.A_first_moment = downloadTensor(*a_group.m_tensor, a_group.m_state(), identity + ".A.m");
            target.A_second_moment = downloadTensor(*a_group.v_tensor, a_group.v_state(), identity + ".A.v");
            target.B_first_moment = downloadTensor(*b_group.m_tensor, b_group.m_state(), identity + ".B.m");
            target.B_second_moment = downloadTensor(*b_group.v_tensor, b_group.v_state(), identity + ".B.v");
            snapshot.targets.push_back(std::move(target));
        }
    }

    const auto lora_hp = GRIM::HyperParameters::loraTrainingHP(ctx.config);
    snapshot.optimizer.family = "GRIM::OptimizerKind=" + std::to_string(static_cast<int>(GRIM::HyperParameters::optimizerUpdateHP(ctx.config).kind));
    snapshot.optimizer.canonical_config = canonicalOptimizerConfig(ctx.config);
    snapshot.optimizer.optimizer_step = static_cast<std::uint64_t>(ctx.optimizer.optimizer_step.step);
    snapshot.optimizer.learning_rate_lora = lora_hp.learning_rate_lora;
    const std::int64_t current_schedule_step =
        static_cast<std::int64_t>(ctx.lr_schedule_step_at_start) +
        static_cast<std::int64_t>(ctx.optimizer.optimizer_step.step) -
        static_cast<std::int64_t>(ctx.lr_schedule_optimizer_step_at_start);
    if (current_schedule_step < 0) throw std::runtime_error("LoRA checkpoint LR schedule step is negative");
    snapshot.optimizer.lr_scheduler_state = encodeI64(current_schedule_step);
    snapshot.optimizer.soft_restart_state = encodeI64(ctx.optimizer.soft_restart_controller.state().last_restart_step);
    snapshot.progress = {static_cast<std::uint64_t>(ctx.global_step), static_cast<std::uint64_t>(epochs_completed), 0, static_cast<std::uint32_t>(ctx.optimizer.accumulationSlot()), ctx.best_val_loss, encodeDataOrder(ctx.epoch_batch_order)};
    snapshot.rng.base_seed = ctx.rng.base_seed;
    snapshot.rng.data_seed = ctx.rng.data_seed;
    snapshot.rng.init_seed = ctx.rng.init_seed;
    snapshot.rng.cuda_seed = ctx.rng.cuda_seed;
    std::ostringstream rng_state;
    rng_state.imbue(std::locale::classic());
    rng_state << ctx.rng.data_rng;
    snapshot.rng.data_rng_state = rng_state.str();
    // This generator currently has no draw sites, so its exact durable position
    // is zero. Persist the position explicitly rather than duplicating the seed.
    snapshot.rng.cuda_rng_state = encodeI64(0);
    return snapshot;
}

void restoreSnapshot(TrainingContext& ctx, const LoRATrainingCheckpointSnapshot& snapshot) {
    if (snapshot.optimizer.canonical_config != canonicalOptimizerConfig(ctx.config)) throw std::runtime_error("LoRA checkpoint optimizer configuration does not match current configuration");
    const auto lora_hp = GRIM::HyperParameters::loraTrainingHP(ctx.config);
    const std::string expected_optimizer_family = "GRIM::OptimizerKind=" + std::to_string(static_cast<int>(GRIM::HyperParameters::optimizerUpdateHP(ctx.config).kind));
    if (snapshot.training_config_canonical != ctx.lora_checkpoint.training_config_canonical ||
        snapshot.optimizer.family != expected_optimizer_family ||
        snapshot.optimizer.learning_rate_lora != lora_hp.learning_rate_lora) {
        throw std::runtime_error("LoRA checkpoint canonical training or optimizer identity does not match current configuration");
    }
    if (snapshot.progress.global_step > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        snapshot.progress.epochs_completed > static_cast<std::uint64_t>(ctx.epoch_batch_order.size()) ||
        snapshot.optimizer.optimizer_step > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        snapshot.progress.batch_cursor > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        snapshot.progress.accumulation_cursor > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("LoRA checkpoint progress exceeds runtime limits");
    }
    const auto schedule_hp = GRIM::HyperParameters::trainingScheduleHP(ctx.config);
    if (snapshot.progress.accumulation_cursor >= static_cast<std::uint32_t>(schedule_hp.gradient_accumulation_steps)) {
        throw std::runtime_error("LoRA checkpoint accumulation cursor is outside the configured window");
    }
    if (snapshot.progress.batch_cursor % static_cast<std::uint64_t>(schedule_hp.gradient_accumulation_steps) !=
        snapshot.progress.accumulation_cursor) {
        throw std::runtime_error("LoRA checkpoint batch and accumulation cursors are inconsistent");
    }

    std::unordered_set<std::string> restored;
    const cudaStream_t stream = ctx.requireTrainingState("restoreLoRATrainingCheckpoint").stream_ctrl.getPrimaryStream();
    for (const auto& target : snapshot.targets) {
        auto& pair = ctx.parameter_registry.requireLoRAParameterPair(static_cast<int>(target.layer_index), target.matrix_class, "restoreLoRATrainingCheckpoint");
        const std::string expected_identity = targetIdentity(target.layer_index, target.matrix_class);
        if (target.target_identity != expected_identity || pair.rank != target.rank || pair.alpha != target.alpha) throw std::runtime_error("LoRA checkpoint target does not match allocated adapter: " + target.target_identity);
        const auto& a_group = groupForTensor(ctx, pair.A, expected_identity + ".A");
        const auto& b_group = groupForTensor(ctx, pair.B, expected_identity + ".B");
        uploadTensor(pair.A, pair.A.data, target.A, stream, expected_identity + ".A");
        uploadTensor(pair.B, pair.B.data, target.B, stream, expected_identity + ".B");
        if (snapshot.progress.accumulation_cursor != 0) {
            if (!target.A_gradient || !target.B_gradient) throw std::runtime_error("LoRA checkpoint is missing accumulated gradients for " + expected_identity);
            uploadTensor(pair.A, pair.A.grad_data(), *target.A_gradient, stream, expected_identity + ".A.grad");
            uploadTensor(pair.B, pair.B.grad_data(), *target.B_gradient, stream, expected_identity + ".B.grad");
        }
        uploadTensor(*a_group.m_tensor, a_group.m_state(), target.A_first_moment, stream, expected_identity + ".A.m");
        uploadTensor(*a_group.v_tensor, a_group.v_state(), target.A_second_moment, stream, expected_identity + ".A.v");
        uploadTensor(*b_group.m_tensor, b_group.m_state(), target.B_first_moment, stream, expected_identity + ".B.m");
        uploadTensor(*b_group.v_tensor, b_group.v_state(), target.B_second_moment, stream, expected_identity + ".B.v");
        restored.insert(expected_identity);
    }
    std::size_t expected_targets = 0;
    for (const auto& layer : ctx.parameter_registry.loraLayerParameterPairs()) for (const auto& pair : layer.pairs) if (pair) ++expected_targets;
    if (restored.size() != expected_targets) throw std::runtime_error("LoRA checkpoint target inventory is incomplete");
    const cudaError_t sync_error = cudaStreamSynchronize(stream);
    if (sync_error != cudaSuccess) throw std::runtime_error(std::string("LoRA checkpoint restore synchronization failed: ") + cudaGetErrorString(sync_error));

    restoreDataOrder(ctx, snapshot.progress.data_order);
    if (snapshot.progress.epochs_completed < ctx.epoch_batch_order.size()) {
        const std::size_t epoch_size = ctx.epoch_batch_order[static_cast<std::size_t>(snapshot.progress.epochs_completed)].size();
        if (snapshot.progress.batch_cursor >= epoch_size && snapshot.progress.batch_cursor != 0) {
            throw std::runtime_error("LoRA checkpoint batch cursor is not a resumable position in the restored epoch order");
        }
    } else if (snapshot.progress.batch_cursor != 0) {
        throw std::runtime_error("LoRA checkpoint has a batch cursor after the final configured epoch");
    }
    ctx.global_step = static_cast<int>(snapshot.progress.global_step);
    ctx.epochs_completed = static_cast<int>(snapshot.progress.epochs_completed);
    ctx.resume_batch_cursor = static_cast<int>(snapshot.progress.batch_cursor);
    ctx.best_val_loss = snapshot.progress.best_validation_loss;
    ctx.optimizer.optimizer_step.step = static_cast<int>(snapshot.optimizer.optimizer_step);
    ctx.optimizer.setAccumulationSlot(static_cast<int>(snapshot.progress.accumulation_cursor));
    ctx.optimizer.optimizer_state_restored = true;
    const std::int64_t restored_schedule_step = decodeI64(snapshot.optimizer.lr_scheduler_state, "LoRA LR scheduler state");
    if (restored_schedule_step < 0 || restored_schedule_step > std::numeric_limits<int>::max()) throw std::runtime_error("LoRA checkpoint LR scheduler step exceeds runtime limits");
    ctx.lr_schedule_step_at_start = static_cast<int>(restored_schedule_step);
    ctx.lr_schedule_optimizer_step_at_start = ctx.optimizer.optimizer_step.step;
    GRIM::SoftRestart::SoftRestartState soft_restart_state;
    soft_restart_state.last_restart_step = decodeI64(snapshot.optimizer.soft_restart_state, "LoRA soft-restart state");
    ctx.optimizer.soft_restart_controller.restoreState(soft_restart_state);
    ctx.rng.base_seed = snapshot.rng.base_seed;
    ctx.rng.data_seed = snapshot.rng.data_seed;
    ctx.rng.init_seed = snapshot.rng.init_seed;
    ctx.rng.cuda_seed = snapshot.rng.cuda_seed;
    std::istringstream rng_state(snapshot.rng.data_rng_state);
    rng_state.imbue(std::locale::classic());
    if (!(rng_state >> ctx.rng.data_rng)) throw std::runtime_error("LoRA checkpoint data RNG state is invalid");
    const std::int64_t cuda_rng_offset =
        decodeI64(snapshot.rng.cuda_rng_state, "LoRA CUDA RNG state");
    if (cuda_rng_offset < 0) throw std::runtime_error("LoRA checkpoint CUDA RNG offset is negative");
    if (!ctx.rng.cuda_rng_initialized || !ctx.rng.cuda_rng_generator) throw std::runtime_error("LoRA checkpoint restore requires an initialized CUDA RNG generator");
    const auto cuda_generator = static_cast<curandGenerator_t>(ctx.rng.cuda_rng_generator);
    const curandStatus_t seed_error = curandSetPseudoRandomGeneratorSeed(cuda_generator, ctx.rng.cuda_seed);
    if (seed_error != CURAND_STATUS_SUCCESS) throw std::runtime_error("LoRA checkpoint CUDA RNG seed restore failed");
    const curandStatus_t offset_error = curandSetGeneratorOffset(
        cuda_generator,
        static_cast<unsigned long long>(cuda_rng_offset));
    if (offset_error != CURAND_STATUS_SUCCESS) throw std::runtime_error("LoRA checkpoint CUDA RNG offset restore failed");
}

} // namespace

void initializeLoRACheckpointLifecycle(TrainingContext& ctx) {
    if (!GRIM::HyperParameters::snapshotTrainingConfigField<bool>(ctx.config, "lora_model")) return;
    if (ctx.loaded_checkpoint_path.empty()) throw std::runtime_error("LoRA training requires an explicitly loaded base .grimckpt checkpoint");
    if (!ctx.config.model_config) throw std::runtime_error("LoRA checkpoint lifecycle requires the selected model.grimcfg");

    auto& lifecycle = ctx.lora_checkpoint;
    lifecycle.active = true;
    const std::string model_store_identity = fs::absolute(ctx.config.model_config->source_path.parent_path()).lexically_normal().generic_string();
    lifecycle.adapter_id = sanitizeId(ctx.config.model_config->source_path.parent_path().filename().string()) + "-" + digestPrefix(sha256String(model_store_identity));
    lifecycle.base_checkpoint_identity = fs::absolute(ctx.loaded_checkpoint_path).lexically_normal().generic_string();
    lifecycle.base_checkpoint_sha256 = sha256File(lifecycle.base_checkpoint_identity);
    lifecycle.training_config_canonical = canonicalLoRAConfig(ctx.config);
    lifecycle.training_config_sha256 = sha256String(lifecycle.training_config_canonical);

    const auto latest = GRIM::Checkpoint::findLatestLoRATrainingCheckpointName(ctx.config);
    if (!latest) {
        lifecycle.next_output_adapter_revision = 1;
        ctx.logging.logger->log("[LORA_RESUME] No .grimlorackpt exists in the selected model store; starting fresh adapter_id=\"" + lifecycle.adapter_id + "\" revision=1");
        return;
    }

    const auto snapshot = GRIM::Checkpoint::loadLoRATrainingCheckpoint(ctx.config, *latest,
        lifecycle.base_checkpoint_identity, lifecycle.base_checkpoint_sha256,
        lifecycle.training_config_canonical, lifecycle.training_config_sha256);
    restoreSnapshot(ctx, snapshot);
    lifecycle.adapter_id = snapshot.adapter_id;
    lifecycle.parent_adapter_revision = snapshot.output_adapter_revision;
    if (snapshot.output_adapter_revision == std::numeric_limits<std::uint64_t>::max()) throw std::runtime_error("LoRA adapter revision overflow");
    lifecycle.next_output_adapter_revision = snapshot.output_adapter_revision + 1;
    ctx.logging.logger->log("[LORA_RESUME] Restored newest checkpoint \"" + latest.value() + "\" adapter_id=\"" + lifecycle.adapter_id + "\" revision=" + std::to_string(snapshot.output_adapter_revision));
}

std::string saveLoRATrainingCheckpointAtBoundary(TrainingContext& ctx, const std::string& boundary, int epochs_completed) {
    if (!ctx.lora_checkpoint.active) return {};
    if (boundary.empty()) throw std::runtime_error("LoRA checkpoint boundary name is empty");
    auto snapshot = captureSnapshot(ctx, boundary, epochs_completed);
    std::ostringstream revision;
    revision << std::setw(8) << std::setfill('0') << snapshot.output_adapter_revision;
    const std::string name = snapshot.adapter_id + ".r" + revision.str() + "." + boundary + ".s" + std::to_string(ctx.global_step) + ".t" + std::to_string(snapshot.creation_timestamp_ms) + ".grimlorackpt";
    const fs::path saved = GRIM::Checkpoint::saveLoRATrainingCheckpoint(ctx.config, name, snapshot);
    ctx.lora_checkpoint.parent_adapter_revision = snapshot.output_adapter_revision;
    if (snapshot.output_adapter_revision == std::numeric_limits<std::uint64_t>::max()) throw std::runtime_error("LoRA adapter revision overflow after save");
    ctx.lora_checkpoint.next_output_adapter_revision = snapshot.output_adapter_revision + 1;
    ctx.logging.logger->log("[LORA_CHECKPOINT] Saved immutable checkpoint: " + saved.string());
    return saved.string();
}

} // namespace GRIMText::Training