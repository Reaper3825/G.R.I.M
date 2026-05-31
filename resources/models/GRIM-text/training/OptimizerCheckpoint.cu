//======================================================//
//  OptimizerCheckpoint.cu
//  Save/load optimizer state as a binary sidecar file
//======================================================//

#include "OptimizerCheckpoint.hpp"
#include "Phases/Phase1_Startup.hpp"

#include "../Shared/LogRecorder/LogRecorder.hpp"

#include <fstream>
#include <filesystem>
#include <cstring>
#include <vector>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using GRIM::Logging::ModuleId;
using GRIM::Logging::EmitModuleInfo;
using GRIM::Logging::EmitModuleWarning;
using GRIM::Logging::EmitModuleError;

namespace fs = std::filesystem;

namespace GRIMText::Training {

//======================================================//
//  Constants
//======================================================//

static constexpr uint32_t OPT_MAGIC   = 0x47524F50;  // "GROP"
static constexpr uint32_t OPT_VERSION = 1;
static constexpr size_t   RESERVED_BYTES = 32;

//======================================================//
//  Header Layout (POD, no padding surprises)
//======================================================//

#pragma pack(push, 1)
struct OptFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_groups;
    int32_t  optimizer_step;
    int32_t  global_step;
    float    best_val_loss;
    int32_t  current_epoch;
    int32_t  accumulation_slot;
    uint8_t  reserved[RESERVED_BYTES];
};
#pragma pack(pop)

static_assert(sizeof(OptFileHeader) == 4+4+4+4+4+4+4+4+RESERVED_BYTES,
              "OptFileHeader must be tightly packed");

//======================================================//
//  Path Helpers
//======================================================//

std::string optimizerSidecarPath(const std::string& checkpoint_path) {
    if (checkpoint_path.size() < 4 ||
        checkpoint_path.substr(checkpoint_path.size() - 4) != ".bin") {
        throw std::runtime_error(
            "[optimizerSidecarPath] checkpoint path must end with .bin: " + checkpoint_path);
    }
    return checkpoint_path.substr(0, checkpoint_path.size() - 4) + ".opt";
}

//======================================================//
//  Save
//======================================================//

bool saveOptimizerState(const TrainingContext& ctx, const std::string& sidecar_path) {
#ifdef USE_CUDA
    const auto& groups = ctx.parameter_registry.requireParameterGroups("saveOptimizerState");
    const auto& ts     = ctx.requireTrainingState("saveOptimizerState");
    const auto& opt_state = ctx.optimizer.optimizer_state;

    if (groups.empty()) {
        EmitModuleWarning(ModuleId::Checkpoint,
            "[saveOptimizerState] No parameter groups — skipping optimizer save",
            ctx.global_step);
        return false;
    }

    if (!opt_state.allocated) {
        throw std::runtime_error(
            "[saveOptimizerState] Optimizer states not allocated — "
            "buildParameterGroups() must be called before saving");
    }

    // Sync GPU before reading tensor data
    cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        EmitModuleError(ModuleId::Checkpoint,
            std::string("[saveOptimizerState] CUDA stream sync failed: ") +
            cudaGetErrorString(sync_err), ctx.global_step);
        return false;
    }

    // Build header
    OptFileHeader header{};
    header.magic          = OPT_MAGIC;
    header.version        = OPT_VERSION;
    header.num_groups     = static_cast<uint32_t>(groups.size());
    header.optimizer_step = ctx.optimizer.optimizer_step.step;
    header.global_step    = ctx.global_step;
    header.best_val_loss  = ctx.best_val_loss;
    header.current_epoch  = ctx.epochs_completed;
    header.accumulation_slot = ctx.optimizer.accumulationSlot();
    std::memset(header.reserved, 0, RESERVED_BYTES);

    // Open file
    std::ofstream out(sidecar_path, std::ios::binary);
    if (!out) {
        EmitModuleError(ModuleId::Checkpoint,
            "[saveOptimizerState] Failed to open file for writing: " + sidecar_path,
            ctx.global_step);
        return false;
    }

    // Write header
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write per-group directory
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto& g = groups[i];
        uint16_t name_len = static_cast<uint16_t>(g.name.size());
        out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        out.write(g.name.data(), name_len);

        uint64_t numel = static_cast<uint64_t>(g.size());
        out.write(reinterpret_cast<const char*>(&numel), sizeof(numel));
    }

    // Write bulk tensor data (D2H per group)
    for (size_t i = 0; i < groups.size(); ++i) {
        const size_t numel = groups[i].size();
        if (numel == 0) continue;

        const size_t bytes = numel * sizeof(float);
        std::vector<float> host_buf(numel);

        // m state
        if (i < opt_state.m_states.size() && opt_state.m_states[i].data) {
            cudaError_t err = cudaMemcpy(host_buf.data(), opt_state.m_states[i].data,
                                          bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    "[saveOptimizerState] cudaMemcpy m_state[" + std::to_string(i) +
                    "] failed: " + cudaGetErrorString(err));
            }
        } else {
            std::fill(host_buf.begin(), host_buf.end(), 0.0f);
        }
        out.write(reinterpret_cast<const char*>(host_buf.data()), bytes);

        // v state
        if (i < opt_state.v_states.size() && opt_state.v_states[i].data) {
            cudaError_t err = cudaMemcpy(host_buf.data(), opt_state.v_states[i].data,
                                          bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    "[saveOptimizerState] cudaMemcpy v_state[" + std::to_string(i) +
                    "] failed: " + cudaGetErrorString(err));
            }
        } else {
            std::fill(host_buf.begin(), host_buf.end(), 0.0f);
        }
        out.write(reinterpret_cast<const char*>(host_buf.data()), bytes);
    }

    out.flush();
    if (!out.good()) {
        EmitModuleError(ModuleId::Checkpoint,
            "[saveOptimizerState] Write error during flush: " + sidecar_path,
            ctx.global_step);
        return false;
    }
    out.close();

    // Log file size
    if (fs::exists(sidecar_path)) {
        auto file_size = fs::file_size(sidecar_path);
        EmitModuleInfo(ModuleId::Checkpoint,
            "✓ Optimizer state saved: " + sidecar_path +
            " (" + std::to_string(file_size / (1024*1024)) + " MB)",
            ctx.global_step);
    }

    return true;
#else
    (void)ctx; (void)sidecar_path;
    return false;
#endif
}

//======================================================//
//  Load
//======================================================//

bool loadOptimizerState(TrainingContext& ctx, const std::string& sidecar_path) {
#ifdef USE_CUDA
    if (!fs::exists(sidecar_path)) {
        return false;  // No sidecar file — not an error, just no optimizer state to restore
    }

    const auto& groups = ctx.parameter_registry.requireParameterGroups("loadOptimizerState");
    auto& ts = ctx.requireTrainingState("loadOptimizerState");
    auto& opt_state = ctx.optimizer.optimizer_state;

    if (!opt_state.allocated) {
        throw std::runtime_error(
            "[loadOptimizerState] Optimizer states not allocated — "
            "buildParameterGroups() must be called before loading");
    }

    // Open file
    std::ifstream in(sidecar_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(
            "[loadOptimizerState] Failed to open sidecar file: " + sidecar_path);
    }

    // Read header
    OptFileHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in.good()) {
        throw std::runtime_error(
            "[loadOptimizerState] Failed to read header from: " + sidecar_path);
    }

    // Validate magic
    if (header.magic != OPT_MAGIC) {
        throw std::runtime_error(
            "[loadOptimizerState] Invalid magic number in " + sidecar_path +
            " (expected 0x47524F50, got 0x" +
            ([&]{
                char buf[16];
                snprintf(buf, sizeof(buf), "%08X", header.magic);
                return std::string(buf);
            })() + ")");
    }

    // Validate version
    if (header.version != OPT_VERSION) {
        throw std::runtime_error(
            "[loadOptimizerState] Unsupported optimizer state version " +
            std::to_string(header.version) + " (expected " +
            std::to_string(OPT_VERSION) + ") in " + sidecar_path);
    }

    // Validate group count
    if (header.num_groups != static_cast<uint32_t>(groups.size())) {
        throw std::runtime_error(
            "[loadOptimizerState] Parameter group count mismatch: "
            "sidecar has " + std::to_string(header.num_groups) +
            " groups, current model has " + std::to_string(groups.size()) +
            ". Architecture changed since checkpoint — cannot resume optimizer.");
    }

    // Read and validate per-group directory
    struct GroupEntry {
        std::string name;
        uint64_t numel;
    };
    std::vector<GroupEntry> entries(header.num_groups);

    for (uint32_t i = 0; i < header.num_groups; ++i) {
        uint16_t name_len = 0;
        in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        if (!in.good()) {
            throw std::runtime_error(
                "[loadOptimizerState] Truncated directory at group " + std::to_string(i));
        }

        entries[i].name.resize(name_len);
        in.read(entries[i].name.data(), name_len);

        in.read(reinterpret_cast<char*>(&entries[i].numel), sizeof(entries[i].numel));
        if (!in.good()) {
            throw std::runtime_error(
                "[loadOptimizerState] Truncated directory at group " + std::to_string(i));
        }

        // Validate name match
        if (entries[i].name != groups[i].name) {
            throw std::runtime_error(
                "[loadOptimizerState] Parameter group name mismatch at index " +
                std::to_string(i) + ": sidecar has \"" + entries[i].name +
                "\", current model has \"" + groups[i].name +
                "\". Architecture changed since checkpoint — cannot resume optimizer.");
        }

        // Validate size match
        if (entries[i].numel != static_cast<uint64_t>(groups[i].size())) {
            throw std::runtime_error(
                "[loadOptimizerState] Parameter group size mismatch for \"" +
                entries[i].name + "\": sidecar has " +
                std::to_string(entries[i].numel) + " elements, current model has " +
                std::to_string(groups[i].size()) +
                ". Architecture changed since checkpoint — cannot resume optimizer.");
        }
    }

    // Read bulk tensor data (H2D per group)
    cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();

    for (uint32_t i = 0; i < header.num_groups; ++i) {
        const size_t numel = static_cast<size_t>(entries[i].numel);
        if (numel == 0) continue;

        const size_t bytes = numel * sizeof(float);
        std::vector<float> host_buf(numel);

        // m state
        in.read(reinterpret_cast<char*>(host_buf.data()), bytes);
        if (!in.good()) {
            throw std::runtime_error(
                "[loadOptimizerState] Truncated m_state data at group " + std::to_string(i));
        }
        if (i < opt_state.m_states.size() && opt_state.m_states[i].data) {
            cudaError_t err = cudaMemcpyAsync(opt_state.m_states[i].data, host_buf.data(),
                                               bytes, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    "[loadOptimizerState] cudaMemcpyAsync m_state[" + std::to_string(i) +
                    "] failed: " + cudaGetErrorString(err));
            }
        }

        // v state
        in.read(reinterpret_cast<char*>(host_buf.data()), bytes);
        if (!in.good()) {
            throw std::runtime_error(
                "[loadOptimizerState] Truncated v_state data at group " + std::to_string(i));
        }
        if (i < opt_state.v_states.size() && opt_state.v_states[i].data) {
            cudaError_t err = cudaMemcpyAsync(opt_state.v_states[i].data, host_buf.data(),
                                               bytes, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    "[loadOptimizerState] cudaMemcpyAsync v_state[" + std::to_string(i) +
                    "] failed: " + cudaGetErrorString(err));
            }
        }
    }

    // Sync to ensure all H2D copies complete before training starts
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            "[loadOptimizerState] Final stream sync failed: " +
            std::string(cudaGetErrorString(sync_err)));
    }

    // Restore metadata into TrainingContext
    ctx.optimizer.optimizer_step.step = header.optimizer_step;
    ctx.global_step                    = header.global_step;
    ctx.best_val_loss                  = header.best_val_loss;
    ctx.epochs_completed               = header.current_epoch;
    ctx.optimizer.setAccumulationSlot(header.accumulation_slot);

    EmitModuleInfo(ModuleId::Checkpoint,
        "✓ Optimizer state restored from " + sidecar_path +
        " (step=" + std::to_string(header.optimizer_step) +
        ", global_step=" + std::to_string(header.global_step) +
        ", best_val_loss=" + std::to_string(header.best_val_loss) +
        ", epoch=" + std::to_string(header.current_epoch) +
        ", accumulation_slot=" + std::to_string(header.accumulation_slot) +
        ", groups=" + std::to_string(header.num_groups) + ")",
        ctx.global_step);

    return true;
#else
    (void)ctx; (void)sidecar_path;
    return false;
#endif
}

} // namespace GRIMText::Training
