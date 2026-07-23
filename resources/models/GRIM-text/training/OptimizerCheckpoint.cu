//======================================================//
//  OptimizerCheckpoint.cu
//  Save/load optimizer state as a binary sidecar file
//======================================================//

#include "OptimizerCheckpoint.hpp"
#include "Phases/Phase1_Startup.hpp"

#include "../Shared/LogRecorder/LogRecorder.hpp"

#include <fstream>
#include <filesystem>
#include <cerrno>
#include <cstring>
#include <limits>
#include <sstream>
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

namespace {

uint64_t checkedAdd(uint64_t lhs, uint64_t rhs, const char* context) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        throw std::runtime_error(
            std::string("[OptimizerCheckpoint] Byte-count overflow while computing ") + context);
    }
    return lhs + rhs;
}

uint64_t checkedTensorBytes(uint64_t numel, const char* context) {
    constexpr uint64_t float_bytes = sizeof(float);
    if (numel > std::numeric_limits<uint64_t>::max() / float_bytes) {
        throw std::runtime_error(
            std::string("[OptimizerCheckpoint] Tensor byte-count overflow while computing ") +
            context);
    }
    return numel * float_bytes;
}

std::string streamState(const std::ios& stream) {
    std::ostringstream state;
    state << "good=" << (stream.good() ? "true" : "false")
          << ",eof=" << (stream.eof() ? "true" : "false")
          << ",fail=" << (stream.fail() ? "true" : "false")
          << ",bad=" << (stream.bad() ? "true" : "false");
    return state.str();
}

uint64_t fileSizeOrThrow(const std::string& path, const char* operation) {
    std::error_code ec;
    const auto bytes = fs::file_size(path, ec);
    if (ec) {
        throw std::runtime_error(
            std::string("[") + operation + "] Failed to query sidecar size: path=\"" +
            path + "\", error_code=" + std::to_string(ec.value()) +
            ", error=\"" + ec.message() + "\"");
    }
    if (bytes > std::numeric_limits<uint64_t>::max()) {
        throw std::runtime_error(
            std::string("[") + operation + "] Sidecar size exceeds uint64 range: path=\"" +
            path + "\"");
    }
    return static_cast<uint64_t>(bytes);
}

} // namespace

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

    uint64_t expected_file_bytes = sizeof(header);
    for (const auto& group : groups) {
        if (group.name.size() > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error(
                "[saveOptimizerState] Parameter group name is too long: group_name=\"" +
                group.name + "\", name_bytes=" + std::to_string(group.name.size()));
        }
        expected_file_bytes = checkedAdd(
            expected_file_bytes,
            sizeof(uint16_t) + static_cast<uint64_t>(group.name.size()) + sizeof(uint64_t),
            "optimizer sidecar directory");
        const uint64_t tensor_bytes =
            checkedTensorBytes(static_cast<uint64_t>(group.size()), "optimizer tensor data");
        expected_file_bytes = checkedAdd(
            expected_file_bytes, tensor_bytes, "optimizer m_state data");
        expected_file_bytes = checkedAdd(
            expected_file_bytes, tensor_bytes, "optimizer v_state data");
    }

    EmitModuleInfo(ModuleId::Checkpoint,
        "[saveOptimizerState] Sidecar write begin: path=\"" + sidecar_path +
        "\", groups=" + std::to_string(groups.size()) +
        ", expected_file_bytes=" + std::to_string(expected_file_bytes),
        ctx.global_step);

    // Open file
    std::ofstream out(sidecar_path, std::ios::binary);
    if (!out) {
        EmitModuleError(ModuleId::Checkpoint,
            "[saveOptimizerState] Failed to open file for writing: " + sidecar_path,
            ctx.global_step);
        return false;
    }

    auto writeBytes = [&](const char* data,
                          size_t bytes,
                          const char* stage,
                          size_t group_index,
                          const std::string& group_name) {
        if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error(
                std::string("[saveOptimizerState] Write request exceeds streamsize: stage=") +
                stage + ", requested_bytes=" + std::to_string(bytes));
        }

        const std::streampos raw_offset = out.tellp();
        const std::string offset = raw_offset == std::streampos(-1)
            ? std::string("unknown")
            : std::to_string(static_cast<uint64_t>(
                static_cast<std::streamoff>(raw_offset)));

        errno = 0;
        out.write(data, static_cast<std::streamsize>(bytes));
        if (!out.good()) {
            const int write_errno = errno;
            std::error_code size_ec;
            const auto actual_size = fs::file_size(sidecar_path, size_ec);
            throw std::runtime_error(
                std::string("[saveOptimizerState] Sidecar write failed: path=\"") +
                sidecar_path + "\", stage=" + stage +
                ", group_index=" +
                (group_index == std::numeric_limits<size_t>::max()
                    ? std::string("none")
                    : std::to_string(group_index)) +
                ", group_name=\"" + group_name +
                "\", offset=" + offset +
                ", requested_bytes=" + std::to_string(bytes) +
                ", expected_file_bytes=" + std::to_string(expected_file_bytes) +
                ", actual_file_bytes_on_disk=" +
                (size_ec ? std::string("unknown") : std::to_string(actual_size)) +
                ", errno=" + std::to_string(write_errno) +
                ", errno_message=\"" +
                (write_errno == 0 ? std::string("not reported") : std::strerror(write_errno)) +
                "\", stream_state={" + streamState(out) + "}");
        }
    };

    // Write header
    writeBytes(
        reinterpret_cast<const char*>(&header),
        sizeof(header),
        "header",
        std::numeric_limits<size_t>::max(),
        "");

    // Write per-group directory
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto& g = groups[i];
        uint16_t name_len = static_cast<uint16_t>(g.name.size());
        writeBytes(
            reinterpret_cast<const char*>(&name_len),
            sizeof(name_len),
            "directory_name_length",
            i,
            g.name);
        writeBytes(g.name.data(), name_len, "directory_name", i, g.name);

        uint64_t numel = static_cast<uint64_t>(g.size());
        writeBytes(
            reinterpret_cast<const char*>(&numel),
            sizeof(numel),
            "directory_numel",
            i,
            g.name);
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
        writeBytes(
            reinterpret_cast<const char*>(host_buf.data()),
            bytes,
            "m_state",
            i,
            groups[i].name);

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
        writeBytes(
            reinterpret_cast<const char*>(host_buf.data()),
            bytes,
            "v_state",
            i,
            groups[i].name);
    }

    errno = 0;
    out.flush();
    if (!out.good()) {
        const int flush_errno = errno;
        std::error_code size_ec;
        const auto actual_size = fs::file_size(sidecar_path, size_ec);
        throw std::runtime_error(
            "[saveOptimizerState] Sidecar flush failed: path=\"" + sidecar_path +
            "\", expected_file_bytes=" + std::to_string(expected_file_bytes) +
            ", actual_file_bytes_on_disk=" +
            (size_ec ? std::string("unknown") : std::to_string(actual_size)) +
            ", errno=" + std::to_string(flush_errno) +
            ", errno_message=\"" +
            (flush_errno == 0 ? std::string("not reported") : std::strerror(flush_errno)) +
            "\", stream_state={" + streamState(out) + "}");
    }
    out.close();

    const uint64_t actual_file_bytes =
        fileSizeOrThrow(sidecar_path, "saveOptimizerState");
    if (actual_file_bytes != expected_file_bytes) {
        const bool truncated = actual_file_bytes < expected_file_bytes;
        throw std::runtime_error(
            "[saveOptimizerState] Sidecar size mismatch after close: path=\"" +
            sidecar_path + "\", expected_file_bytes=" +
            std::to_string(expected_file_bytes) + ", actual_file_bytes=" +
            std::to_string(actual_file_bytes) +
            (truncated
                ? ", missing_bytes=" +
                    std::to_string(expected_file_bytes - actual_file_bytes)
                : ", trailing_bytes=" +
                    std::to_string(actual_file_bytes - expected_file_bytes)));
    }

    // Log file size
    if (fs::exists(sidecar_path)) {
        EmitModuleInfo(ModuleId::Checkpoint,
            "✓ Optimizer state saved: " + sidecar_path +
            " (" + std::to_string(actual_file_bytes / (1024*1024)) +
            " MB, bytes=" + std::to_string(actual_file_bytes) +
            ", groups=" + std::to_string(groups.size()) + ")",
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

    const std::streampos raw_tensor_data_offset = in.tellg();
    if (raw_tensor_data_offset == std::streampos(-1)) {
        throw std::runtime_error(
            "[loadOptimizerState] Failed to determine tensor data offset: path=\"" +
            sidecar_path + "\", stream_state={" + streamState(in) + "}");
    }
    const uint64_t tensor_data_offset =
        static_cast<uint64_t>(
            static_cast<std::streamoff>(raw_tensor_data_offset));

    uint64_t expected_file_bytes = tensor_data_offset;
    for (const auto& entry : entries) {
        const uint64_t tensor_bytes =
            checkedTensorBytes(entry.numel, "optimizer tensor data");
        expected_file_bytes = checkedAdd(
            expected_file_bytes, tensor_bytes, "optimizer m_state data");
        expected_file_bytes = checkedAdd(
            expected_file_bytes, tensor_bytes, "optimizer v_state data");
    }

    const uint64_t actual_file_bytes =
        fileSizeOrThrow(sidecar_path, "loadOptimizerState");

    EmitModuleInfo(ModuleId::Checkpoint,
        "[loadOptimizerState] Sidecar preflight: path=\"" + sidecar_path +
        "\", groups=" + std::to_string(header.num_groups) +
        ", actual_file_bytes=" + std::to_string(actual_file_bytes) +
        ", expected_file_bytes=" + std::to_string(expected_file_bytes) +
        ", header_bytes=" + std::to_string(sizeof(header)) +
        ", directory_bytes=" +
        std::to_string(tensor_data_offset - sizeof(header)) +
        ", tensor_data_offset=" + std::to_string(tensor_data_offset),
        ctx.global_step);

    if (actual_file_bytes < expected_file_bytes) {
        uint64_t tensor_offset = tensor_data_offset;
        for (uint32_t i = 0; i < header.num_groups; ++i) {
            const uint64_t tensor_bytes =
                checkedTensorBytes(entries[i].numel, "optimizer tensor data");
            for (const char* tensor_name : {"m_state", "v_state"}) {
                const uint64_t tensor_end =
                    checkedAdd(tensor_offset, tensor_bytes, "optimizer tensor end offset");
                if (actual_file_bytes < tensor_end) {
                    const uint64_t available_bytes = actual_file_bytes > tensor_offset
                        ? actual_file_bytes - tensor_offset
                        : 0;
                    throw std::runtime_error(
                        std::string("[loadOptimizerState] Truncated ") + tensor_name +
                        " data at group " + std::to_string(i) +
                        ": path=\"" + sidecar_path +
                        "\", group_name=\"" + entries[i].name +
                        "\", group_numel=" + std::to_string(entries[i].numel) +
                        ", groups=" + std::to_string(header.num_groups) +
                        ", tensor_offset=" + std::to_string(tensor_offset) +
                        ", tensor_bytes=" + std::to_string(tensor_bytes) +
                        ", tensor_available_bytes=" + std::to_string(available_bytes) +
                        ", tensor_missing_bytes=" +
                        std::to_string(tensor_bytes - available_bytes) +
                        ", actual_file_bytes=" + std::to_string(actual_file_bytes) +
                        ", expected_file_bytes=" + std::to_string(expected_file_bytes) +
                        ", total_missing_bytes=" +
                        std::to_string(expected_file_bytes - actual_file_bytes) +
                        ", tensor_data_offset=" + std::to_string(tensor_data_offset));
                }
                tensor_offset = tensor_end;
            }
        }
    } else if (actual_file_bytes > expected_file_bytes) {
        EmitModuleWarning(ModuleId::Checkpoint,
            "[loadOptimizerState] Sidecar has trailing data: path=\"" + sidecar_path +
            "\", actual_file_bytes=" + std::to_string(actual_file_bytes) +
            ", expected_file_bytes=" + std::to_string(expected_file_bytes) +
            ", trailing_bytes=" +
            std::to_string(actual_file_bytes - expected_file_bytes),
            ctx.global_step);
    }

    // Read bulk tensor data (H2D per group)
    cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();

    auto readTensor = [&](char* destination,
                          size_t bytes,
                          const char* tensor_name,
                          uint32_t group_index) {
        const std::streampos raw_offset = in.tellg();
        const uint64_t offset = raw_offset == std::streampos(-1)
            ? 0
            : static_cast<uint64_t>(static_cast<std::streamoff>(raw_offset));
        in.read(destination, static_cast<std::streamsize>(bytes));
        const uint64_t bytes_read = in.gcount() < 0
            ? 0
            : static_cast<uint64_t>(in.gcount());
        if (bytes_read != static_cast<uint64_t>(bytes)) {
            throw std::runtime_error(
                std::string("[loadOptimizerState] Truncated ") + tensor_name +
                " data at group " + std::to_string(group_index) +
                ": path=\"" + sidecar_path +
                "\", group_name=\"" + entries[group_index].name +
                "\", group_numel=" + std::to_string(entries[group_index].numel) +
                ", offset=" + std::to_string(offset) +
                ", requested_bytes=" + std::to_string(bytes) +
                ", bytes_read=" + std::to_string(bytes_read) +
                ", actual_file_bytes_at_preflight=" +
                std::to_string(actual_file_bytes) +
                ", expected_file_bytes=" + std::to_string(expected_file_bytes) +
                ", stream_state={" + streamState(in) + "}");
        }
    };

    for (uint32_t i = 0; i < header.num_groups; ++i) {
        const size_t numel = static_cast<size_t>(entries[i].numel);
        if (numel == 0) continue;

        const size_t bytes = numel * sizeof(float);
        std::vector<float> host_buf(numel);

        // m state
        readTensor(
            reinterpret_cast<char*>(host_buf.data()), bytes, "m_state", i);
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
        readTensor(
            reinterpret_cast<char*>(host_buf.data()), bytes, "v_state", i);
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
