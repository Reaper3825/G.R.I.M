//======================================================//
//  OptimizerCheckpoint.hpp
//  Save/load optimizer state as a binary sidecar file
//======================================================//
//
//  PURPOSE
//  =======
//  Enables true training resume by persisting AdamW optimizer
//  state (m/v moment buffers, step counter, global_step) alongside
//  model weight checkpoints.
//
//  SIDECAR CONVENTION
//  ==================
//  For a model checkpoint at:  checkpoint_epoch_7.bin
//  The optimizer sidecar is:   checkpoint_epoch_7.opt
//
//  BINARY FORMAT (v1)
//  ==================
//  [Header]
//    magic:          uint32  = 0x47524F50  ("GROP" = GRIM Optimizer)
//    version:        uint32  = 1
//    num_groups:     uint32  = number of parameter groups
//    optimizer_step: int32   = AdamW bias-correction step counter
//    global_step:    int32   = training batch counter
//    best_val_loss:  float32 = best validation loss seen
//    current_epoch:  int32   = epochs completed so far
//    accumulation_slot: int32 = accumulation-slot cursor
//    reserved:       uint8[32] = zero (future expansion)
//
//  [Per-group directory] (num_groups entries)
//    name_len:       uint16  = length of param group name
//    name:           char[]  = param group name (NOT null-terminated)
//    numel:          uint64  = number of elements in m/v buffers
//
//  [Bulk tensor data]
//    For each group i:
//      m_data[i]:    float[numel_i]  = first moment buffer
//      v_data[i]:    float[numel_i]  = second moment buffer
//
//  VALIDATION
//  ==========
//  On load, the sidecar is validated against the current model:
//  - num_groups must match
//  - Each group name and numel must match the current parameter groups
//  - Mismatches throw (Rule 20: fail loud, no silent fallback)
//
//  Author: Austin Wadkins
//  Date: April 2026
//  Version: 1.0.0
//======================================================//

#pragma once

#include <string>

namespace GRIMText::Training {

// Forward declaration
struct TrainingContext;

//======================================================//
//  Public API
//======================================================//

/**
 * @brief Derive the optimizer sidecar path from a model checkpoint path.
 *
 * Replaces ".bin" extension with ".opt".
 * Throws if the input path does not end with ".bin".
 *
 * @param checkpoint_path Path to the model checkpoint (.bin)
 * @return Corresponding optimizer sidecar path (.opt)
 */
std::string optimizerSidecarPath(const std::string& checkpoint_path);

/**
 * @brief Save optimizer state to a binary sidecar file.
 *
 * Writes AdamW moment buffers (m, v), step counter, global_step,
 * best_val_loss, epochs_completed, and the accumulation-slot cursor.
 *
 * GPU tensors are downloaded to host via cudaMemcpy before writing.
 *
 * @param ctx Training context (provides model, optimizer, training state)
 * @param sidecar_path Output file path (typically from optimizerSidecarPath)
 * @return true on success, false on I/O failure
 */
bool saveOptimizerState(const TrainingContext& ctx, const std::string& sidecar_path);

/**
 * @brief Load optimizer state from a binary sidecar file.
 *
 * Restores AdamW moment buffers (m, v), step counter, global_step,
 * best_val_loss, epochs_completed, and the accumulation-slot cursor.
 *
 * Validates that the sidecar matches the current model parameter groups
 * (count, names, sizes). Mismatches throw std::runtime_error.
 *
 * GPU tensors are uploaded from host via cudaMemcpyAsync on the primary stream.
 *
 * @param ctx Training context to restore into
 * @param sidecar_path Input file path
 * @return true if loaded successfully, false if file doesn't exist
 * @throws std::runtime_error on format/validation errors
 */
bool loadOptimizerState(TrainingContext& ctx, const std::string& sidecar_path);

} // namespace GRIMText::Training
