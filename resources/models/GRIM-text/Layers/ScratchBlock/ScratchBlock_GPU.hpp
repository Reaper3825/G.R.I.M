//======================================================//
//  ScratchBlock_GPU.hpp
//  Internal Reasoning Layer for GRIM-text
//  
//  The ScratchBlock layer provides an internal reasoning
//  space where the model can store and retrieve structured
//  atoms (numbers, URLs, dates, etc.) during generation.
//  
//  Features:
//  - Togglable at runtime (model works with or without)
//  - AtomTable integration for structured reasoning
//  - Zero overhead when disabled (passthrough)
//  - GPU-accelerated atom lookup and embedding
//  
//  When ENABLED:
//  - Input tokens are scanned for atom placeholders
//  - Atoms are looked up from AtomTable
//  - Atom embeddings are injected into hidden states
//  - Model can reason over structured representations
//  
//  When DISABLED:
//  - Layer acts as pure passthrough (zero overhead)
//  - No atom detection or lookup
//  - Normal token flow unchanged
//  
//  Author: GRIM Team
//  Date: December 2025
//======================================================//

#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "../grim_layer_gpu.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

//======================================================//
//  ScratchBlock Configuration
//======================================================//
struct ScratchBlockConfig {
    int d_model = 0;              // Hidden dimension
    int max_atoms = 0;            // Max atoms per sequence
    int atom_embedding_dim = 0;    // Dimension for atom type embeddings
    int atom_token_start = 256;     // Atom token range start
    int atom_token_end = 512;       // Atom token range end (exclusive)
    bool enabled = true;            // Master toggle
    bool inject_atom_embeddings = true;  // Add atom embeddings to hidden states
    float atom_scale = 1.0f;        // Scale factor for atom injection
    cudaStream_t stream = nullptr;
};

//======================================================//
//  Forward Arguments
//======================================================//
struct ScratchBlockForwardArgs {
    // Input/Output (passthrough when disabled)
    // TensorView: [total_tokens, d_model] BSM layout
    TensorContract::TensorView input;    // const - read only
    TensorContract::TensorView output;   // mutable - write output
    int total_tokens = 0;
    int seq_len = 0;                // Optional (if 0, assume batch=1)
    
    // Token IDs for atom detection
    const int* token_ids = nullptr; // [total_tokens] - to detect atom placeholders
    
    // Per-token numeric side channel (aligned to token_ids)
    const float* token_numeric_values = nullptr;  // [total_tokens] - 1D, not TensorView
    const uint8_t* token_numeric_mask = nullptr;  // [total_tokens]
    
    // Per-token text features (GRMT v4)
    const uint16_t* token_text_features = nullptr;  // [total_tokens * kTextFeatureDim] FP16
    const uint8_t* token_text_mask = nullptr;       // [total_tokens] - 1 if atom has text features
    
    // Stream
    cudaStream_t stream = nullptr;
    
    // Activation cache (optional, for backward pass)
    // TensorView: [num_atoms, atom_embedding_dim] BSM layout
    TensorContract::TensorView cache_atom_embeddings;
    int* cache_atom_positions = nullptr;     // [num_atoms] - which tokens have atoms
    int* cache_atom_types = nullptr;         // [num_atoms] - atom type index for each atom
    int* cache_num_atoms = nullptr;          // Scalar - count of atoms in sequence
    
    // RULE 20: Validate required fields - throws on NULL pointers
    void validate(const char* context) const {
        input.require(context);
        output.require(context);
        if (!token_ids) {
            throw std::runtime_error(std::string(context) + ": token_ids is NULL");
        }
        if (!stream) {
            throw std::runtime_error(std::string(context) + ": stream is NULL");
        }
    }
};

//======================================================//
//  ScratchBlock Layer
//======================================================//
class ScratchBlockLayer final : public Layer<ScratchBlockLayer, float> {
public:
    static constexpr LayerType layer_type = LayerType::kUnknown;  // Custom layer type

    ScratchBlockLayer();
    explicit ScratchBlockLayer(const ScratchBlockConfig& config);
    ~ScratchBlockLayer();
    
    // Disable copy
    ScratchBlockLayer(const ScratchBlockLayer&) = delete;
    ScratchBlockLayer& operator=(const ScratchBlockLayer&) = delete;
    
    // Move support
    ScratchBlockLayer(ScratchBlockLayer&&) noexcept;
    ScratchBlockLayer& operator=(ScratchBlockLayer&&) noexcept;

    //--------------------------------------------------//
    // Configuration
    //--------------------------------------------------//
    
    void setConfig(const ScratchBlockConfig& config);
    const ScratchBlockConfig& config() const noexcept { return config_; }
    
    // Runtime enable/disable (zero overhead when disabled)
    void setEnabled(bool enabled);
    bool isEnabled() const noexcept { return config_.enabled; }

    //--------------------------------------------------//
    // Forward/Backward
    //--------------------------------------------------//
    
    // Main forward pass
    // When enabled: detects atoms, injects embeddings
    // When disabled: pure passthrough (output = input)
    void forward(const ScratchBlockForwardArgs& args);
    
    // Backward pass (gradient flow)
    void backward(const ScratchBlockForwardArgs& args,
                  const float* grad_output,
                  float* grad_input);

    // Layer interface implementation
    void onConfigure(const Dimensions& dims);

    //--------------------------------------------------//
    // Memory
    //--------------------------------------------------//
    
    std::size_t requiredWorkspaceBytes(int total_tokens) const;
    
    // Atom type embeddings (learnable Tensor)
    Tensor& atomTypeEmbeddings() { return atom_type_embeddings_; }
    
    // Projection weights (atom_embedding_dim → d_model)
    Tensor& atomProjection() { return atom_projection_; }
    
    // Text feature projection (16-dim FP16 → d_model) - VALUE encoding path
    Tensor& textFeatureProjection() { return text_feature_projection_; }

    //--------------------------------------------------//
    // Statistics
    //--------------------------------------------------//
    
    struct Stats {
        size_t total_forward_calls = 0;
        size_t total_atoms_processed = 0;
        size_t passthrough_calls = 0;     // When disabled
        size_t active_calls = 0;          // When enabled
    };
    
    const Stats& stats() const noexcept { return stats_; }
    Stats getStats() const noexcept { return stats_; }
    void resetStats() { stats_ = Stats{}; }
    
    // Logging control
    void setLoggingEnabled(bool enabled) { logging_enabled_ = enabled; }
    bool isLoggingEnabled() const noexcept { return logging_enabled_; }
    void setGlobalStep(std::uint64_t step) { global_step_ = step; }
    std::uint64_t globalStep() const noexcept { return global_step_; }
    const ScratchBlockConfig& getConfig() const noexcept { return config_; }

private:
    void allocateWeights();
    void freeWeights();
    void initializeWeights();
    
    // Passthrough (when disabled)
    void forwardPassthrough(const ScratchBlockForwardArgs& args);
    
    // Active mode (when enabled)
    void forwardActive(const ScratchBlockForwardArgs& args);
    
    // Detect atom tokens in sequence (async, device counter)
    void detectAtomTokensAsync(const int* token_ids,
                               int total_tokens,
                               int* out_positions,
                               int max_atoms,
                               int* out_num_atoms_device,
                               cudaStream_t stream);
    
    // Inject atom embeddings into hidden states
    void injectAtomEmbeddings(float* hidden_states,
                              int total_tokens,
                              const int* atom_positions,
                              int num_atoms,
                              const float* token_numeric_values,
                              const uint8_t* token_numeric_mask,
                              cudaStream_t stream);

    ScratchBlockConfig config_;
    Stats stats_;
    
    // GPU weights (Tensor-managed — RAII, gradient via ensure_grad)
    Tensor atom_type_embeddings_;         // [num_atom_types, atom_embedding_dim]
    Tensor atom_projection_;              // [atom_embedding_dim, d_model]
    Tensor text_feature_projection_;      // [16, d_model] - text feature value encoding
    
    // Temporary buffers
    int* d_atom_positions_ = nullptr;              // [max_atoms]
    int* d_num_atoms_ = nullptr;                   // Scalar
    float* d_atom_embeddings_ = nullptr;           // [max_atoms, atom_embedding_dim]
    float* d_grad_atom_embeddings_ = nullptr;      // [max_atoms, atom_embedding_dim]
    
    bool weights_allocated_ = false;
    bool logging_enabled_ = false;
    std::uint64_t global_step_ = 0;
    
    // Logging helpers
    void logForward(int num_atoms, float duration_ms);
    void logBackward(float duration_ms);
    void logWeightInit();
    void logConfigChange(const char* param, float old_val, float new_val);
};

} // namespace GRIM
