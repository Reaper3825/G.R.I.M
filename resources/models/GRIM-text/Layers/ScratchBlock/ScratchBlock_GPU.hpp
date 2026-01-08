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
    const float* input = nullptr;   // [total_tokens, d_model]
    float* output = nullptr;        // [total_tokens, d_model]
    int total_tokens = 0;
    int seq_len = 0;                // Optional (if 0, assume batch=1)
    
    // Token IDs for atom detection
    const int* token_ids = nullptr; // [total_tokens] - to detect atom placeholders
    
    // Per-token numeric side channel (aligned to token_ids)
    const float* token_numeric_values = nullptr;  // [total_tokens]
    const uint8_t* token_numeric_mask = nullptr;  // [total_tokens]
    
    // Per-token text features (GRMT v4)
    const uint16_t* token_text_features = nullptr;  // [total_tokens * kTextFeatureDim] FP16
    const uint8_t* token_text_mask = nullptr;       // [total_tokens] - 1 if atom has text features
    
    // Stream
    cudaStream_t stream = nullptr;
    
    // Activation cache (optional, for backward pass)
    float* cache_atom_embeddings = nullptr;  // [num_atoms, atom_embedding_dim]
    int* cache_atom_positions = nullptr;     // [num_atoms] - which tokens have atoms
    int* cache_atom_types = nullptr;         // [num_atoms] - atom type index for each atom
    int* cache_num_atoms = nullptr;          // Scalar - count of atoms in sequence
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
    void forwardImpl(const LayerIO<float>& io, LayerWorkspace<float>* workspace);
    void backwardImpl(const LayerIO<float>& io, LayerWorkspace<float>* workspace);

    //--------------------------------------------------//
    // Memory
    //--------------------------------------------------//
    
    std::size_t requiredWorkspaceBytes(int total_tokens) const;
    
    // Atom type embeddings (learnable)
    float* getAtomTypeEmbeddings() { return d_atom_type_embeddings_; }
    float* getAtomTypeEmbeddingsGrad() { return d_atom_type_embeddings_grad_; }
    
    // Projection weights (atom_embedding_dim → d_model)
    float* getAtomProjection() { return d_atom_projection_; }
    float* getAtomProjectionGrad() { return d_atom_projection_grad_; }
    
    // Text feature projection (16-dim FP16 → d_model) - VALUE encoding path
    float* getTextFeatureProjection() { return d_text_feature_projection_; }
    float* getTextFeatureProjectionGrad() { return d_text_feature_projection_grad_; }

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
    
    // GPU weights
    float* d_atom_type_embeddings_ = nullptr;      // [num_atom_types, atom_embedding_dim]
    float* d_atom_type_embeddings_grad_ = nullptr;
    float* d_atom_projection_ = nullptr;           // [atom_embedding_dim, d_model]
    float* d_atom_projection_grad_ = nullptr;
    float* d_text_feature_projection_ = nullptr;   // [16, d_model] - text feature value encoding
    float* d_text_feature_projection_grad_ = nullptr;
    
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
