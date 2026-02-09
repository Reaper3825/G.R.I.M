//======================================================//
//  ForwardIntermediates.hpp
//  Storage for intermediate tensors during forward pass
//  
//  PURPOSE (Issue #56 Fix):
//  When EncodingLayer::forward() returns, all local Tensor variables
//  are destroyed, which triggers their grad_fn destructors. This 
//  cascades to destroy the entire autograd graph BEFORE backward runs.
//  
//  SOLUTION:
//  This struct stores ALL intermediate tensors from the forward pass.
//  The caller (AutogradTraining.cu) keeps ForwardIntermediates alive
//  until backward completes, preserving the autograd graph.
//  
//  RULE 20: No backwards compatibility - this is the ONLY way to do autograd.
//  The old approach of letting intermediates die was fundamentally broken.
//======================================================//



#pragma once

#include "TensorContract_GPU.hpp"
#include <vector>

namespace GRIM {

/**
 * ForwardIntermediates - Keeps autograd tensors alive during forward-backward cycle
 * 
 * CRITICAL INVARIANT: All Tensors stored here MUST have their autograd state preserved:
 *   - owns_data = true (they own their GPU memory)
 *   - grad_fn != nullptr (linked to computation graph)
 *   - grad_fn is a shared_ptr<GradFn> (lifetime managed automatically)
 * 
 * When forward() stores a Tensor here via std::move(), the Tensor is transferred
 * from the local stack to this storage. The grad_fn chain remains intact because
 * the Tensor objects themselves (and their ownership) are preserved.
 * 
 * LIFECYCLE:
 *   1. Before layer forward: caller creates ForwardIntermediates instance
 *   2. During forward: layer stores all intermediate Tensors via std::move
 *   3. After forward: ForwardIntermediates is stored in AutogradContext
 *   4. During backward: grad_fn->apply() traverses the chain (still intact!)
 *   5. After backward: AutogradContext::clearIntermediates() destroys this struct
 *   6. Destructors run: NOW it's safe to delete grad_fn objects
 */
struct ForwardIntermediates {
    //--------------------------------------------------
    // Per-Layer Storage (one ForwardIntermediates per encoder layer)
    //--------------------------------------------------
    
    // RMSNorm outputs
    Tensor ln1_out;          // Output of first RMSNorm (pre-attention)
    Tensor ln2_out;          // Output of second RMSNorm (pre-FFN)
    
    // QKV Projection outputs
    Tensor qkv_out;          // Fused QKV projection [tokens, d_model + 2*kv_dim]
    Tensor Q;                // Query projection [tokens, d_model]
    Tensor K;                // Key projection [tokens, kv_dim]
    Tensor V;                // Value projection [tokens, kv_dim]
    
    // Reshaped attention tensors (BHSD format)
    Tensor Q_bhsd;           // [batch, heads, seq, head_dim]
    Tensor K_bhsd;           // [batch, kv_heads, seq, head_dim]
    Tensor V_bhsd;           // [batch, kv_heads, seq, head_dim]
    
    // Attention output
    Tensor attn_out_bhsd;    // Flash attention output [batch, heads, seq, head_dim]
    Tensor attn_out;         // Reshaped attention output [tokens, d_model]
    
    // Output projection
    Tensor proj_out;         // W_o projection output [tokens, d_model]
    
    // Residual connection outputs
    Tensor residual1;        // input + proj_out
    
    // FFN outputs
    Tensor ffn_out;          // FFN layer output [tokens, d_model]
    
    // Final output (also stored in layer_outputs in AutogradContext)
    // This is the only tensor that would have survived the old approach
    Tensor output;           // residual1 + ffn_out
    
    //--------------------------------------------------
    // FFN internal intermediates (if FFN uses autograd internally)
    //--------------------------------------------------
    Tensor ffn_linear1_out;  // W1 @ input
    Tensor ffn_gelu_out;     // GELU(ffn_linear1_out)
    
    //--------------------------------------------------
    // Lifecycle Management
    //--------------------------------------------------
    
    ForwardIntermediates() = default;
    ~ForwardIntermediates() = default;
    
    // Movable
    ForwardIntermediates(ForwardIntermediates&&) = default;
    ForwardIntermediates& operator=(ForwardIntermediates&&) = default;
    
    // Non-copyable (Tensors are non-copyable)
    ForwardIntermediates(const ForwardIntermediates&) = delete;
    ForwardIntermediates& operator=(const ForwardIntermediates&) = delete;
    
    /**
     * Clear all stored tensors, releasing GPU memory and grad_fn objects.
     * Call this ONLY after backward pass completes.
     */
    void clear() {
        ln1_out = Tensor();
        ln2_out = Tensor();
        qkv_out = Tensor();
        Q = Tensor();
        K = Tensor();
        V = Tensor();
        Q_bhsd = Tensor();
        K_bhsd = Tensor();
        V_bhsd = Tensor();
        attn_out_bhsd = Tensor();
        attn_out = Tensor();
        proj_out = Tensor();
        residual1 = Tensor();
        ffn_out = Tensor();
        output = Tensor();
        ffn_linear1_out = Tensor();
        ffn_gelu_out = Tensor();
    }
    
    /**
     * Debug: Count how many tensors have valid grad_fn attached.
     * Useful for verifying autograd graph is intact.
     */
    int countGradFns() const {
        int count = 0;
        if (ln1_out.grad_fn) count++;
        if (ln2_out.grad_fn) count++;
        if (qkv_out.grad_fn) count++;
        if (Q.grad_fn) count++;
        if (K.grad_fn) count++;
        if (V.grad_fn) count++;
        if (Q_bhsd.grad_fn) count++;
        if (K_bhsd.grad_fn) count++;
        if (V_bhsd.grad_fn) count++;
        if (attn_out_bhsd.grad_fn) count++;
        if (attn_out.grad_fn) count++;
        if (proj_out.grad_fn) count++;
        if (residual1.grad_fn) count++;
        if (ffn_out.grad_fn) count++;
        if (output.grad_fn) count++;
        if (ffn_linear1_out.grad_fn) count++;
        if (ffn_gelu_out.grad_fn) count++;
        return count;
    }
};

/**
 * Container for all layers' intermediates during forward-backward cycle.
 * 
 * Stored in AutogradContext and persists until clearIntermediates() is called.
 */
struct AllLayerIntermediates {
    std::vector<ForwardIntermediates> layers;
    
    // Embedding intermediates (before encoder layers)
    Tensor embedding_lookup_out;  // Raw embedding lookup result
    Tensor position_embedding_out; // Position embedding if separate
    Tensor embedding_combined;     // embedding + position
    
    // Post-encoder intermediates
    Tensor final_rms_out;         // Final RMSNorm output (before LM head)
    
    void reserve(size_t num_layers) {
        layers.reserve(num_layers);
    }
    
    void clear() {
        for (auto& layer : layers) {
            layer.clear();
        }
        layers.clear();
        embedding_lookup_out = Tensor();
        position_embedding_out = Tensor();
        embedding_combined = Tensor();
        final_rms_out = Tensor();
    }
    
    /**
     * Debug: Total grad_fn count across all layers
     */
    int totalGradFnCount() const {
        int count = 0;
        for (const auto& layer : layers) {
            count += layer.countGradFns();
        }
        if (embedding_lookup_out.grad_fn) count++;
        if (position_embedding_out.grad_fn) count++;
        if (embedding_combined.grad_fn) count++;
        if (final_rms_out.grad_fn) count++;
        return count;
    }
};

}  // namespace GRIM
