//======================================================//
//  TrainingTensors.cu
//  Implementation of autograd-enabled tensor storage
//======================================================//

#include "TrainingTensors.hpp"
#include <stdexcept>
#include <cmath>

#ifdef USE_CUDA

namespace GRIM {

void TrainingTensors::initializeParams(
    int vocab_sz, int d_mod, int d_ffn,
    int n_layers, int n_heads, int n_kv_heads,
    int max_seq, bool tie_emb, bool bias,
    HyperParameters::PositionalEncodingType positional_encoding,
    bool use_layer_scale_flag,
    float layer_scale_init_val,
    cudaStream_t stream
) {
    // Store config
    vocab_size = vocab_sz;
    d_model = d_mod;
    d_ff = d_ffn;
    num_layers = n_layers;
    num_heads = n_heads;
    num_kv_heads = n_kv_heads;
    head_dim = d_model / num_heads;
    max_seq_len = max_seq;
    tie_embeddings = tie_emb;
    use_bias = bias;
    use_layer_scale = use_layer_scale_flag;
    layer_scale_init = layer_scale_init_val;
    
    // GQA dimensions
    const int kv_dim = num_kv_heads * head_dim;
    const int total_qkv_dim = d_model + (2 * kv_dim);  // Q + K + V
    
    //==================================================//
    //  EMBEDDING LAYER
    //==================================================//
    
    // Token embeddings [vocab_size, d_model]
    embedding_weights = Tensor::zeros({vocab_size, d_model}, stream);
    embedding_weights.requires_grad_();
    embedding_weights.ensure_grad();  // Allocate grad NOW so share_grad() works
    Tensor::xavier_uniform_(embedding_weights, stream);
    
    // ISSUE #96 FIX: Position embeddings ONLY allocated for LEARNED positional encoding.
    // Current modes (NONE, ALIBI, ROPE, ALIBI_ROPE) all use attention-based position
    // encoding - they do NOT add learned position embeddings to token embeddings.
    // The position_embedding_weights will be left UNINITIALIZED (null data pointer)
    // which tells AutogradTraining.cu to SKIP the position embedding addition.
    //
    // If a LEARNED positional encoding mode is added later, add:
    //   if (positional_encoding == PositionalEncodingType::LEARNED) { ... }
    //
    // For now, this block is NEVER executed because no LEARNED mode exists.
    // Rule 20: Don't silently allocate - config must explicitly enable features.
    fprintf(stdout, "[TrainingTensors] Position embeddings: SKIPPED (positional_encoding=%s uses attention-based encoding)\n",
            HyperParameters::positionalEncodingTypeToString(positional_encoding));
    // position_embedding_weights intentionally left uninitialized (data=nullptr)
    
    //==================================================//        
    //  LM HEAD
    //==================================================//
    
    if (tie_embeddings) {
        // WEIGHT TYING: LM head shares BOTH data AND grad with embedding
        // This is critical - both must be aliased for:
        // 1. Forward: lm_head_weights.data == embedding_weights.data (shared projection)
        // 2. Backward: lm_head_weights.grad == embedding_weights.grad (accumulates both gradients)
        // 3. Optimizer: Only one buffer to update
        lm_head_weights = Tensor::from_ptr(
            embedding_weights.data,
            {vocab_size, d_model},
            stream
        );
        // CRITICAL: Share the grad Tensor object, NOT separate allocation!
        // ISSUE #59: Use share_grad() for proper shared_ptr semantics
        lm_head_weights.share_grad(embedding_weights);
        lm_head_weights.owns_data = false;  // embedding_weights owns it
        lm_head_weights.requires_grad = true;
        
        // Debug: Verify weight tying is correct at source
        fprintf(stdout, "[TrainingTensors] Weight tying: emb.data=%p lm.data=%p (same=%s)\n",
                (void*)embedding_weights.data, (void*)lm_head_weights.data,
                (embedding_weights.data == lm_head_weights.data) ? "YES" : "NO");
        fprintf(stdout, "[TrainingTensors] Weight tying: emb.grad=%p lm.grad=%p (same=%s)\n",
                (void*)embedding_weights.grad_data(), (void*)lm_head_weights.grad_data(),
                (embedding_weights.grad_data() == lm_head_weights.grad_data()) ? "YES" : "NO");
    } else {
        // Separate LM head weights
        lm_head_weights = Tensor::zeros({vocab_size, d_model}, stream);
        lm_head_weights.requires_grad_();
        Tensor::xavier_uniform_(lm_head_weights, stream);
    }
    
    if (use_bias) {
        lm_head_bias = Tensor::zeros({vocab_size}, stream);
        lm_head_bias.requires_grad_();
    }
    
    //==================================================//
    //  NUMERIC HEAD (optional)
    //==================================================//
    
    numeric_head_weights = Tensor::zeros({d_model}, stream);
    numeric_head_weights.requires_grad_();
    // Initialize with small values
    const float numeric_scale = 1.0f / std::sqrt(static_cast<float>(d_model));
    // TODO: Fill with uniform(-scale, scale)
    
    numeric_head_bias = Tensor::zeros({1}, stream);
    numeric_head_bias.requires_grad_();
    
    //==================================================//
    //  FINAL RMSNORM
    //==================================================//
    
    final_rms_gamma = Tensor::zeros({d_model}, stream);
    final_rms_gamma.requires_grad_();
    // Initialize to 1.0
    {
        std::vector<float> ones(d_model, 1.0f);
        cudaMemcpyAsync(final_rms_gamma.data, ones.data(),
                        d_model * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }
    
    //==================================================//
    //  ENCODER LAYERS
    //==================================================//
    
    encoder_layers.resize(n_layers);
    
    for (int layer = 0; layer < n_layers; ++layer) {
        EncoderLayerParams& params = encoder_layers[layer];
        
        // RMSNorm gammas initialized to 1.0
        params.rms1_gamma = Tensor::zeros({d_model}, stream);
        params.rms1_gamma.requires_grad_();
        
        params.rms2_gamma = Tensor::zeros({d_model}, stream);
        params.rms2_gamma.requires_grad_();
        
        // Fill with 1.0
        {
            std::vector<float> ones(d_model, 1.0f);
            cudaMemcpyAsync(params.rms1_gamma.data, ones.data(),
                            d_model * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(params.rms2_gamma.data, ones.data(),
                            d_model * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
        }
        
        // Attention QKV projection [total_qkv_dim, d_model]
        params.attn_qkv_weight = Tensor::zeros({total_qkv_dim, d_model}, stream);
        params.attn_qkv_weight.requires_grad_();
        Tensor::xavier_uniform_(params.attn_qkv_weight, stream);
        
        if (use_bias) {
            params.attn_qkv_bias = Tensor::zeros({total_qkv_dim}, stream);
            params.attn_qkv_bias.requires_grad_();
        }
        
        // Attention output projection [d_model, d_model]
        params.attn_out_weight = Tensor::zeros({d_model, d_model}, stream);
        params.attn_out_weight.requires_grad_();
        Tensor::xavier_uniform_(params.attn_out_weight, stream);
        
        if (use_bias) {
            params.attn_out_bias = Tensor::zeros({d_model}, stream);
            params.attn_out_bias.requires_grad_();
        }
        
        // QK-norm learned scales (nGPT)
        params.alpha_q = Tensor::zeros({num_heads}, stream);
        params.alpha_q.requires_grad_();
        // Initialize to 1.0
        {
            std::vector<float> ones(num_heads, 1.0f);
            cudaMemcpyAsync(params.alpha_q.data, ones.data(),
                            num_heads * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
        }
        
        params.alpha_k = Tensor::zeros({num_kv_heads}, stream);
        params.alpha_k.requires_grad_();
        {
            std::vector<float> ones(num_kv_heads, 1.0f);
            cudaMemcpyAsync(params.alpha_k.data, ones.data(),
                            num_kv_heads * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
        }
        
        // FFN W1 [d_model, d_ff] - up-projection (matches FeedForwardLayer::useExternalWeights)
        // Issue #89 FIX: Was [d_ff, d_model] - swapped to match consumer's expectation
        params.ffn_w1 = Tensor::zeros({d_model, d_ff}, stream);
        params.ffn_w1.requires_grad_();
        Tensor::xavier_uniform_(params.ffn_w1, stream);
        
        if (use_bias) {
            params.ffn_b1 = Tensor::zeros({d_ff}, stream);
            params.ffn_b1.requires_grad_();
        }
        
        // FFN W2 [d_ff, d_model] - down-projection (matches FeedForwardLayer::useExternalWeights)
        // Issue #89 FIX: Was [d_model, d_ff] - swapped to match consumer's expectation
        params.ffn_w2 = Tensor::zeros({d_ff, d_model}, stream);
        params.ffn_w2.requires_grad_();
        Tensor::xavier_uniform_(params.ffn_w2, stream);
        
        if (use_bias) {
            params.ffn_b2 = Tensor::zeros({d_model}, stream);
            params.ffn_b2.requires_grad_();
        }
        
        // LayerScale (Issue #109) - learnable scalars that multiply sublayer outputs before residual
        if (use_layer_scale) {
            // layer_scale1: scales attention output before residual
            params.layer_scale1 = Tensor::zeros({1}, stream);
            params.layer_scale1.requires_grad_();
            cudaMemcpyAsync(params.layer_scale1.data, &layer_scale_init, sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            
            // layer_scale2: scales FFN output before residual
            params.layer_scale2 = Tensor::zeros({1}, stream);
            params.layer_scale2.requires_grad_();
            cudaMemcpyAsync(params.layer_scale2.data, &layer_scale_init, sizeof(float),
                            cudaMemcpyHostToDevice, stream);
        }
    }
    
    initialized_ = true;
    
    // PRE-ALLOCATE ALL GRADIENT BUFFERS
    // This is required because GradAccumulationController::bindToModel() reads
    // the .grad pointers at startup, and lazy allocation won't have happened yet.
    allocateAllGradients();
    
    // Sync to ensure all initialization is complete
    if (stream) {
        cudaStreamSynchronize(stream);
    }
}

void TrainingTensors::allocateAllGradients() {
    // Global tensors
    embedding_weights.ensure_grad();
    position_embedding_weights.ensure_grad();
    lm_head_weights.ensure_grad();
    if (lm_head_bias.data) lm_head_bias.ensure_grad();
    if (numeric_head_weights.data) numeric_head_weights.ensure_grad();
    if (numeric_head_bias.data) numeric_head_bias.ensure_grad();
    if (final_rms_gamma.data) final_rms_gamma.ensure_grad();
    
    // Encoder layer tensors
    for (auto& layer : encoder_layers) {
        layer.rms1_gamma.ensure_grad();
        layer.rms2_gamma.ensure_grad();
        layer.attn_qkv_weight.ensure_grad();
        if (layer.attn_qkv_bias.data) layer.attn_qkv_bias.ensure_grad();
        layer.attn_out_weight.ensure_grad();
        if (layer.attn_out_bias.data) layer.attn_out_bias.ensure_grad();
        layer.alpha_q.ensure_grad();
        layer.alpha_k.ensure_grad();
        layer.ffn_w1.ensure_grad();
        if (layer.ffn_b1.data) layer.ffn_b1.ensure_grad();
        layer.ffn_w2.ensure_grad();
        if (layer.ffn_b2.data) layer.ffn_b2.ensure_grad();
        // LayerScale (Issue #109)
        if (layer.layer_scale1.data) layer.layer_scale1.ensure_grad();
        if (layer.layer_scale2.data) layer.layer_scale2.ensure_grad();
    }
    
    fprintf(stdout, "[INFO] TrainingTensors: Pre-allocated gradient buffers for %zu encoder layers\n",
            encoder_layers.size());
}


void TrainingTensors::allocateCaches(int batch_size, int seq_len, cudaStream_t stream) {
    if (!initialized_) {
        throw std::runtime_error("TrainingTensors::allocateCaches: params not initialized");
    }
    
    const int total_tokens = batch_size * seq_len;
    const int kv_dim = num_kv_heads * head_dim;
    
    //==================================================//
    //  INPUT LAYER CACHES
    //==================================================//
    
    cached_embeddings = Tensor::zeros({total_tokens, d_model}, stream);
    
    //==================================================//
    //  ENCODER LAYER CACHES
    //==================================================//
    
    encoder_caches.resize(num_layers);
    
    for (int layer = 0; layer < num_layers; ++layer) {
        EncoderLayerCache& cache = encoder_caches[layer];
        
        cache.ln1_output = Tensor::zeros({total_tokens, d_model}, stream);
        cache.attn_output = Tensor::zeros({total_tokens, d_model}, stream);
        cache.residual1_output = Tensor::zeros({total_tokens, d_model}, stream);
        cache.ln2_output = Tensor::zeros({total_tokens, d_model}, stream);
        cache.ffn_pre_gelu = Tensor::zeros({total_tokens, d_ff}, stream);
        cache.ffn_output = Tensor::zeros({total_tokens, d_model}, stream);
        cache.layer_output = Tensor::zeros({total_tokens, d_model}, stream);
        
        // QKV caches in BHSD format
        cache.Q = Tensor::zeros({batch_size, num_heads, seq_len, head_dim}, stream);
        cache.K = Tensor::zeros({batch_size, num_kv_heads, seq_len, head_dim}, stream);
        cache.V = Tensor::zeros({batch_size, num_kv_heads, seq_len, head_dim}, stream);
        cache.attn_input = Tensor::zeros({total_tokens, d_model}, stream);
        cache.attn_bhsd = Tensor::zeros({batch_size, num_heads, seq_len, head_dim}, stream);
        
        // Softmax LSE for FlashAttention
        cache.softmax_lse = Tensor::zeros({batch_size, num_heads, seq_len}, stream);
    }
    
    //==================================================//
    //  OUTPUT LAYER CACHES
    //==================================================//
    
    cached_encoder_output = Tensor::zeros({total_tokens, d_model}, stream);
    cached_logits = Tensor::zeros({total_tokens, vocab_size}, stream);
    cached_final_rms_input = Tensor::zeros({total_tokens, d_model}, stream);
    
    //==================================================//
    //  GRADIENT TEMPORARIES
    //==================================================//
    
    grad_logits = Tensor::zeros({total_tokens, vocab_size}, stream);
    grad_encoder_out = Tensor::zeros({total_tokens, d_model}, stream);
    
    grad_ffn_input = Tensor::zeros({total_tokens, d_model}, stream);
    grad_ffn_hidden = Tensor::zeros({total_tokens, d_ff}, stream);
    grad_attn_input = Tensor::zeros({total_tokens, d_model}, stream);
    grad_attn_out_before_proj = Tensor::zeros({total_tokens, d_model}, stream);
    grad_attn_out_reshaped = Tensor::zeros({batch_size, num_heads, seq_len, head_dim}, stream);
    grad_q = Tensor::zeros({batch_size, num_heads, seq_len, head_dim}, stream);
    grad_k = Tensor::zeros({batch_size, num_kv_heads, seq_len, head_dim}, stream);
    grad_v = Tensor::zeros({batch_size, num_kv_heads, seq_len, head_dim}, stream);
    
    const int total_qkv_dim = d_model + 2 * kv_dim;
    grad_qkv_concat = Tensor::zeros({total_tokens, total_qkv_dim}, stream);
    grad_qkv_input = Tensor::zeros({total_tokens, d_model}, stream);
    grad_attn_bsm_scratch = Tensor::zeros({batch_size, seq_len, d_model}, stream);
    
    // Issue #43 centering scratch (max of d_model and d_ff)
    const int scratch_dim = (d_ff > d_model) ? d_ff : d_model;
    centered_activation_scratch = Tensor::zeros({total_tokens, scratch_dim}, stream);
}


void TrainingTensors::zeroGrad(cudaStream_t stream) {
    if (!initialized_) return;
    
    // Zero embedding grads
    embedding_weights.zero_grad(stream);
    position_embedding_weights.zero_grad(stream);
    
    // Zero LM head grads
    if (!tie_embeddings) {
        lm_head_weights.zero_grad(stream);
    }
    if (lm_head_bias.data) {
        lm_head_bias.zero_grad(stream);
    }
    
    // Zero numeric head grads
    numeric_head_weights.zero_grad(stream);
    numeric_head_bias.zero_grad(stream);
    
    // Zero final RMSNorm grad
    final_rms_gamma.zero_grad(stream);
    
    // Zero encoder layer grads
    for (auto& layer : encoder_layers) {
        layer.rms1_gamma.zero_grad(stream);
        layer.rms2_gamma.zero_grad(stream);
        layer.attn_qkv_weight.zero_grad(stream);
        if (layer.attn_qkv_bias.data) layer.attn_qkv_bias.zero_grad(stream);
        layer.attn_out_weight.zero_grad(stream);
        if (layer.attn_out_bias.data) layer.attn_out_bias.zero_grad(stream);
        layer.alpha_q.zero_grad(stream);
        layer.alpha_k.zero_grad(stream);
        layer.ffn_w1.zero_grad(stream);
        if (layer.ffn_b1.data) layer.ffn_b1.zero_grad(stream);
        layer.ffn_w2.zero_grad(stream);
        if (layer.ffn_b2.data) layer.ffn_b2.zero_grad(stream);
    }
}


float* TrainingTensors::getParamData(const std::string& name) {
    // Legacy compatibility - map name to tensor
    if (name == "embedding_weights") return embedding_weights.data;
    if (name == "position_embedding_weights") return position_embedding_weights.data;
    if (name == "lm_head_weights") return lm_head_weights.data;
    if (name == "lm_head_bias") return lm_head_bias.data;
    if (name == "numeric_head_weights") return numeric_head_weights.data;
    if (name == "numeric_head_bias") return numeric_head_bias.data;
    if (name == "final_rms_gamma") return final_rms_gamma.data;
    
    // For per-layer params, use getLayerParamData(name, layer)
    throw std::runtime_error("TrainingTensors::getParamData: unknown param '" + name + "'");
}


float* TrainingTensors::getGradData(const std::string& name) {
    // Legacy compatibility - map name to gradient
    // ISSUE #59: Use grad_data() accessor
    if (name == "embedding_weights") return embedding_weights.grad_data();
    if (name == "position_embedding_weights") return position_embedding_weights.grad_data();
    if (name == "lm_head_weights") return lm_head_weights.grad_data();
    if (name == "lm_head_bias") return lm_head_bias.grad_data();
    if (name == "numeric_head_weights") return numeric_head_weights.grad_data();
    if (name == "numeric_head_bias") return numeric_head_bias.grad_data();
    if (name == "final_rms_gamma") return final_rms_gamma.grad_data();
    
    throw std::runtime_error("TrainingTensors::getGradData: unknown param '" + name + "'");
}

}  // namespace GRIM

#endif  // USE_CUDA
