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
    uint64_t seed,
    cudaStream_t stream
) {
    // ── Validate parameters (Rule 20: crash loud, no silent truncation) ──
    if (n_heads <= 0) {
        throw std::invalid_argument("TrainingTensors::initializeParams: num_heads must be > 0, got " + std::to_string(n_heads));
    }
    if (d_mod % n_heads != 0) {
        throw std::invalid_argument("TrainingTensors::initializeParams: d_model (" + std::to_string(d_mod)
            + ") must be divisible by num_heads (" + std::to_string(n_heads) + ")");
    }
    if (n_kv_heads <= 0) {
        throw std::invalid_argument("TrainingTensors::initializeParams: num_kv_heads must be > 0, got " + std::to_string(n_kv_heads));
    }
    if (n_kv_heads > n_heads) {
        throw std::invalid_argument("TrainingTensors::initializeParams: num_kv_heads (" + std::to_string(n_kv_heads)
            + ") must be <= num_heads (" + std::to_string(n_heads) + ")");
    }
    if (n_heads % n_kv_heads != 0) {
        throw std::invalid_argument("TrainingTensors::initializeParams: num_heads (" + std::to_string(n_heads)
            + ") must be divisible by num_kv_heads (" + std::to_string(n_kv_heads) + ") for GQA");
    }

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
    positional_encoding_type = positional_encoding;
    
    // GQA dimensions
    const int kv_dim = num_kv_heads * head_dim;
    const int total_qkv_dim = d_model + (2 * kv_dim);  // Q + K + V
    
    //==================================================//
    //  EMBEDDING LAYER
    //==================================================//
    
    // Token embeddings [vocab_size, d_model]
    embedding_weights = Tensor::zeros({vocab_size, d_model}, stream, "embedding_weights");
    embedding_weights.requires_grad_();
    embedding_weights.ensure_grad();  // Allocate grad NOW so share_grad() works
    Tensor::xavier_uniform_(embedding_weights, seed + 0, stream);
    
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
            stream,
            "lm_head_weights_tied"
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
        lm_head_weights = Tensor::zeros({vocab_size, d_model}, stream, "lm_head_weights");
        lm_head_weights.requires_grad_();
        Tensor::xavier_uniform_(lm_head_weights, seed + 1, stream);
    }
    
    if (use_bias) {
        lm_head_bias = Tensor::zeros({vocab_size}, stream, "lm_head_bias");
        lm_head_bias.requires_grad_();
    }
    
    //==================================================//
    //  NUMERIC HEAD (optional)
    //==================================================//
    
    numeric_head_weights = Tensor::zeros({d_model}, stream, "numeric_head_weights");
    numeric_head_weights.requires_grad_();
    // Xavier uniform: U[-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out))]
    // For 1D shape {d_model}: fan_in=d_model, fan_out=1 → scale=sqrt(6/(d_model+1))
    Tensor::xavier_uniform_(numeric_head_weights, seed + 99, stream);
    
    numeric_head_bias = Tensor::zeros({1}, stream, "numeric_head_bias");
    numeric_head_bias.requires_grad_();
    
    //==================================================//
    //  FINAL RMSNORM
    //==================================================//
    
    final_rms_gamma = Tensor::zeros({d_model}, stream, "final_rms_gamma");
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
        params.rms1_gamma = Tensor::zeros({d_model}, stream, "rms1_gamma");
        params.rms1_gamma.requires_grad_();
        
        params.rms2_gamma = Tensor::zeros({d_model}, stream, "rms2_gamma");
        params.rms2_gamma.requires_grad_();
        
        // Sandwich norm gammas (post-residual normalization) initialized to 1.0
        params.rms_post_attn_gamma = Tensor::zeros({d_model}, stream, "rms_post_attn_gamma");
        params.rms_post_attn_gamma.requires_grad_();
        
        params.rms_post_ffn_gamma = Tensor::zeros({d_model}, stream, "rms_post_ffn_gamma");
        params.rms_post_ffn_gamma.requires_grad_();
        
        // Fill all RMSNorm gammas with 1.0
        {
            std::vector<float> ones(d_model, 1.0f);
            cudaMemcpyAsync(params.rms1_gamma.data, ones.data(),
                            d_model * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(params.rms2_gamma.data, ones.data(),
                            d_model * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(params.rms_post_attn_gamma.data, ones.data(),
                            d_model * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(params.rms_post_ffn_gamma.data, ones.data(),
                            d_model * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
        }
        
        // Attention QKV projection [total_qkv_dim, d_model]
        params.attn_qkv_weight = Tensor::zeros({total_qkv_dim, d_model}, stream, "attn_qkv_weight");
        params.attn_qkv_weight.requires_grad_();
        Tensor::xavier_uniform_(params.attn_qkv_weight, seed + 2 + layer * 10, stream);
        
        if (use_bias) {
            params.attn_qkv_bias = Tensor::zeros({total_qkv_dim}, stream, "attn_qkv_bias");
            params.attn_qkv_bias.requires_grad_();
        }
        
        // Attention output projection [d_model, d_model]
        params.attn_out_weight = Tensor::zeros({d_model, d_model}, stream, "attn_out_weight");
        params.attn_out_weight.requires_grad_();
        Tensor::xavier_uniform_(params.attn_out_weight, seed + 2 + layer * 10 + 1, stream);
        
        if (use_bias) {
            params.attn_out_bias = Tensor::zeros({d_model}, stream, "attn_out_bias");
            params.attn_out_bias.requires_grad_();
        }
        
        // QK-norm learned scales (nGPT)
        params.alpha_q = Tensor::zeros({num_heads}, stream, "alpha_q");
        params.alpha_q.requires_grad_();
        // Initialize to 1.0
        {
            std::vector<float> ones(num_heads, 1.0f);
            cudaMemcpyAsync(params.alpha_q.data, ones.data(),
                            num_heads * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
        }
        
        params.alpha_k = Tensor::zeros({num_kv_heads}, stream, "alpha_k");
        params.alpha_k.requires_grad_();
        {
            std::vector<float> ones(num_kv_heads, 1.0f);
            cudaMemcpyAsync(params.alpha_k.data, ones.data(),
                            num_kv_heads * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
        }
        
        // FFN W1 [d_model, d_ff] - up-projection (passed to FeedForwardLayer constructor)
        // Issue #89 FIX: Was [d_ff, d_model] - swapped to match consumer's expectation
        params.ffn_w1 = Tensor::zeros({d_model, d_ff}, stream, "ffn_w1");
        params.ffn_w1.requires_grad_();
        Tensor::xavier_uniform_(params.ffn_w1, seed + 2 + layer * 10 + 2, stream);
        
        if (use_bias) {
            params.ffn_b1 = Tensor::zeros({d_ff}, stream, "ffn_b1");
            params.ffn_b1.requires_grad_();
        }
        
        // FFN W2 [d_ff, d_model] - down-projection (passed to FeedForwardLayer constructor)
        // Issue #89 FIX: Was [d_model, d_ff] - swapped to match consumer's expectation
        params.ffn_w2 = Tensor::zeros({d_ff, d_model}, stream, "ffn_w2");
        params.ffn_w2.requires_grad_();
        Tensor::xavier_uniform_(params.ffn_w2, seed + 2 + layer * 10 + 3, stream);
        
        if (use_bias) {
            params.ffn_b2 = Tensor::zeros({d_model}, stream, "ffn_b2");
            params.ffn_b2.requires_grad_();
        }
        
        // LayerScale (Issue #109) - learnable scalars that multiply sublayer outputs before residual
        if (use_layer_scale) {
            // layer_scale1: scales attention output before residual
            params.layer_scale1 = Tensor::zeros({1}, stream, "layer_scale1");
            params.layer_scale1.requires_grad_();
            cudaMemcpyAsync(params.layer_scale1.data, &layer_scale_init, sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            
            // layer_scale2: scales FFN output before residual
            params.layer_scale2 = Tensor::zeros({1}, stream, "layer_scale2");
            params.layer_scale2.requires_grad_();
            cudaMemcpyAsync(params.layer_scale2.data, &layer_scale_init, sizeof(float),
                            cudaMemcpyHostToDevice, stream);
        }
    }
    
    initialized_ = true;
    
    // PRE-ALLOCATE ALL GRADIENT BUFFERS
    // Required because buildParameterGroups() reads tensor.has_grad() at startup,
    // and ParameterGroup::grads() dereferences tensor->grad_data(). Without eager
    // allocation, grad pointers would be nullptr until the first backward pass.
    allocateAllGradients();
    
    // Sync to ensure all initialization is complete
    if (stream) {
        cudaStreamSynchronize(stream);
    }
}

void TrainingTensors::allocateAllGradients() {
    // Global tensors
    embedding_weights.ensure_grad();
    if (position_embedding_weights.data) position_embedding_weights.ensure_grad();
    lm_head_weights.ensure_grad();
    if (lm_head_bias.data) lm_head_bias.ensure_grad();
    if (numeric_head_weights.data) numeric_head_weights.ensure_grad();
    if (numeric_head_bias.data) numeric_head_bias.ensure_grad();
    if (final_rms_gamma.data) final_rms_gamma.ensure_grad();
    
    // Encoder layer tensors
    for (auto& layer : encoder_layers) {
        layer.rms1_gamma.ensure_grad();
        layer.rms2_gamma.ensure_grad();
        layer.rms_post_attn_gamma.ensure_grad();
        layer.rms_post_ffn_gamma.ensure_grad();
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
    
    cached_embeddings = Tensor::zeros({total_tokens, d_model}, stream, "cached_embeddings");
    
    //==================================================//
    //  ENCODER LAYER CACHES
    //==================================================//
    
    encoder_caches.resize(num_layers);
    
    for (int layer = 0; layer < num_layers; ++layer) {
        EncoderLayerCache& cache = encoder_caches[layer];
        
        cache.ln1_output = Tensor::zeros({total_tokens, d_model}, stream, "cache_ln1_output");
        cache.attn_output = Tensor::zeros({total_tokens, d_model}, stream, "cache_attn_output");
        cache.residual1_output = Tensor::zeros({total_tokens, d_model}, stream, "cache_residual1_output");
        cache.ln2_output = Tensor::zeros({total_tokens, d_model}, stream, "cache_ln2_output");
        cache.ffn_pre_gelu = Tensor::zeros({total_tokens, d_ff}, stream, "cache_ffn_pre_gelu");
        cache.ffn_output = Tensor::zeros({total_tokens, d_model}, stream, "cache_ffn_output");
        cache.layer_output = Tensor::zeros({total_tokens, d_model}, stream, "cache_layer_output");
        
        // QKV caches in BHSD format
        cache.Q = Tensor::zeros({batch_size, num_heads, seq_len, head_dim}, stream, "cache_Q");
        cache.K = Tensor::zeros({batch_size, num_kv_heads, seq_len, head_dim}, stream, "cache_K");
        cache.V = Tensor::zeros({batch_size, num_kv_heads, seq_len, head_dim}, stream, "cache_V");
        cache.attn_input = Tensor::zeros({total_tokens, d_model}, stream, "cache_attn_input");
        cache.attn_bhsd = Tensor::zeros({batch_size, num_heads, seq_len, head_dim}, stream, "cache_attn_bhsd");
        
        // Softmax LSE for FlashAttention
        cache.softmax_lse = Tensor::zeros({batch_size, num_heads, seq_len}, stream, "cache_softmax_lse");
    }
    
    //==================================================//
    //  OUTPUT LAYER CACHES
    //==================================================//
    
    cached_encoder_output = Tensor::zeros({total_tokens, d_model}, stream, "cached_encoder_output");
    cached_logits = Tensor::zeros({total_tokens, vocab_size}, stream, "cached_logits");
    cached_final_rms_input = Tensor::zeros({total_tokens, d_model}, stream, "cached_final_rms_input");
    
    //==================================================//
    //  GRADIENT TEMPORARIES
    //==================================================//
    
    grad_logits = Tensor::zeros({total_tokens, vocab_size}, stream, "grad_logits");
    grad_encoder_out = Tensor::zeros({total_tokens, d_model}, stream, "grad_encoder_out");
    
    grad_ffn_input = Tensor::zeros({total_tokens, d_model}, stream, "grad_ffn_input");
    grad_ffn_hidden = Tensor::zeros({total_tokens, d_ff}, stream, "grad_ffn_hidden");
    grad_attn_input = Tensor::zeros({total_tokens, d_model}, stream, "grad_attn_input");
    grad_attn_out_before_proj = Tensor::zeros({total_tokens, d_model}, stream, "grad_attn_out_before_proj");
    grad_attn_out_reshaped = Tensor::zeros({batch_size, num_heads, seq_len, head_dim}, stream, "grad_attn_out_reshaped");
    grad_q = Tensor::zeros({batch_size, num_heads, seq_len, head_dim}, stream, "grad_q");
    grad_k = Tensor::zeros({batch_size, num_kv_heads, seq_len, head_dim}, stream, "grad_k");
    grad_v = Tensor::zeros({batch_size, num_kv_heads, seq_len, head_dim}, stream, "grad_v");
    
    const int total_qkv_dim = d_model + 2 * kv_dim;
    grad_qkv_concat = Tensor::zeros({total_tokens, total_qkv_dim}, stream, "grad_qkv_concat");
    grad_qkv_input = Tensor::zeros({total_tokens, d_model}, stream, "grad_qkv_input");
    grad_attn_bsm_scratch = Tensor::zeros({batch_size, seq_len, d_model}, stream, "grad_attn_bsm_scratch");
    
    // Issue #43 centering scratch (max of d_model and d_ff)
    const int scratch_dim = (d_ff > d_model) ? d_ff : d_model;
    centered_activation_scratch = Tensor::zeros({total_tokens, scratch_dim}, stream, "centered_activation_scratch");
}


void TrainingTensors::zeroGrad(cudaStream_t stream) {
    if (!initialized_) return;
    
    // ── Verify tied-weight grad aliasing invariant (Rule 20: detect broken aliasing) ──
    if (tie_embeddings) {
        if (embedding_weights.grad_data() != lm_head_weights.grad_data()) {
            throw std::runtime_error("TrainingTensors::zeroGrad: CRITICAL - tie_embeddings=true but "
                "embedding_weights.grad (" + std::to_string(reinterpret_cast<uintptr_t>(embedding_weights.grad_data()))
                + ") != lm_head_weights.grad (" + std::to_string(reinterpret_cast<uintptr_t>(lm_head_weights.grad_data()))
                + "). Grad aliasing is broken!");
        }
    }
    
    // Zero embedding grads
    embedding_weights.zero_grad(stream);
    if (position_embedding_weights.data) position_embedding_weights.zero_grad(stream);
    
    // Zero LM head grads
    if (!tie_embeddings) {
        lm_head_weights.zero_grad(stream);
    }
    // NOTE: When tie_embeddings=true, lm_head_weights.grad IS embedding_weights.grad
    // (verified above). Zeroing embedding_weights.grad already zeroed both.
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
        layer.rms_post_attn_gamma.zero_grad(stream);
        layer.rms_post_ffn_gamma.zero_grad(stream);
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
        if (layer.layer_scale1.data) layer.layer_scale1.zero_grad(stream);
        if (layer.layer_scale2.data) layer.layer_scale2.zero_grad(stream);
    }
}


}  // namespace GRIM

#endif  // USE_CUDA
