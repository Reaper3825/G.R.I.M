//======================================================//
//  DIAGNOSTIC MINIMAL TRAINING LOOP - MANUAL MODE
//  
//  PURPOSE: Human-in-the-loop token prediction training
//  
//  What this does:
//  - Load model and tokenizer
//  - Load training sequences
//  - For each token position:
//    - Show TARGET token (what model should predict)
//    - Show PREDICTED token (what model actually predicts)
//    - Wait for human feedback (y/n/q)
//    - Train based on feedback
//  
//  Controls:
//  - 'y' or Enter = prediction correct (low loss)
//  - 'n' = prediction wrong (high loss from actual target)
//  - 'q' = quit
//  - 's' = skip to next sequence
//  - 'a' = auto-mode (run N tokens automatically)
//======================================================//

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdint>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// HyperParameters_GPU.hpp is the single entry point; it defines
// GRIM_HP_GPU_DEFINED_TRAINING_STRUCTS and transitively includes
// control/ai_config_paths.hpp in the correct order.
#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"

// Core model
#include "../GRIM/grim_language_model_cuda.hpp"
#include "../Layers/Encoding/Encoding_GPU.hpp"
#include "../Shared/Batching/BatchPayload.hpp"
#include "Autograd/AutogradTraining.hpp"
#include "../Shared/Optimizers/AdamW/AdamW_Kernal_GPU.hpp"

// Tokenizer  
#include "../Shared/UnigramByte/UniByte.hpp"

// Data loading
#include "training_data_loader.hpp"

//======================================================//
//  CUDA Error Check Macro (only if not already defined)
//======================================================//
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)
#endif

//======================================================//
//  Compute RMS of a gradient buffer (sync to CPU)
//======================================================//
float computeGradNormSync(float* d_buffer, size_t size, cudaStream_t stream) {
    if (!d_buffer || size == 0) return 0.0f;
    
    // Copy to host and compute RMS
    std::vector<float> h_buffer(size);
    CUDA_CHECK(cudaMemcpyAsync(h_buffer.data(), d_buffer, size * sizeof(float), 
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    double sum_sq = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum_sq += static_cast<double>(h_buffer[i]) * h_buffer[i];
    }
    return static_cast<float>(std::sqrt(sum_sq / size));
}

//======================================================//
//  Detailed Gradient Component Tracer
//  
//  PURPOSE: Show WHERE gradients collapse during backward
//======================================================//
void traceGradientComponents(GRIM::LanguageModel& model, int batch, cudaStream_t stream, int trace_interval) {
    auto& ts = model.getTrainingState();
    const auto& config = model.getConfig();
    
    // Only trace every N batches to avoid spam
    if (trace_interval <= 0 || (batch % trace_interval) != 0) return;
    
    std::cout << "\n[GradientTrace] batch=" << batch << " ──────────────────────────" << std::endl;
    
    // Get GPU encoder for accessing layer-internal gradient buffers
    auto* gpu_encoder = &model.getGpuEncoder();
    
    // 1. LM Head gradients (output layer - should be largest)
    auto* lm_head = model.getLmHeadLayer();
    if (!lm_head) throw std::runtime_error("[traceGradientComponents] LMHeadLayer is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    if (lm_head->weights().grad_data()) {
        size_t lm_size = static_cast<size_t>(config.vocab_size) * config.d_model;
        float lm_head_rms = computeGradNormSync(lm_head->weights().grad_data(), lm_size, stream);
        std::cout << "  LM_HEAD: " << std::scientific << std::setprecision(4) << lm_head_rms 
                  << " (should be largest - closest to loss)" << std::endl;
    }
    
    // 2. Per-layer gradients (show gradient flow through encoder)
    // Uses Tensor& accessors + numel() — no manual size computation.
    for (int layer = config.num_layers - 1; layer >= 0; --layer) {
        std::cout << "  Layer " << layer << ":" << std::endl;
        
        auto* enc = gpu_encoder ? gpu_encoder->getLayer(layer) : nullptr;
        if (!enc) {
            std::cout << "    SKIP: encoder layer is null" << std::endl;
            continue;
        }
        
        // Helper: compute and print grad RMS for a named Tensor
        auto printGradRms = [&](const char* label, GRIM::Tensor& tensor) {
            if (tensor.has_grad()) {
                float rms = computeGradNormSync(tensor.grad_data(), tensor.numel(), stream);
                std::cout << "  " << label << ": " << std::scientific << std::setprecision(4) << rms;
            } else {
                std::cout << "  " << label << ": NULL";
            }
        };
        
        // Attention gradients
        printGradRms("QKV", enc->attnWqkv());
        printGradRms("W_o", enc->attnWo());
        
        // FFN gradients (SwiGLU)
        printGradRms("FFN_W_gate", enc->ffnWGate());
        printGradRms("FFN_W1", enc->ffnW1());
        printGradRms("FFN_W2", enc->ffnW2());
        std::cout << std::endl;
        
        // RMSNorm gradients (should be smallest)
        std::cout << "    ";
        printGradRms("RMS1_gamma", enc->rms1Gamma());
        printGradRms("RMS2_gamma", enc->rms2Gamma());
        std::cout << std::endl;
    }
    
    // 3. Embedding gradients (input layer - should be attenuated from output)
    auto* emb_layer = model.getEmbeddingLayer();
    if (!emb_layer) throw std::runtime_error("[traceGradientComponents] EmbeddingLayer is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    if (emb_layer->tokenWeights().grad_data()) {
        const size_t embedding_grad_size = config.vocab_size * config.d_model;
        float emb_rms = computeGradNormSync(emb_layer->tokenWeights().grad_data(), embedding_grad_size, stream);
        std::cout << "  EMBEDDING: " << emb_rms 
                  << " (most attenuated - farthest from loss)" << std::endl;
    }
    
    std::cout << "────────────────────────────────────────────────────────" << std::endl;
}

//======================================================//
//  Get Top-K Predictions from Logits
//======================================================//
struct TokenPrediction {
    int token_id;
    float probability;
    std::string decoded;
};

std::vector<TokenPrediction> getTopKPredictions(
    float* d_logits,           // GPU logits [vocab_size]
    int vocab_size,
    int k,
    GRIM::Tokenizer::UniByte& tokenizer,
    cudaStream_t stream
) {
    // Copy logits to host
    std::vector<float> h_logits(vocab_size);
    CUDA_CHECK(cudaMemcpyAsync(h_logits.data(), d_logits, vocab_size * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Softmax for probabilities
    float max_logit = *std::max_element(h_logits.begin(), h_logits.end());
    std::vector<float> probs(vocab_size);
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(h_logits[i] - max_logit);
        sum_exp += probs[i];
    }
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] /= sum_exp;
    }
    
    // Get top-K indices
    std::vector<std::pair<float, int>> prob_idx(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        prob_idx[i] = {probs[i], i};
    }
    std::partial_sort(prob_idx.begin(), prob_idx.begin() + k, prob_idx.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Build results
    std::vector<TokenPrediction> results;
    results.reserve(k);
    for (int i = 0; i < k; ++i) {
        TokenPrediction pred;
        pred.token_id = prob_idx[i].second;
        pred.probability = prob_idx[i].first;
        
        // Decode token
        std::vector<int> ids = {pred.token_id};
        pred.decoded = tokenizer.decode(ids);
        
        // Escape special characters for display
        std::string escaped;
        for (char c : pred.decoded) {
            if (c == '\n') escaped += "\\n";
            else if (c == '\r') escaped += "\\r";
            else if (c == '\t') escaped += "\\t";
            else if (c < 32 || c > 126) escaped += "[0x" + std::to_string((unsigned char)c) + "]";
            else escaped += c;
        }
        pred.decoded = escaped;
        
        results.push_back(pred);
    }
    
    return results;
}

//======================================================//
//  Display Token Comparison
//======================================================//
void displayTokenComparison(
    int position,
    const std::vector<int>& context_ids,  // Full context token IDs
    int target_token_id,
    const std::vector<TokenPrediction>& predictions,
    GRIM::Tokenizer::UniByte& tokenizer
) {
    // Escape for display
    auto escape = [](const std::string& s) {
        std::string escaped;
        for (char c : s) {
            if (c == '\n') escaped += "\\n";
            else if (c == '\r') escaped += "\\r";
            else if (c == '\t') escaped += "\\t";
            else if (c < 32 || c > 126) escaped += "[0x" + std::to_string((unsigned char)c) + "]";
            else escaped += c;
        }
        return escaped;
    };
    
    // Decode full context (show last ~50 chars)
    std::string full_context = tokenizer.decode(std::vector<int>(context_ids.begin(), context_ids.end()));
    std::string context_preview = full_context;
    if (context_preview.length() > 50) {
        context_preview = "..." + context_preview.substr(context_preview.length() - 47);
    }
    
    // Decode target
    std::vector<int> tgt_ids = {target_token_id};
    std::string target_str = tokenizer.decode(tgt_ids);
    
    // Get last token for display
    int last_token_id = context_ids.empty() ? -1 : context_ids.back();
    std::vector<int> last_ids = {last_token_id};
    std::string last_token_str = last_token_id >= 0 ? tokenizer.decode(last_ids) : "?";
    
    std::cout << "\n╔══════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Position " << std::setw(5) << position 
              << "  (context_len=" << std::setw(4) << context_ids.size() << ")                              ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  CONTEXT: \"" << std::setw(55) << std::left << escape(context_preview) << "\"  ║" << std::endl;
    std::cout << "║  LAST TOKEN: \"" << std::setw(15) << std::left << escape(last_token_str) 
              << "\" (id=" << std::setw(6) << last_token_id << ")                            ║" << std::endl;
    std::cout << "║  → PREDICT: \"" << std::setw(15) << std::left << escape(target_str) 
              << "\" (id=" << std::setw(6) << target_token_id << ")                            ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  MODEL PREDICTIONS (Top-5):                                              ║" << std::endl;
    
    bool target_in_top5 = false;
    int target_rank = -1;
    for (size_t i = 0; i < predictions.size(); ++i) {
        const auto& p = predictions[i];
        std::string marker = (p.token_id == target_token_id) ? " ◀ TARGET" : "";
        if (p.token_id == target_token_id) {
            target_in_top5 = true;
            target_rank = static_cast<int>(i) + 1;
        }
        std::cout << "║    #" << (i+1) << ": \"" << std::setw(12) << std::left << p.decoded 
                  << "\" id=" << std::setw(6) << std::left << p.token_id
                  << " prob=" << std::fixed << std::setprecision(2) << (p.probability * 100.0f) << "%" 
                  << std::setw(12) << marker << "  ║" << std::endl;
    }
    
    std::cout << "╠══════════════════════════════════════════════════════════════════════════╣" << std::endl;
    if (target_in_top5) {
        std::cout << "║  ✓ Target is rank #" << target_rank << " in predictions                                     ║" << std::endl;
    } else {
        std::cout << "║  ✗ Target NOT in top-5 predictions                                       ║" << std::endl;
    }
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝" << std::endl;
}

//======================================================//
//  Get User Feedback
//======================================================//
enum class UserFeedback {
    CORRECT,        // y - prediction correct
    INCORRECT,      // n - prediction wrong  
    SKIP_SEQUENCE,  // s - skip to next sequence
    AUTO_MODE,      // a - enable auto mode for N tokens
    QUIT            // q - quit
};

UserFeedback getUserFeedback(int& auto_remaining) {
    if (auto_remaining > 0) {
        auto_remaining--;
        return UserFeedback::INCORRECT;  // In auto mode, always train on actual target
    }
    
    std::cout << "\n  [y/Enter=correct, n=wrong(train), s=skip, a=auto(N), q=quit]: ";
    std::string input;
    std::getline(std::cin, input);
    
    if (input.empty() || input[0] == 'y' || input[0] == 'Y') {
        return UserFeedback::CORRECT;
    } else if (input[0] == 'n' || input[0] == 'N') {
        return UserFeedback::INCORRECT;
    } else if (input[0] == 's' || input[0] == 'S') {
        return UserFeedback::SKIP_SEQUENCE;
    } else if (input[0] == 'a' || input[0] == 'A') {
        // Parse number after 'a'
        int n = 10;  // default
        if (input.length() > 1) {
            try {
                n = std::stoi(input.substr(1));
            } catch (...) {
                n = 10;
            }
        }
        auto_remaining = n;
        std::cout << "  → Auto-mode enabled for " << n << " tokens" << std::endl;
        return UserFeedback::INCORRECT;  // Start auto mode with training
    } else if (input[0] == 'q' || input[0] == 'Q') {
        return UserFeedback::QUIT;
    }
    
    return UserFeedback::INCORRECT;  // Default to training
}

//======================================================//
//  Main Entry Point
//======================================================//
int main(int argc, char** argv) {
    std::cout << "╔════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     MANUAL TOKEN PREDICTION TRAINING                   ║" << std::endl;
    std::cout << "║     Human-in-the-loop feedback mode                    ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nControls:" << std::endl;
    std::cout << "  y/Enter = Prediction looks correct (skip training)" << std::endl;
    std::cout << "  n       = Prediction wrong (train on target)" << std::endl;
    std::cout << "  s       = Skip to next sequence" << std::endl;
    std::cout << "  a[N]    = Auto-mode for N tokens (e.g., a50)" << std::endl;
    std::cout << "  q       = Quit" << std::endl;
    
    //--------------------------------------------------
    // 1. Load Config (same pattern as Phase1_Startup)
    //--------------------------------------------------
    std::cout << "\n[1] Loading config..." << std::endl;
    
    auto snapshot = GRIM::Config::loadAiConfigSnapshot();
    if (!snapshot) {
        std::cerr << "  ✗ Failed to load ai_config.json" << std::endl;
        return 1;
    }
    
    std::cout << "  Config file: " << snapshot->config_path << std::endl;
    
    // Check if paths exists
    if (!snapshot->document.contains("paths")) {
        std::cerr << "  ✗ No 'paths' object in ai_config.json" << std::endl;
        return 1;
    }
    if (!snapshot->document["paths"].contains("grim_text")) {
        std::cerr << "  ✗ No 'paths.grim_text' object in ai_config.json" << std::endl;
        return 1;
    }
    
    const auto& grim_text = snapshot->document["paths"]["grim_text"];
    std::string vocab_path = grim_text.value("vocab", "");
    std::string data_path = grim_text.value("training_data", "");
    
    std::cout << "  vocab_path: " << (vocab_path.empty() ? "(empty)" : vocab_path) << std::endl;
    std::cout << "  data_path: " << (data_path.empty() ? "(empty)" : data_path) << std::endl;
    
    if (vocab_path.empty() || data_path.empty()) {
        std::cerr << "  ✗ Missing vocab or training_data paths in ai_config.json" << std::endl;
        return 1;
    }
    
    // Get model architecture from config (path: training.config).
    // Delegates to the single source of truth in HyperParameters_GPU.hpp —
    // do NOT re-parse training.config keys here.
    std::cout << "  Getting model architecture..." << std::endl;

    if (!snapshot->document.contains("training") || !snapshot->document["training"].contains("config")) {
        std::cerr << "  ✗ No 'training.config' object in ai_config.json" << std::endl;
        return 1;
    }

    GRIM::HyperParameters::ModelArchitecture arch;
    GRIM::HyperParameters::loadModelArchitecture(*snapshot, arch);

    int d_model        = arch.d_model;
    int num_layers     = arch.num_layers;
    int num_heads      = arch.num_heads;
    int num_kv_heads   = arch.num_kv_heads;
    int d_ff           = arch.d_ff;
    bool tie_embeddings = arch.tie_embeddings;

    std::cout << "  Model config: d_model=" << d_model << " layers=" << num_layers 
              << " heads=" << num_heads << " kv_heads=" << num_kv_heads << std::endl;
    
    // Hardcode simple hyperparameters for manual mode
    const float learning_rate = 0.0001f;  // 1e-4
    const int max_seq_len = 2048;
    
    std::cout << "  Config: " << snapshot->config_path << std::endl;
    std::cout << "  lr=" << learning_rate << " max_seq_len=" << max_seq_len << std::endl;
    
    //--------------------------------------------------
    // 2. Load Tokenizer
    //--------------------------------------------------
    std::cout << "\n[2] Loading tokenizer..." << std::endl;
    
    GRIM::Tokenizer::UniByte tokenizer;
    
    if (!tokenizer.load(vocab_path)) {
        std::cerr << "  ✗ Failed to load vocab from: " << vocab_path << std::endl;
        return 1;
    }
    
    uint32_t vocab_size = tokenizer.vocabSize();
    std::cout << "  ✓ Loaded vocab: " << vocab_size << " tokens" << std::endl;
    
    //--------------------------------------------------
    // 3. Load Training Data
    //--------------------------------------------------
    std::cout << "\n[3] Loading training data..." << std::endl;
    
    GRMTDataLoader data_loader;
    
    if (!data_loader.load(data_path)) {
        std::cerr << "  ✗ Failed to load data from: " << data_path << std::endl;
        return 1;
    }
    
    const auto& sequences = data_loader.getSequences();
    std::cout << "  ✓ Loaded " << sequences.size() << " sequences" << std::endl;
    if (sequences.empty()) {
        std::cerr << "  ✗ No sequences loaded - cannot run diagnostic" << std::endl;
        return 1;
    }
    std::mt19937 rng(1);
    std::uniform_int_distribution<size_t> seq_dist(0, sequences.size() - 1);
    
    //--------------------------------------------------
    // 4. Initialize Model
    //--------------------------------------------------
    std::cout << "\n[4] Initializing model..." << std::endl;
    
    GRIM::HyperParameters::LanguageModelConfig lm_config;
    lm_config.vocab_size = vocab_size;
    lm_config.d_model = d_model;
    lm_config.num_layers = num_layers;
    lm_config.num_heads = num_heads;
    lm_config.num_kv_heads = num_kv_heads;
    lm_config.d_ff = d_ff;
    lm_config.tie_embeddings = tie_embeddings;
    lm_config.max_seq_len = max_seq_len;
    lm_config.use_bias = false;
    lm_config.execution_mode = GRIM::HyperParameters::ModelExecutionMode::TRAINING;
    lm_config.computeDerivedValues();  // Compute head_dim = d_model / num_heads
    
    std::cout << "  d_model=" << lm_config.d_model 
              << " layers=" << lm_config.num_layers
              << " heads=" << lm_config.num_heads 
              << " kv_heads=" << lm_config.num_kv_heads
              << " d_ff=" << lm_config.d_ff << std::endl;
    std::cout << "  tie_embeddings=" << (lm_config.tie_embeddings ? "true" : "false") << std::endl;
    
    GRIM::LanguageModel model(lm_config);
    
    // STEP 1: Initialize StreamController first
    {
        GRIM::StreamControllerConfig stream_config;
        stream_config.verbose = true;
        
        if (!model.getTrainingState().stream_ctrl.initialize(stream_config)) {
            std::cerr << "  ✗ Failed to initialize StreamController" << std::endl;
            return 1;
        }
        std::cout << "  ✓ StreamController initialized" << std::endl;
    }
    
    // STEP 2: Initialize cuBLAS handle
    model.initCuBLASHandle();
    std::cout << "  ✓ cuBLAS handle initialized" << std::endl;
    
    // STEP 3: Initialize RoPE ONLY (before initGPU so encoder gets the RoPE spec)
    model.initPBM();
    std::cout << "  ✓ PBM initialized" << std::endl;
    
    // PBM DIAGNOSTIC: Verify PBM is initialized
    {
        auto& ts = model.getTrainingState();
        std::cout << "\n  [PBM] State Check:" << std::endl;
        std::cout << "    pbm_initialized: " << (ts.pbm_initialized ? "YES ✓" : "NO ✗") << std::endl;
        std::cout << "    pbm_spec.valid: " << (ts.pbm_spec.valid ? "YES ✓" : "NO ✗") << std::endl;
        std::cout << "    rope_inv_freq: " << (ts.pbm_spec.rope_inv_freq ? "VALID ✓" : "NULL ✗") << std::endl;
        std::cout << "    alibi_slopes: " << (ts.pbm_spec.alibi_slopes ? "VALID ✓" : "NULL ✗") << std::endl;
        std::cout << "    rotary_dim: " << ts.pbm_spec.rotary_dim << std::endl;
        std::cout << "    num_heads: " << ts.pbm_spec.num_heads << std::endl;
        
        if (!ts.pbm_initialized || !ts.pbm_spec.valid) {
            std::cerr << "\n  ✗ FATAL ERROR: PBM not initialized!" << std::endl;
            std::cerr << "  pbm_initialized: " << ts.pbm_initialized << std::endl;
            std::cerr << "  pbm_spec.valid: " << ts.pbm_spec.valid << std::endl;
            std::cerr << "  rope_inv_freq: " << (void*)ts.pbm_spec.rope_inv_freq << std::endl;
            throw std::runtime_error("PBM not initialized! Attention REQUIRES positional encoding. Fix initPBM().");
        }
    }
    
    // STEP 4: Initialize GPU encoder (uses PBM from TrainingState)
    model.initGPU();
    std::cout << "  ✓ GPU encoder initialized" << std::endl;
    
    // STEP 5: Initialize TrainingState (grad buffers) - needs GPU embedder from initGPU
    model.initTrainingState();
    std::cout << "  ✓ TrainingState initialized" << std::endl;
    
    // Count parameters for gradient size calculations (for info only)
    size_t emb_size = static_cast<size_t>(vocab_size) * lm_config.d_model;
    const int head_dim = lm_config.head_dim;  // Use pre-computed value from config
    const int kv_dim = lm_config.num_kv_heads * head_dim;
    const int total_qkv_dim = lm_config.d_model + 2 * kv_dim;  // GQA
    size_t qkv_size = static_cast<size_t>(total_qkv_dim) * lm_config.d_model;
    size_t w1_size = static_cast<size_t>(lm_config.d_model) * lm_config.d_ff;
    
    std::cout << "  ✓ Model initialized" << std::endl;
    std::cout << "  emb_size=" << emb_size << " qkv_size=" << qkv_size << " w1_size=" << w1_size << std::endl;

    // Suppress unused variable warnings
    (void)emb_size;
    (void)qkv_size;
    (void)w1_size;
    
    //--------------------------------------------------
    // 5. Get Training State and Optimizer
    //--------------------------------------------------
    auto& ts = model.getTrainingState();
    
    // Get stream from model's stream controller
    cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
    
    std::cout << "\n[5] Training state initialized" << std::endl;
    
    //--------------------------------------------------
    // 6. MANUAL TOKEN PREDICTION LOOP
    //--------------------------------------------------
    std::cout << "\n[6] Starting manual token prediction mode..." << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════" << std::endl;
    
    int total_tokens_seen = 0;
    int correct_predictions = 0;
    int trained_tokens = 0;
    int skipped_tokens = 0;
    int seq_idx = 0;
    int auto_remaining = 0;  // For auto-mode
    bool quit_requested = false;
    
    GRIM::OptimizerState manual_optimizer_state;
    auto training_start = std::chrono::steady_clock::now();
    
    // Process sequences one at a time
    while (!quit_requested && seq_idx < static_cast<int>(sequences.size())) {
        // Select sequence
        const auto& seq = sequences[seq_idx];
        int seq_len = static_cast<int>(seq.token_ids.size());
        
        if (seq_len < 2) {
            std::cout << "\n[Sequence " << seq_idx << "] Too short (len=" << seq_len << "), skipping..." << std::endl;
            seq_idx++;
            continue;
        }
        
        if (seq_len > max_seq_len) seq_len = max_seq_len;
        
        std::cout << "\n╔═══════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  SEQUENCE " << std::setw(5) << seq_idx << " / " << sequences.size() 
                  << "  (length: " << seq_len << " tokens)                    ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;
        
        // Show first few tokens as context
        std::cout << "Context preview: ";
        int preview_len = std::min(10, seq_len);
        std::vector<int> preview_ids(seq.token_ids.begin(), seq.token_ids.begin() + preview_len);
        std::string preview_text = tokenizer.decode(preview_ids);
        // Truncate and escape
        if (preview_text.length() > 50) preview_text = preview_text.substr(0, 50) + "...";
        for (char& c : preview_text) {
            if (c == '\n' || c == '\r' || c == '\t') c = ' ';
        }
        std::cout << "\"" << preview_text << "\"" << std::endl;
        
        bool skip_sequence = false;
        
        // Process each token position in the sequence
        // BUG FIX: Start at pos=1 (context_len=2) because prepareLossBatchInputs masks the 
        // LAST target position for sliding window boundary. With context_len=1, the single
        // target token gets masked, resulting in valid_tokens=0 and a FATAL error.
        // Position 0 has no meaningful "next token prediction" context anyway.
        for (int pos = 1; pos < seq_len - 1 && !quit_requested && !skip_sequence; ++pos) {
            total_tokens_seen++;
            
            // Build context: all tokens up to and including current position
            int context_len = pos + 1;
            std::vector<int> context_ids(seq.token_ids.begin(), seq.token_ids.begin() + context_len);
            // GRMT v5: Use precomputed targets instead of computing at runtime
            int target_id = seq.targets[pos];
            
            // Skip if this target is masked (-1) - e.g., BOS/EOS boundary
            if (target_id < 0) {
                skipped_tokens++;
                continue;
            }
            
            // Get numeric side-channels for context
            std::vector<float> numeric_values(seq.token_numeric_values.begin(),
                                              seq.token_numeric_values.begin() + context_len);
            std::vector<uint8_t> atom_mask(seq.token_atom_mask.begin(),
                                           seq.token_atom_mask.begin() + context_len);
            
            //--------------------------------------------------
            // 6a. FORWARD PASS (get logits for prediction)
            //--------------------------------------------------
            // Use getNextTokenLogits to get logits for the next token position
            GRIM::Vector logits_vec = model.getNextTokenLogits(context_ids, numeric_values, atom_mask);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            // Copy logits to a GPU buffer for getTopKPredictions
            // logits_vec.data contains logits for vocab_size
            float* d_logits = nullptr;
            CUDA_CHECK(cudaMalloc(&d_logits, vocab_size * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_logits, logits_vec.data.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice));
            
            //--------------------------------------------------
            // 6b. GET TOP-K PREDICTIONS
            //--------------------------------------------------
            auto predictions = getTopKPredictions(d_logits, vocab_size, 5, tokenizer, stream);
            CUDA_CHECK(cudaFree(d_logits));
            
            //--------------------------------------------------
            // 6c. DISPLAY AND GET FEEDBACK
            //--------------------------------------------------
            displayTokenComparison(pos, context_ids, target_id, predictions, tokenizer);
            
            // Check if model's top prediction matches target
            bool is_correct = (!predictions.empty() && predictions[0].token_id == target_id);
            
            UserFeedback feedback = getUserFeedback(auto_remaining);
            
            switch (feedback) {
                case UserFeedback::CORRECT:
                    correct_predictions++;
                    skipped_tokens++;
                    std::cout << "  → Marked as correct, skipping training step" << std::endl;
                    break;
                    
                case UserFeedback::INCORRECT: {
                    trained_tokens++;
                    
                    //--------------------------------------------------
                    // 6d. COMPUTE LOSS AND BACKWARD
                    //--------------------------------------------------
                    // computeLossBatch expects: input[i] predicts target[i]
                    // So input = context (all tokens), target = context shifted left by 1 + current target
                    // For training on THIS specific position, we use the full context
                    // Input: tokens 0..pos, Target: tokens 1..pos+1 (where pos+1 is target_id)
                    
                    std::vector<int> train_input_ids = context_ids;  // All context tokens
                    std::vector<int> train_target_ids(context_ids.begin() + 1, context_ids.end());
                    train_target_ids.push_back(target_id);  // Add the target we're training on
                    
                    // Numeric values for input (same length as input)
                    std::vector<float> train_numeric_values = numeric_values;
                    std::vector<uint8_t> train_atom_mask = atom_mask;
                    
                    // Construct BatchPayload for single-sequence batch
                    GRIM::Batching::BatchPayload payload;
                    payload.batch_size = 1;
                    payload.max_seq_len = static_cast<int>(train_input_ids.size());
                    payload.total_tokens = payload.max_seq_len;
                    payload.actual_tokens = payload.max_seq_len;
                    payload.padding_tokens = 0;
                    payload.packing_efficiency = 1.0f;
                    payload.seq_lengths = { payload.max_seq_len };
                    
                    // Count valid targets (non-masked, non-pad)
                    int valid_count = 0;
                    for (int tid : train_target_ids) {
                        if (tid >= 0) valid_count++;
                    }
                    payload.valid_tokens = valid_count;
                    payload.valid_target_counts = { valid_count };
                    
                    // Copy data (no padding needed for single-sequence batch)
                    payload.input_ids = train_input_ids;
                    payload.target_ids = train_target_ids;
                    payload.numeric_values = train_numeric_values;
                    payload.atom_mask = train_atom_mask;
                    payload.text_features.resize(payload.total_tokens * GRIM::Batching::BatchPayload::kTextFeatureDim, 0);
                    payload.token_to_slot_map.assign(payload.total_tokens, -1);
                    payload.fits_in_cache = true;  // Assume fits for diagnostic purposes
                    
                    // Unified forward+loss+backward via autograd
                    auto loss_result = GRIM::Autograd::autogradTrainingStep(
                        model, model.getTrainingState(), payload,
                        /*accumulate=*/false, /*grad_scale=*/1.0f, manual_optimizer_state.step);
                    if (!loss_result.success) {
                        std::cout << "  → autogradTrainingStep FAILED: " << loss_result.error_message << std::endl;
                        break;
                    }
                    float loss = loss_result.loss_value;
                    
                    std::cout << "  → Training on target (loss=" << std::fixed << std::setprecision(4) 
                              << loss << ", context_len=" << context_len << ")" << std::endl;
                    
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    
                    // Update weights
                    GRIM::launchAdamWStep(model.parameterGroups(),
                                          learning_rate,
                                          GRIM::HyperParameters::ADAMW_WEIGHT_DECAY,
                                          manual_optimizer_state.step,
                                          stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    
                    manual_optimizer_state.step++;
                    break;
                }
                    
                case UserFeedback::SKIP_SEQUENCE:
                    std::cout << "  → Skipping to next sequence" << std::endl;
                    skip_sequence = true;
                    break;
                    
                case UserFeedback::QUIT:
                    std::cout << "  → Quit requested" << std::endl;
                    quit_requested = true;
                    break;
                    
                default:
                    break;
            }
        }
        
        seq_idx++;
    }
    
    auto training_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(training_end - training_start).count();
    
    //--------------------------------------------------
    // 7. FINAL SUMMARY
    //--------------------------------------------------
    std::cout << "\n═══════════════════════════════════════════════════════════" << std::endl;
    std::cout << "[7] Manual Training Summary" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════" << std::endl;
    std::cout << "  Sequences processed: " << seq_idx << " / " << sequences.size() << std::endl;
    std::cout << "  Total tokens seen:   " << total_tokens_seen << std::endl;
    std::cout << "  Correct predictions: " << correct_predictions 
              << " (" << std::fixed << std::setprecision(1) 
              << (total_tokens_seen > 0 ? 100.0f * correct_predictions / total_tokens_seen : 0.0f) 
              << "%)" << std::endl;
    std::cout << "  Trained tokens:      " << trained_tokens << std::endl;
    std::cout << "  Skipped tokens:      " << skipped_tokens << std::endl;
    std::cout << "  Optimizer steps:     " << manual_optimizer_state.step << std::endl;
    std::cout << "  Duration:            " << std::setprecision(1) << elapsed << " seconds" << std::endl;
    
    std::cout << "\n✓ Manual training session complete" << std::endl;
    
    return 0;
}
