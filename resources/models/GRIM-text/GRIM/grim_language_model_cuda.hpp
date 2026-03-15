//======================================================//
//  grim_language_model_cuda.hpp
//  CUDA-safe declarations for LanguageModel
//  NO IMPLEMENTATIONS - declarations only
//  NO .cpp includes - CUDA compilation safe
//          ONLY SOURCE OF TRUTH
//======================================================//

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <utility>
#include <random>
#include <cmath>
#include <cstdint>
#include <cstddef>

// HyperParameters - Single source of truth for model configuration
#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"

// TensorContract - Autograd system (includes ParamGroupType, ParameterGroup, Tensor)
#include "../Shared/TensorContract/TensorContract_GPU.hpp"

// BatchPayload - Single source of truth for per-batch metadata
#include "../Shared/Batching/BatchPayload.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp"
#include "../Layers/Embedding/Embedding_GPU.hpp"
#include "../Layers/LMHead/lm_head_GPU.hpp"
#include "../Layers/NumericHead/numeric_head_GPU.hpp"
#include "../Shared/GPUBuffer/GPUBuffer.hpp"
#include "../Shared/PBM/PositionalBiasMethod.hpp"
#include "../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../Shared/Loss/LossContext/LossContext.hpp"
#endif

namespace GRIM {

//======================================================//
//  Forward Declarations
//======================================================//

// GPUGrimEncoder is defined later in this file but used by LanguageModel class
class GPUGrimEncoder;

//======================================================//
//  CUDA Error Handling Macros
//======================================================//

#ifdef USE_CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                   std::to_string(__LINE__) + " - " + \
                                   cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuBLAS error at ") + __FILE__ + ":" + \
                                   std::to_string(__LINE__)); \
        } \
    } while(0)
#endif

//======================================================//
//  Core Data Structures - Minimal Declarations
//======================================================//

// Vector type - minimal definition
class Vector {
public:
    std::vector<float> data;
    
    Vector() = default;
    Vector(size_t size, float val = 0.0f);
    
    size_t size() const;
    float& operator[](size_t idx);
    const float& operator[](size_t idx) const;
    Vector& operator+=(const Vector& other);
    Vector& operator*=(float scalar);
    Vector operator+(const Vector& other) const;
    Vector operator*(float scalar) const;
};

// Matrix type - minimal definition
class Matrix {
public:
    std::vector<Vector> rows;
    int num_rows;
    int num_cols;
    
    Matrix() = default;
    Matrix(int rows, int cols, float init_val = 0.0f, bool random = false);
    
    Vector& operator[](size_t idx);
    const Vector& operator[](size_t idx) const;
};

// Context state for embeddings
struct ContextState {
    std::string domain;
    float sentiment;
    int depth;
    std::vector<std::string> active_tags;
    
    ContextState();
};

//======================================================//
//  Configuration Structures
//  NOTE: All numeric defaults are 0 - caller MUST populate
//  from HyperParameters::loadModelArchitecture() or explicitly
//  DO NOT add hardcoded defaults here - HyperParameters_GPU.hpp
//  is the ONLY source of truth for model architecture values
//======================================================//

// Model execution mode - determines memory allocation strategy
enum class ModelExecutionMode {
    TRAINING,    // Full training state with gradient buffers (~1GB+)
    INFERENCE    // Lightweight inference state with only forward caches (~385MB)
};

struct EncoderConfig {
    // Architecture - MUST be populated from HyperParameters
    int d_model = 0;           // Use HyperParameters::DEFAULT_D_MODEL
    int num_heads = 0;         // Use HyperParameters::DEFAULT_NUM_HEADS
    int num_kv_heads = 0;      // Use HyperParameters::DEFAULT_NUM_KV_HEADS (GQA - Grouped Query Attention)
    int head_dim = 0;          // = d_model / num_heads (set from LanguageModelConfig.head_dim)
    int d_ff = 0;              // Use HyperParameters::DEFAULT_D_FF
    int num_layers = 0;        // Use HyperParameters::DEFAULT_NUM_LAYERS
    int max_seq_len = 0;       // Use HyperParameters::DEFAULT_MAX_SEQ_LEN
    float dropout_rate = 0.0f; // Use HyperParameters::DEFAULT_DROPOUT_RATE
    float residual_dropout_rate = 0.0f; // Use HyperParameters::DEFAULT_RESIDUAL_DROPOUT_RATE
    float attention_dropout = 0.0f; // Use HyperParameters::DEFAULT_ATTENTION_DROPOUT
    
    // Cache limits
    int max_cached_batch = 0;
    int max_cached_seq_len = 0;
    
    // Fixed config values (not architecture-dependent)
    // Issue #104 FIX: Changed from 1e-3 to 1e-5. The old value was too large for Layer 0 embeddings:
    // - Layer 0 input: per_row_rms ≈ 0.006, so mean(x²) ≈ 0.00004
    // - With eps=1e-3: epsilon dominated denominator (25x larger than mean(x²))
    // - Result: output_rms = 0.19 instead of expected 1.0
    // Standard practice (LLaMA, Mistral, GPT): eps = 1e-5 to 1e-6
    float rms_epsilon = 1e-5f;  // RMSNorm epsilon - standard value matching LLaMA/Mistral
    HyperParameters::PositionalEncodingType positional_encoding = HyperParameters::DEFAULT_POSITIONAL_ENCODING;
    bool causal_mask = true;                        
    bool use_pre_norm = true;
    bool fuse_qkv = true;
    bool use_simd = true;
    int num_threads = 4;
    
    // Flash Attention settings
    bool use_flash_attention = true;  // Use Flash Attention 2 for memory efficiency
    int min_seq_len_for_flash = 0;    // REQUIRED - set from hyperparameters (no defaults)
    
    // Positional encoding (ALiBi+RoPE hybrid) - pointer to shared state in TrainingState
    // WARNING: If nullptr, attention sees no positional info - positions become equivalent!
    const PBM::PBMSpec* pos_encoding = nullptr;
    
    // LayerScale (Issue #109/#129 fix - config propagation)
    // These fields were missing, causing reliance on EncoderLayerConfig defaults
    bool use_layer_scale = false;        // Enable per-sublayer learnable scaling
    float layer_scale_init = 1.0f;       // Issue #129: init=1.0 (NOT CaiT's 0.1 — caused 10x gradient attenuation)
    
    // Per-layer residual centering (Issue #126 fix - can be disabled to improve gradient signal)
    // When true, applies center_columns after each residual add in every encoder layer (24 total).
    // This prevents mode collapse but attenuates gradient signal through 24 centering projections.
    // When false, only the LM head centering (center_hidden_states) helps prevent mode collapse.
    bool center_encoder_residuals = false;
    
    // Bias control - when false, encoder layers skip bias addition (b_qkv, b_o not used)
    bool use_bias = true;

    // QK-Norm: RMSNorm applied to Q and K projections before attention scoring
    bool qk_norm_enabled = false;
    
    // Layer self-allocation (Pattern B): seed and scaling config for weight initialization
    uint64_t weight_seed = 0;         // Base seed for Xavier init (per-layer offset: seed + layer*10)
    float residual_scale = 1.0f;      // Issue #142: 1/sqrt(2*num_layers) for W_o and W2
    float layer_scale_init_value = 1.0f;  // Issue #109: CaiT LayerScale initial scalar value
    
    // CUDA execution
    cudaStream_t stream = nullptr;       // CUDA stream for async execution
    cublasHandle_t cublas_handle = nullptr;  // Centralized cuBLAS handle (Rule 22)
};

struct LMHeadConfig {
    int d_model = 0;        // MUST be populated from HyperParameters
    int vocab_size = 0;     // MUST come from .grmt training data or tokenizer
    bool tie_weights = false;
    const Matrix* embedding_weights = nullptr;
    bool use_bias = true;
    bool use_simd = true;
    float epsilon = 1e-5f;
};

enum class SamplingStrategy {
    GREEDY,
    TOP_K,
    TOP_P,
    MIN_P,           // Min-P (relative threshold)
    TYPICAL,         // Locally typical sampling
    TOP_K_TOP_P,     // Combined: Top-K first, then Top-P within survivors
    BEAM_SEARCH      // NOT SUPPORTED - exists only to give clear error
};

struct GenerationConfig {
    SamplingStrategy strategy = SamplingStrategy::TOP_P;
    int max_new_tokens = 100;
    int min_new_tokens = 0;
    float temperature = 1.0f;
    int top_k = 50;
    float top_p = 0.9f;
    float min_p = 0.0f;                    // Min-P threshold (0 = disabled)
    float typical_p = 1.0f;                // Typical sampling mass (1.0 = disabled)
    float repetition_penalty = 1.0f;
    int repetition_penalty_window = 64;
    float frequency_penalty = 0.0f;        // Additive penalty per occurrence (0 = disabled)
    float presence_penalty = 0.0f;         // Additive penalty if token appeared (0 = disabled)
    float length_penalty = 1.0f;
    int num_beams = 1;
    int num_return_sequences = 1;
    bool early_stopping = false;
    int eos_token_id = 0;
    int pad_token_id = 0;
    int bos_token_id = 2;
    int unk_token_id = 0;
    int no_repeat_ngram_size = 0;
    bool do_sample = true;
    float diversity_penalty = 0.0f;
    std::vector<int> bad_words_ids;
    unsigned int seed = 0;

    // ScratchBlock reasoning during inference
    // When true, generated atom tokens (numbers, URLs, etc.) are classified
    // and their metadata (numeric_value, atom_mask) is fed back into forwardStep()
    // so the ScratchBlock layer can inject structured reasoning embeddings.
    bool enable_scratchblock_reasoning = true;
};

using GenerationStreamCallback = std::function<void(int token_id, float score)>;

 struct ActivationQuantizationConfig {
    bool enabled = false;
    bool apply_to_embeddings = false;
    bool apply_to_encoder_outputs = false;
    bool apply_to_layer_caches = false;
    bool apply_to_qkv_cache = false;
    bool apply_to_logits = false;
    float scale = 1.0f;
    float clip_min = -127.0f;
    float clip_max = 127.0f;
    int zero_point = 0;
    bool symmetric = false;
};

struct LanguageModelConfig {
    // Architecture - MUST be populated from HyperParameters
    // DO NOT add hardcoded defaults here - use HyperParameters::loadModelArchitecture()
    int vocab_size = 0;        // MUST come from .grmt training data or tokenizer
    int d_model = 0;           // Use HyperParameters::DEFAULT_D_MODEL
    int num_heads = 0;         // Use HyperParameters::DEFAULT_NUM_HEADS
    int num_kv_heads = 0;      // Use HyperParameters::DEFAULT_NUM_KV_HEADS (GQA - Grouped Query Attention)
    int d_ff = 0;              // Use HyperParameters::DEFAULT_D_FF
    int num_layers = 0;        // Use HyperParameters::DEFAULT_NUM_LAYERS
    int max_seq_len = 0;       // Use HyperParameters::DEFAULT_MAX_SEQ_LEN
    float dropout_rate = 0.0f; // Use HyperParameters::DEFAULT_DROPOUT_RATE
    float residual_dropout_rate = 0.0f; // Use HyperParameters::DEFAULT_RESIDUAL_DROPOUT_RATE
    float attention_dropout = 0.0f; // Use HyperParameters::DEFAULT_ATTENTION_DROPOUT
    
    // Derived values - computed from above, DO NOT set directly
    // Call computeDerivedValues() after setting d_model/num_heads
    int head_dim = 0;          // = d_model / num_heads (computed)
    
    // Compute derived values from primary values - MUST be called after setting d_model/num_heads
    void computeDerivedValues() {
        if (num_heads > 0 && d_model > 0) {
            head_dim = d_model / num_heads;
        }
    }
    
    // Cache limits
    int max_cached_batch = 0;
    int max_cached_seq_len = 0;
    int max_tokens_per_batch = 0;  // Optional token budget for training logits/loss
    
    // Positional encoding configuration
    HyperParameters::PositionalEncodingType positional_encoding = HyperParameters::DEFAULT_POSITIONAL_ENCODING;
    
    // Fixed config values (not architecture-dependent)
    // Issue #104 FIX: Changed from 1e-3 to 1e-5 (see TransformerConfig above for rationale)
    float rms_epsilon = 1e-5f;  // RMSNorm epsilon - shared across all RMSNorm layers
    bool causal_mask = true;
    bool use_pre_norm = true;
    bool fuse_qkv = true;
    bool use_simd = true;
    int num_threads = 4;
    bool tie_embeddings = true;
    bool use_bias = true;
    bool qk_norm_enabled = false;  // QK-Norm: RMSNorm applied to Q and K before attention scoring
    
    // Issue #109: LayerScale - learnable residual scaling from CaiT paper
    // Reduces correlation buildup between layers by gating sublayer outputs
    // with learnable scalars (initialized to layer_scale_init, typically 0.1)
    bool use_layer_scale = false;         // Enable LayerScale (gated residual scaling)
    float layer_scale_init = 1.0f;       // Issue #129: init=1.0 (NOT CaiT's 0.1 — caused 10x gradient attenuation)
    
    bool use_gpu = true;
    bool use_flash_attention = true;  // Use Flash Attention 2 for memory efficiency
    int min_seq_len_for_flash = 0;     // REQUIRED - set from hyperparameters (no defaults)
    std::string vocab_path;            // Optional: source vocab file for auto-detection
    bool infer_vocab_from_file = false; // When true, read vocab size from vocab_path at init
    
    // Execution mode - determines memory allocation strategy
    ModelExecutionMode execution_mode = ModelExecutionMode::INFERENCE;
    
    // ScratchBlock reasoning layer config - populated from ai_config.json
    bool use_scratch_block = true;            // Enable ScratchBlock reasoning layer
    int scratch_block_atom_embedding_dim = 64; // Atom embedding dimension
    int scratch_block_max_atoms = 256;         // Max atoms per sequence
    float scratch_block_atom_scale = 1.0f;     // Scale factor for atom injection (unit scale)
    
    // LM Head centering config (Issue #37 / #40 fixes)
    // When enabled, centers hidden states before LM head projection.
    // Centering backward is handled automatically by CenterRowsGradFn/CenterColumnsGradFn
    // inside the autograd graph (Issues #125/#132).
    bool lm_head_center_hidden_states = false;  // Center encoder output before projection
    bool project_out_pc1 = false;              // Project out PC1 direction before LM head (Issue #149)
    int  pc1_power_iters = 5;                  // Power iteration steps for PC1 estimation
    bool center_logits = false;                 // Center logits per position (row-wise, mean→0)
    bool center_encoder_residuals = false;        // Center residuals INSIDE encoder layers. Prevents ρ buildup from causal attention prefix averaging.
                                                     // Gradient cost: negligible ((1-1/n_tokens)^24 ≈ 0.996 for n≈6000).
    
    // Hardcoded Hidden States Diagnostic (Issue #42)
    // When enabled, replaces encoder output with synthetic patterns to isolate
    // whether mode collapse is caused by encoder or LM head/gradient system.
    enum class HardcodedPattern {
        DISABLED,
        RANDOM_CENTERED,      // Random normal with mean=0 (tests Issue #37)
        ORTHOGONAL_W277,      // Orthogonal to W[277] (should give logit[277]≈0)
        ALIGNED_W277,         // Aligned with W[277] (tests collapse mechanism)
        CONSTANT_UNIFORM,     // Constant [1/√768] (tests Issue #40 row sum bias)
        ZERO_MEAN_SINE        // Sine wave with zero mean (centering robustness)
    };
    HardcodedPattern hardcoded_hidden_pattern = HardcodedPattern::DISABLED;
    int hardcoded_log_every_n_batches = 1;
    
    GenerationConfig generation;
    ActivationQuantizationConfig activation_quantization;

    // Multi-token prediction (MTP) - auxiliary heads for trajectory learning (Gloeckle et al. 2024)
    bool mtp_enabled = false;
    int mtp_k = 0;                    // Number of auxiliary future-prediction heads (2-4)
    float mtp_alpha = 0.2f;           // MTP loss coefficient
    int mtp_alpha_warmup_steps = 500; // Steps to linearly warm up alpha from 0 to mtp_alpha
};

// GPU Configuration - FULL definition (source of truth)
struct GPUConfig {
    bool use_tensor_cores = true;        // Enable FP16 Tensor Core ops
    bool use_cuda_graphs = true;         // Use CUDA graphs for repeated inference
    bool use_pinned_memory = true;       // Pinned host buffers
    bool use_dynamic_batching = true;    // Merge micro-batches
    int max_batch_size = 1024;            // Max dynamic batch size
    int micro_batch_size = 64;           // Micro-batch for long sequences
    cudaStream_t stream = 0;             // CUDA stream for async ops
    int device_id = 0;                   // CUDA device to use
};

//======================================================//
//  Forward Declarations
//======================================================//

// Positional Encoding Type - re-exported from HyperParameters_GPU.hpp
using HyperParameters::PositionalEncodingType;

class ALiBiPositionalBias {
private:

    PBM::PBMState pbm_state_{};   // Unified PBM state (both ALiBi + RoPE)
    int num_heads;                // Number of attention heads
    bool initialized;             // Initialization status
    PositionalEncodingType type;  // Encoding type

public:
    
    ALiBiPositionalBias();
    ALiBiPositionalBias(const ALiBiPositionalBias&) = delete;
    ALiBiPositionalBias& operator=(const ALiBiPositionalBias&) = delete;
    ALiBiPositionalBias(ALiBiPositionalBias&& other) noexcept;
    ALiBiPositionalBias& operator=(ALiBiPositionalBias&& other) noexcept;
    ~ALiBiPositionalBias();

    // num_kv_heads: REQUIRED. Only GQA (grouped-KV) is supported — pass model's num_kv_heads (<= num_heads).

    void computeSlopes(
        int num_heads,
        int num_kv_heads,
        int d_head,
        int max_seq_len,
        PositionalEncodingType type
    );


    float* getSlopes() const;
    float* getRoPEFreqs() const;
    bool isInitialized() const;
    PositionalEncodingType getType() const { return type; }
    void cleanup();
};

// GrimEmbeddingStack - minimal interface
// NOTE: Position embeddings are initialized directly on GPU in TrainingOps.cu
class GrimEmbeddingStack {
public:
    GrimEmbeddingStack(int vocab_size, int d_model, int max_seq_len);
    
    // num_kv_heads REQUIRED - only GQA is supported
    void enableALiBi(int num_heads, int num_kv_heads, int max_seq_len);
    void enableHybridPositionalEncoding(int num_heads, int d_head, int num_kv_heads, int max_seq_len);
    const ALiBiPositionalBias* getALiBiBias() const;
    const Matrix& getTokenEmbeddings() const;
    
    // Public members needed by GPU code
    Matrix token_embed;        // Token embedding matrix [vocab_size x d_model]
    Vector rms_gamma;          // RMSNorm gamma (scale)
    // NOTE: Position embeddings are GPU-only (initialized in TrainingOps.cu)
    
private:
    int vocab_size_;
    int d_model_;
    int max_seq_len_;
    std::unique_ptr<ALiBiPositionalBias> alibi_;
};

//======================================================//
//  Generated Sequence
//======================================================//

struct GeneratedSequence {
    std::vector<int> token_ids;
    std::vector<float> token_scores;
    std::vector<float> token_numeric_values;
    std::vector<uint8_t> token_atom_mask;
    std::shared_ptr<const GRIM::Tokenizer::AtomTable> context_atom_table;  // Atom registry from context (null for generated tokens)
    std::vector<uint32_t> atom_entry_ids;  // Per-token atom entry IDs (kAtomEntryNone = no atom)
    float score = 0.0f;
    bool finished = false;

    float getNormalizedScore(float length_penalty) const;
};

//======================================================//
//  Optimizer State for Training
//======================================================//

struct OptimizerState {
    // AdamW optimizer step counter for bias correction
    // NOTE: Actual moment buffers (m, v) are stored in ParameterGroup
    // AdamW hyperparameters are defined in LanguageModel_Training.cu
    int step = 0;
};

//======================================================//
//  Forward Declarations - Classes Defined Later
//======================================================//

class EncoderLayer;
class TextGenerator;
struct TrainingState;  // Forward declare for methods that return references

// Forward declare LossContext namespace and nested struct
namespace LossContext {
    struct LossOptions;
}

//======================================================//
//  Parameter Group for Training
//  (Defined in TensorContract_GPU.hpp - included above)
//======================================================//
// ParamGroupType and ParameterGroup are now part of the
// unified autograd system in TensorContract_GPU.hpp

// Forward declare TokenBufferView for method signatures
struct TokenBufferView;

#ifdef USE_CUDA
struct TokenBufferView {
    int* device_token_ids = nullptr;
    float* device_token_numeric_values = nullptr;
    uint8_t* device_token_atom_mask = nullptr;
    int max_tokens = 0;
    cudaStream_t stream = nullptr;
};
#endif

//======================================================//
//  LanguageModel Class Declaration
//======================================================//

class LanguageModel {
public:
    struct ModelStats {
        size_t total_params = 0;
        size_t embedding_params = 0;
        size_t position_embedding_params = 0;  // Position embeddings (max_seq_len * d_model)
        size_t encoder_params = 0;
        size_t lm_head_params = 0;
        size_t scratchblock_params = 0;  // Atom type embeddings + projection
        float model_size_mb = 0.0f;
    };

    struct UpdateProbeResult {
        std::string group_name;
        float parameter_rms = 0.0f;
        float grad_rms = 0.0f;
        float update_rms = 0.0f;
        float relative_update = 0.0f;
        float max_abs_update = 0.0f;
        float learning_rate = 0.0f;
        uint64_t optimizer_step = 0;
        uint32_t sample_size = 0;
    };
    
    // Constructor / Destructor
    explicit LanguageModel(const LanguageModelConfig& config);
    ~LanguageModel();
    
    // Main API
    Vector forward(const std::vector<int>& token_ids,
                   const std::vector<float>& token_numeric_values,
                   const std::vector<uint8_t>& token_atom_mask);
    Vector getNextTokenLogits(const std::vector<int>& context_tokens,
                              const std::vector<float>& context_numeric_values,
                              const std::vector<uint8_t>& context_atom_mask);
    std::vector<GeneratedSequence> generate(const std::vector<int>& prompt_tokens,
                                            const std::vector<float>& prompt_numeric_values,
                                            const std::vector<uint8_t>& prompt_atom_mask,
                                            const GenerationConfig* gen_config = nullptr,
                                            std::shared_ptr<const GRIM::Tokenizer::AtomTable> prompt_atom_table = nullptr,
                                            const std::vector<uint32_t>& prompt_atom_entry_ids = {});
    GeneratedSequence generateStream(const std::vector<int>& prompt_tokens,
                                     const std::vector<float>& prompt_numeric_values,
                                     const std::vector<uint8_t>& prompt_atom_mask,
                                     GenerationStreamCallback callback,
                                     const GenerationConfig* gen_config = nullptr,
                                     std::shared_ptr<const GRIM::Tokenizer::AtomTable> prompt_atom_table = nullptr,
                                     const std::vector<uint32_t>& prompt_atom_entry_ids = {});
    
    // Training
    float computeLossBatch(const GRIM::Batching::BatchPayload& payload,
                           bool is_training = true);
    
    // =========================================================================
    // INCREMENTAL GENERATION API (KV-Cache Autoregressive)
    // =========================================================================
    // Use these for efficient token-by-token generation:
    //   1. Call forwardInit() once with the prompt tokens - caches K,V for all
    //   2. Call forwardStep() for each new token - computes Q for new token only,
    //      attends to all cached K,V, appends new K,V to cache
    //   3. Call resetKVCache() before starting a new generation session
    // =========================================================================
    
    // Initialize KV cache with prompt tokens (prefill phase)
    // Returns logits for the last prompt token (ready for first sampling)
    Vector forwardInit(const std::vector<int>& prompt_tokens,
                       const std::vector<float>& prompt_numeric_values,
                       const std::vector<uint8_t>& prompt_atom_mask);
    
    // Process a single new token using cached K,V (decode phase)  
    // Returns logits for this token position (ready for next sampling)
    // Appends new token to cached sequence and recomputes full forward pass
    Vector forwardStep(int new_token, float numeric_value, uint8_t atom_mask);
    
    // Reconstruct the predicted numeric value from the cached numeric head
    // output produced by the most recent forward pass. Call AFTER sampling
    // a <NUM> token and BEFORE forwardStep appends the next token.
    float predictNumericValue() const;

    // Clear KV cache (call before starting new generation)
    void resetKVCache();
    
    // Get current KV cache length (number of tokens with cached K,V)
    int getKVCacheLength() const;
    
    void initCuBLASHandle();   // Initialize cuBLAS handle only (MUST be called before initGPU)
    void initPBM();            // Initialize PBM (ALiBi+RoPE hybrid) - MUST be called before initGPU
    void initTrainingState();  // Initialize training state (allocate GPU buffers + gradients)
    void initInferenceState();  // Initialize inference state (allocate GPU buffers WITHOUT gradients)
    // backward() and zeroGrad() DELETED (Rule 26).
    // Backward: Use autogradTrainingStep() which does forward+loss+backward.
    // Zeroing: executeAutogradBackward() zeros all gradients when accumulate=false.
    // updateWeights(), resetOptimizerMoments(), scaleOptimizerMoments() MOVED to
    // AdamW_Kernal_GPU.{hpp,cu} as free functions: launchAdamWStep(), resetAdamWMoments(),
    // scaleAdamWMoments(). AdamW stepping is training infrastructure, not model logic.
    // computeGradNorm(), scaleGradientsByType(), recordGradientClip() DELETED (Rule 26).
    // Phase2 calls GradNorm::measureGradientNorms() + launchScaleGradients() directly.
    void dumpGradientValues(int step, const std::string& filepath);  // Dump gradient values to text file for comparison
    
#ifdef USE_CUDA
    void setLossOptions(const LossContext::LossOptions& opts) { loss_options_ = opts; }
    const LossContext::LossOptions& getLossOptions() const { return loss_options_; }

    // Training state access (for debugging/diagnostics)
    const TrainingState& getTrainingState() const { return training_state_; }
    TrainingState& getTrainingState() { return training_state_; }
    
    // Parameter groups accessor (for direct gradient norm / clipping in Phase2)
    const std::vector<ParameterGroup>& parameterGroups() const { return parameter_groups_; }
    std::vector<ParameterGroup>& parameterGroups() { return parameter_groups_; }
    // Build parameter groups for optimizer (must be called during init so Phase2 grad-norm has max_groups > 0)
    void buildParameterGroups();
#endif

    const UpdateProbeResult& updateProbe() const { return update_probe_result_; }
    bool hasUpdateProbe() const { return update_probe_ready_; }
    void clearUpdateProbeFlag() { update_probe_ready_ = false; }
    void configureUpdateProbe(const std::string& group_name, size_t sample_elems = 1024);
    void disableUpdateProbe();
    const std::vector<float>& updateProbeWeightsBefore() const { return update_probe_weights_before_; }
    const std::vector<float>& updateProbeWeightsAfter() const { return update_probe_weights_after_; }
    const std::vector<float>& updateProbeGradSample() const { return update_probe_grad_sample_; }
    
    // Utilities
    ModelStats getModelStats() const;
    bool save(const std::string& path);
    bool load(const std::string& path);
    
    // Config access
    const LanguageModelConfig& getConfig() const { return config_; }
    
    // GPU methods
    void initGPU();
    Vector forwardGPU(const std::vector<int>& token_ids,
                      const std::vector<float>& token_numeric_values,
                      const std::vector<uint8_t>& token_atom_mask);
    Vector getNextTokenLogitsGPU(const std::vector<int>& context_tokens,
                                 const std::vector<float>& context_numeric_values,
                                 const std::vector<uint8_t>& context_atom_mask);
    TokenBufferView getTokenBufferView();
    void markDevicePromptReady(int token_count);
    
    // Helper accessors for GPU implementation
    GrimEmbeddingStack* getEmbedderPtr() { return embedder_.get(); }
    
#ifdef USE_CUDA
    // GPU runtime accessors - return references to owned objects (fail loud if not initialized)
    GPUGrimEncoder& getGpuEncoder();
    const GPUGrimEncoder& getGpuEncoder() const;

    
    // ScratchBlock reasoning layer access
    ScratchBlockLayer* getScratchBlockLayer() { return scratch_block_layer_.get(); }
    const ScratchBlockLayer* getScratchBlockLayer() const { return scratch_block_layer_.get(); }
    void setScratchBlockEnabled(bool enabled);
    bool isScratchBlockEnabled() const;

    // Embedding layer access (Pattern B: persistent, self-allocating, owns token + pos weights)
    EmbeddingLayer* getEmbeddingLayer() { return embedding_layer_.get(); }
    const EmbeddingLayer* getEmbeddingLayer() const { return embedding_layer_.get(); }

    // LM Head layer access (Pattern B: persistent, self-allocating)
    LMHeadLayer* getLmHeadLayer() { return lm_head_layer_.get(); }
    const LMHeadLayer* getLmHeadLayer() const { return lm_head_layer_.get(); }

    // Numeric Head layer access (nullptr when disabled)
    NumericHeadLayer* getNumericHeadLayer() { return numeric_head_layer_.get(); }
    const NumericHeadLayer* getNumericHeadLayer() const { return numeric_head_layer_.get(); }

    // Multi-token prediction (MTP) auxiliary heads - one weight + bias per head (indices 0..mtp_k-1)
    struct MTPHead {
        Tensor weight;  // [vocab_size, d_model] same layout as LM head
        Tensor bias;    // [vocab_size]
    };
    int getMtpK() const { return config_.mtp_enabled ? config_.mtp_k : 0; }
    MTPHead* getMtpHead(int k);
    const MTPHead* getMtpHead(int k) const;
    
    // Scratch pool configuration (pinned memory transfers)
    void configureScratchPool(bool enabled);
    bool isScratchPoolInitialized() const;
#endif
    
    GeneratedSequence generateSequenceGPU(const std::vector<int>& prompt_tokens,
                                          const std::vector<float>& prompt_numeric_values,
                                          const std::vector<uint8_t>& prompt_atom_mask,
                                          const GenerationConfig& cfg,
                                          GenerationStreamCallback* stream_callback = nullptr,
                                          std::shared_ptr<const GRIM::Tokenizer::AtomTable> prompt_atom_table = nullptr,
                                          const std::vector<uint32_t>& prompt_atom_entry_ids = {});
    
private:
    // Core inference forward: assumes data already in cached_* tensors.
    // Runs autograd forward, extracts last-token logits, returns them.
    // All public inference methods (forwardGPU, getNextTokenLogitsGPU,
    // forwardInit, forwardStep) copy their data to cached tensors then call this.
    Vector executeInferenceForward_(int seq_len);

    LanguageModelConfig config_;
    std::unique_ptr<GrimEmbeddingStack> embedder_;
    
#ifdef USE_CUDA
    // GPU runtime ownership (StreamController model - proper typed ownership, no void*)
    std::unique_ptr<GPUGrimEncoder> gpu_encoder_;
#endif
    
    bool staged_prompt_ready_ = false;
    int staged_prompt_len_ = 0;
    
    UpdateProbeResult update_probe_result_;
    bool update_probe_ready_ = false;
    float last_grad_scale_ = 1.0f;
    std::string update_probe_group_name_;
    size_t update_probe_group_index_ = static_cast<size_t>(-1);
    size_t update_probe_sample_elems_ = 0;
    size_t update_probe_sample_offset_ = 0;  // Offset into buffer to sample from (token 277 for embedding/LM head)
    std::vector<float> update_probe_weights_before_;
    std::vector<float> update_probe_weights_after_;
    std::vector<float> update_probe_grad_sample_;

#ifdef USE_CUDA
    TrainingState training_state_;
    std::vector<ParameterGroup> parameter_groups_;  // Parameter groups for optimizer
    uint32_t backward_call_count_ = 0;              // Tracks backward() calls for deterministic diagnostics
    LossContext::LossOptions loss_options_{};
    
    // ScratchBlock reasoning layer (togglable)
    std::unique_ptr<ScratchBlockLayer> scratch_block_layer_;

    // Embedding layer (Pattern B: persistent, self-allocating, owns token + position weights)
    std::unique_ptr<EmbeddingLayer> embedding_layer_;

    // LM Head layer (Pattern B: persistent, self-allocating, owns final_rms_gamma)
    std::unique_ptr<LMHeadLayer> lm_head_layer_;

    // Numeric Head layer (predicts log-magnitude + sign for <NUM> tokens)
    std::unique_ptr<NumericHeadLayer> numeric_head_layer_;

    // Multi-token prediction (MTP) auxiliary heads - K independent linear heads (not tied to embedding)
    std::vector<MTPHead> mtp_heads_;
#endif
};

using StreamCallback = GenerationStreamCallback;

//======================================================//
//  GPU Classes (Forward Declarations Only)
//======================================================//

#ifdef USE_CUDA
// Forward declarations for GPU classes
struct FlashAttentionBF16Scratch {
    __nv_bfloat16* q = nullptr;
    __nv_bfloat16* k = nullptr;
    __nv_bfloat16* v = nullptr;
    __nv_bfloat16* out = nullptr;
    size_t q_elems = 0;
    size_t kv_elems = 0;
};
class EncodingLayer;
using GPUEncoderLayer = EncodingLayer;

// GPUGrimEncoder - Container for encoder layers, manages layer lifecycle
// Forward pass logic is in ForwardPhase2_Encoder.cu::runFullEncoder()
// This class only owns layers and provides access to them
class GPUGrimEncoder {
public:
    explicit GPUGrimEncoder(const EncoderConfig& config);
    
    // Access to layers for training/forward pass
    GPUEncoderLayer* getLayer(int index);
    const GPUEncoderLayer* getLayer(int index) const;
    int getNumLayers() const;
    
    // Configure Flash Attention for all layers
    void setFlashAttention(bool enable, int min_seq_len);
    
private:
    struct Impl;
    Impl* pImpl = nullptr;
};

#endif // USE_CUDA

} // namespace GRIM
