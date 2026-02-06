#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <filesystem>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "../HyperParameters/HyperParameters_GPU.hpp"  // For DEFAULT_EQ_LOG_ENABLED, DEFAULT_EQ_LOG_INTERVAL

namespace GRIM {

// ============================================================================
// EQUATION LOGGING SYSTEM - CUDA-optimized with async queue
// ============================================================================
// Design goals:
// 1. NO device syncs during training - all logging is async
// 2. Phase-ordered queue - entries logged in phase order per step
// 3. Ring buffer on device - fixed memory, no allocations during training
// 4. Periodic async flush - batch transfer to host when buffer fills
// ============================================================================

// Fixed-size string for device-side storage (no dynamic allocation)
static constexpr int EQ_LOG_STRING_LEN = 256;
static constexpr int EQ_LOG_BUFFER_SIZE = 4096;  // Ring buffer capacity

// Phase identifiers for ordered logging
enum class EquationPhase : int {
    EMBEDDING_LOOKUP = 0,
    POSITION_ENCODING = 1,
    RMSNORM_PRE_ATTN = 2,
    QKV_PROJECTION = 3,
    ROPE_ROTATION = 4,
    ALIBI_BIAS = 5,
    FLASH_ATTENTION_FWD = 6,
    ATTENTION_OUTPUT = 7,
    RMSNORM_PRE_FFN = 8,
    FFN_LAYER1 = 9,
    FFN_GELU = 10,
    FFN_LAYER2 = 11,
    LM_HEAD_PROJECTION = 12,
    LOSS_COMPUTATION = 13,
    // Backward phases
    LOSS_BACKWARD = 14,
    LM_HEAD_BACKWARD = 15,
    FFN_BACKWARD = 16,
    ATTENTION_BACKWARD = 17,
    RMSNORM_BACKWARD = 18,
    EMBEDDING_BACKWARD = 19,
    // Optimizer phases
    GRADIENT_CLIP = 20,
    ADAMW_UPDATE = 21,
    WEIGHT_DECAY = 22,
    // Additional phases for diagnostic logging
    RMSNORM = 23,
    RESIDUAL_ADD = 24,
    ATTENTION_SCORE = 25,
    
    // Token 277 (SPACE) mode collapse tracking phases
    TOKEN277_LOGIT_DECOMPOSITION = 26,  // logit[277] = ||h|| * ||W|| * cos(h,W)
    TOKEN277_SOFTMAX_PROB = 27,         // p(277) = exp(logit[277]) / sum(exp(logits))
    TOKEN277_ARGMAX_ANALYSIS = 28,      // Is 277 the argmax? If so, by how much?
    TOKEN277_GRADIENT_FLOW = 29,        // grad_W[277] from loss backward
    TOKEN277_WEIGHT_UPDATE = 30,        // AdamW update to W[277]
    TOKEN277_HIDDEN_ALIGNMENT = 31,     // cos(h, W[277]) per position
    TOKEN277_LOSS_CONTRIBUTION = 32,    // CE loss when target != 277 but pred = 277
    
    // MSE Loss for regression/numeric heads
    MSE_LOSS_FORWARD = 33,              // E = (1/2)(y - a)² forward computation
    MSE_LOSS_BACKWARD = 34,             // dE/da = -(y - a) backward gradient
    
    PHASE_COUNT = 35
};

// Convert phase to string for logging
inline const char* phaseToString(EquationPhase phase) {
    switch (phase) {
        case EquationPhase::EMBEDDING_LOOKUP:    return "EMBEDDING_LOOKUP";
        case EquationPhase::POSITION_ENCODING:   return "POSITION_ENCODING";
        case EquationPhase::RMSNORM_PRE_ATTN:    return "RMSNORM_PRE_ATTN";
        case EquationPhase::QKV_PROJECTION:      return "QKV_PROJECTION";
        case EquationPhase::ROPE_ROTATION:       return "ROPE_ROTATION";
        case EquationPhase::ALIBI_BIAS:          return "ALIBI_BIAS";
        case EquationPhase::FLASH_ATTENTION_FWD: return "FLASH_ATTENTION_FWD";
        case EquationPhase::ATTENTION_OUTPUT:    return "ATTENTION_OUTPUT";
        case EquationPhase::RMSNORM_PRE_FFN:     return "RMSNORM_PRE_FFN";
        case EquationPhase::FFN_LAYER1:          return "FFN_LAYER1";
        case EquationPhase::FFN_GELU:            return "FFN_GELU";
        case EquationPhase::FFN_LAYER2:          return "FFN_LAYER2";
        case EquationPhase::LM_HEAD_PROJECTION:  return "LM_HEAD_PROJECTION";
        case EquationPhase::LOSS_COMPUTATION:    return "LOSS_COMPUTATION";
        case EquationPhase::LOSS_BACKWARD:       return "LOSS_BACKWARD";
        case EquationPhase::LM_HEAD_BACKWARD:    return "LM_HEAD_BACKWARD";
        case EquationPhase::FFN_BACKWARD:        return "FFN_BACKWARD";
        case EquationPhase::ATTENTION_BACKWARD:  return "ATTENTION_BACKWARD";
        case EquationPhase::RMSNORM_BACKWARD:    return "RMSNORM_BACKWARD";
        case EquationPhase::EMBEDDING_BACKWARD:  return "EMBEDDING_BACKWARD";
        case EquationPhase::GRADIENT_CLIP:       return "GRADIENT_CLIP";
        case EquationPhase::ADAMW_UPDATE:        return "ADAMW_UPDATE";
        case EquationPhase::WEIGHT_DECAY:        return "WEIGHT_DECAY";
        case EquationPhase::RMSNORM:             return "RMSNORM";
        case EquationPhase::RESIDUAL_ADD:        return "RESIDUAL_ADD";
        case EquationPhase::ATTENTION_SCORE:    return "ATTENTION_SCORE";
        // Token 277 tracking phases
        case EquationPhase::TOKEN277_LOGIT_DECOMPOSITION: return "TOKEN277_LOGIT_DECOMPOSITION";
        case EquationPhase::TOKEN277_SOFTMAX_PROB:        return "TOKEN277_SOFTMAX_PROB";
        case EquationPhase::TOKEN277_ARGMAX_ANALYSIS:     return "TOKEN277_ARGMAX_ANALYSIS";
        case EquationPhase::TOKEN277_GRADIENT_FLOW:       return "TOKEN277_GRADIENT_FLOW";
        case EquationPhase::TOKEN277_WEIGHT_UPDATE:       return "TOKEN277_WEIGHT_UPDATE";
        case EquationPhase::TOKEN277_HIDDEN_ALIGNMENT:    return "TOKEN277_HIDDEN_ALIGNMENT";
        case EquationPhase::TOKEN277_LOSS_CONTRIBUTION:   return "TOKEN277_LOSS_CONTRIBUTION";
        // MSE Loss phases
        case EquationPhase::MSE_LOSS_FORWARD:             return "MSE_LOSS_FORWARD";
        case EquationPhase::MSE_LOSS_BACKWARD:            return "MSE_LOSS_BACKWARD";
        default: return "UNKNOWN";
    }
}

// Device-compatible log entry (POD type, fixed size)
struct alignas(16) EquationLogEntryDevice {
    char equation_name[EQ_LOG_STRING_LEN];     // e.g., "RMSNORM_FWD"
    char formula[EQ_LOG_STRING_LEN];           // e.g., "y = x * gamma / sqrt(mean(x²) + eps)"
    char inputs[EQ_LOG_STRING_LEN];            // e.g., "x: shape=[7168,768] rms=0.0063"
    char outputs[EQ_LOG_STRING_LEN];           // e.g., "y: shape=[7168,768] rms=0.9996"
    char expected[EQ_LOG_STRING_LEN];          // e.g., "expected rms ≈ 1.0"
    char actual[EQ_LOG_STRING_LEN];            // e.g., "actual rms = 0.9996"
    
    int batch_idx;                              // Current batch number
    int layer_idx;                              // Encoder layer (-1 if N/A)
    int step_idx;                               // Global training step
    EquationPhase phase;                        // Phase in computation pipeline
    float timestamp_ms;                         // Relative timestamp in milliseconds
    
    // Sequence number for ordering
    uint64_t sequence_num;                      // Global sequence counter
    
    // Valid flag (for ring buffer)
    int valid;                                  // 1 = valid entry, 0 = slot empty
};

// Device-side ring buffer state
struct EquationLogBufferState {
    uint64_t write_head;        // Atomic: next slot to write (modulo buffer size)
    uint64_t read_head;         // Host-side: next slot to read
    uint64_t sequence_counter;  // Atomic: global sequence number
    uint64_t entries_dropped;   // Counter for overflow tracking
    int buffer_size;            // Capacity
    int flush_threshold;        // Trigger flush when this many entries pending
};

#ifdef __CUDACC__

// ============================================================================
// EXTERN DECLARATIONS FOR DEVICE GLOBALS (defined in EquationLogging.cu)
// ============================================================================
// These must be declared at file scope so any .cu file including this header
// can reference them in device code (kernels).
extern __device__ EquationLogEntryDevice* g_eq_log_buffer;
extern __device__ EquationLogBufferState* g_eq_log_state;

// ============================================================================
// DEVICE-SIDE LOGGING KERNEL AND HELPERS
// ============================================================================

// Device helper: safe string copy (no std::string on device)
__device__ __forceinline__ int eq_strcpy_device(char* dst, const char* src, int max_len) {
    int i = 0;
    while (i < max_len - 1 && src[i] != '\0') {
        dst[i] = src[i];
        i++;
    }
    dst[i] = '\0';
    return i;  // Return length written (matches eq_itoa_device, eq_ftoa_device pattern)
}

// Device helper: format integer into string (returns length written)
__device__ __forceinline__ int eq_itoa_device(char* buf, int val, int max_len) {
    if (max_len < 12) return 0;  // Need space for -2147483648 + null
    
    int len = 0;
    bool negative = (val < 0);
    if (negative) {
        buf[len++] = '-';
        val = -val;
    }
    
    // Handle zero
    if (val == 0) {
        buf[len++] = '0';
        buf[len] = '\0';
        return len;
    }
    
    // Build digits in reverse
    char temp[12];
    int temp_len = 0;
    while (val > 0 && temp_len < 10) {
        temp[temp_len++] = '0' + (val % 10);
        val /= 10;
    }
    
    // Copy in correct order
    for (int i = temp_len - 1; i >= 0 && len < max_len - 1; i--) {
        buf[len++] = temp[i];
    }
    
    buf[len] = '\0';
    return len;
}

// Device helper: format float into string
__device__ __forceinline__ int eq_ftoa_device(char* buf, float val, int max_len) {
    if (max_len < 16) return 0;
    
    // Handle special cases
    if (isnan(val)) { buf[0]='N'; buf[1]='a'; buf[2]='N'; buf[3]='\0'; return 3; }
    if (isinf(val)) { 
        if (val < 0) { buf[0]='-'; buf[1]='I'; buf[2]='n'; buf[3]='f'; buf[4]='\0'; return 4; }
        else { buf[0]='I'; buf[1]='n'; buf[2]='f'; buf[3]='\0'; return 3; }
    }
    
    // Simple float formatting (scientific notation for extreme values)
    int len = 0;
    if (val < 0) { buf[len++] = '-'; val = -val; }
    
    if (val == 0.0f) {
        buf[len++] = '0'; buf[len++] = '.'; buf[len++] = '0'; buf[len] = '\0';
        return len;
    }
    
    // Use scientific notation for very large/small values
    int exp = 0;
    while (val >= 10.0f && exp < 38) { val /= 10.0f; exp++; }
    while (val < 1.0f && exp > -38) { val *= 10.0f; exp--; }
    
    // Integer part
    int int_part = (int)val;
    buf[len++] = '0' + int_part;
    buf[len++] = '.';
    
    // Fractional part (6 digits)
    val -= int_part;
    for (int i = 0; i < 6 && len < max_len - 5; i++) {
        val *= 10.0f;
        int digit = (int)val;
        buf[len++] = '0' + digit;
        val -= digit;
    }
    
    // Exponent
    if (exp != 0) {
        buf[len++] = 'e';
        if (exp < 0) { buf[len++] = '-'; exp = -exp; }
        else { buf[len++] = '+'; }
        if (exp >= 10) buf[len++] = '0' + (exp / 10);
        buf[len++] = '0' + (exp % 10);
    }
    
    buf[len] = '\0';
    return len;
}

// Device helper: append string
__device__ __forceinline__ int eq_strcat_device(char* dst, const char* src, int dst_len, int max_len) {
    int i = dst_len;
    int j = 0;
    while (i < max_len - 1 && src[j] != '\0') {
        dst[i++] = src[j++];
    }
    dst[i] = '\0';
    return i;
}

// ============================================================================
// DEVICE KERNEL: Enqueue a log entry (called from any CUDA kernel)
// ============================================================================
// This runs on ONE thread (typically thread 0 of a kernel) to log an equation
// Uses atomics to safely write to the ring buffer without host sync
// ============================================================================

__device__ __forceinline__ void enqueueEquationLog(
    EquationLogEntryDevice* d_buffer,
    EquationLogBufferState* d_state,
    const char* equation_name,
    const char* formula,
    const char* inputs,
    const char* outputs,
    const char* expected,
    const char* actual,
    int batch_idx,
    int layer_idx,
    int step_idx,
    EquationPhase phase
) {
    // Atomically claim a slot in the ring buffer
    uint64_t slot = atomicAdd((unsigned long long*)&d_state->write_head, 1ULL);
    uint64_t seq = atomicAdd((unsigned long long*)&d_state->sequence_counter, 1ULL);
    
    // Compute actual index (ring buffer wrap)
    int idx = (int)(slot % d_state->buffer_size);
    
    // Check for overflow (old entry not yet read)
    if (d_buffer[idx].valid && d_buffer[idx].sequence_num > d_state->read_head) {
        atomicAdd((unsigned long long*)&d_state->entries_dropped, 1ULL);
        return;  // Drop this entry to avoid corrupting unread data
    }
    
    // Write entry
    EquationLogEntryDevice* entry = &d_buffer[idx];
    
    eq_strcpy_device(entry->equation_name, equation_name, EQ_LOG_STRING_LEN);
    eq_strcpy_device(entry->formula, formula, EQ_LOG_STRING_LEN);
    eq_strcpy_device(entry->inputs, inputs, EQ_LOG_STRING_LEN);
    eq_strcpy_device(entry->outputs, outputs, EQ_LOG_STRING_LEN);
    eq_strcpy_device(entry->expected, expected, EQ_LOG_STRING_LEN);
    eq_strcpy_device(entry->actual, actual, EQ_LOG_STRING_LEN);
    
    entry->batch_idx = batch_idx;
    entry->layer_idx = layer_idx;
    entry->step_idx = step_idx;
    entry->phase = phase;
    entry->sequence_num = seq;
    
    // Timestamp (clock64 relative to some baseline)
    entry->timestamp_ms = 0.0f;  // Set by host during flush using CUDA events
    
    // Memory fence then mark valid
    __threadfence();
    entry->valid = 1;
}

// ============================================================================
// HELPER MACROS FOR EASY LOGGING FROM KERNELS
// ============================================================================

#define EQ_LOG_BUFFER_DECL \
    extern __device__ EquationLogEntryDevice* g_eq_log_buffer; \
    extern __device__ EquationLogBufferState* g_eq_log_state;

// Log from thread 0 only (typical pattern)
#define EQ_LOG_IF_THREAD0(name, formula, inputs, outputs, expected, actual, batch, layer, step, phase) \
    do { \
        if (threadIdx.x == 0 && blockIdx.x == 0 && g_eq_log_buffer && g_eq_log_state) { \
            enqueueEquationLog(g_eq_log_buffer, g_eq_log_state, \
                name, formula, inputs, outputs, expected, actual, \
                batch, layer, step, phase); \
        } \
    } while(0)

// ============================================================================
// TOKEN 277 (SPACE) MODE COLLAPSE TRACKING FUNCTIONS
// ============================================================================
// These functions track all components contributing to token 277 becoming argmax.
// See PLATEAU_BUG_INVESTIGATION.md for root cause analysis:
//   logit[277] = ||h|| × ||W[277]|| × cos(h, W[277])
// Mode collapse occurs when hidden states align with W[277] direction.

// Tracks: logit[277] = ||h|| × ||W|| × cos(h, W) decomposition
__device__ __forceinline__ void logToken277LogitDecomposition(
    EquationLogEntryDevice* d_buffer,
    EquationLogBufferState* d_state,
    float hidden_norm,          // ||h|| - hidden state magnitude
    float weight_277_norm,      // ||W[277]|| - weight row magnitude
    float cosine_h_w277,        // cos(h, W[277]) - alignment
    float predicted_logit,      // ||h|| × ||W|| × cos (expected)
    float actual_logit_277,     // h · W[277]^T (actual)
    float max_other_logit,      // Max logit excluding 277
    int max_other_token,        // Token ID with max_other_logit
    int position_idx,
    int batch_idx,
    int step_idx)
{
    char inputs[EQ_LOG_STRING_LEN];
    char expected[EQ_LOG_STRING_LEN];
    char actual[EQ_LOG_STRING_LEN];
    
    // Build inputs: ||h||, ||W[277]||, cos(h, W[277])
    int pos = 0;
    const char* p = "||h||=";
    while (*p) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, hidden_norm, EQ_LOG_STRING_LEN - pos);
    p = " ||W277||=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, weight_277_norm, EQ_LOG_STRING_LEN - pos);
    p = " cos=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, cosine_h_w277, EQ_LOG_STRING_LEN - pos);
    p = " pos=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_itoa_device(inputs + pos, position_idx, EQ_LOG_STRING_LEN - pos);
    inputs[pos] = '\0';
    
    // Build expected (predicted from decomposition)
    pos = 0;
    p = "logit277_predicted=";
    while (*p) expected[pos++] = *p++;
    pos += eq_ftoa_device(expected + pos, predicted_logit, EQ_LOG_STRING_LEN - pos);
    expected[pos] = '\0';
    
    // Build actual with comparison to max other token
    pos = 0;
    p = "logit277_actual=";
    while (*p) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, actual_logit_277, EQ_LOG_STRING_LEN - pos);
    p = " max_other=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, max_other_logit, EQ_LOG_STRING_LEN - pos);
    p = "(tok=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    pos += eq_itoa_device(actual + pos, max_other_token, EQ_LOG_STRING_LEN - pos);
    p = ")";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    
    // Add margin indicator
    float margin = actual_logit_277 - max_other_logit;
    p = " margin=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, margin, EQ_LOG_STRING_LEN - pos);
    actual[pos] = '\0';
    
    enqueueEquationLog(d_buffer, d_state,
                      "TOKEN277_LOGIT",
                      "logit[277] = ||h|| * ||W[277]|| * cos(h, W[277])",
                      inputs, "", expected, actual,
                      batch_idx, -1, step_idx, EquationPhase::TOKEN277_LOGIT_DECOMPOSITION);
}

// Tracks: p(277) = exp(logit[277] - LSE) via softmax
__device__ __forceinline__ void logToken277SoftmaxProb(
    EquationLogEntryDevice* d_buffer,
    EquationLogBufferState* d_state,
    float logit_277,            // Raw logit for token 277
    float max_logit,            // Max logit (for numerical stability)
    float log_sum_exp,          // LSE = log(Σexp(logit_i - max))
    float prob_277,             // Final probability for token 277
    float expected_uniform,     // 1/vocab_size (baseline)
    int position_idx,
    int batch_idx,
    int step_idx
) {
    char inputs[EQ_LOG_STRING_LEN];
    char expected[EQ_LOG_STRING_LEN];
    char actual[EQ_LOG_STRING_LEN];
    
    int pos = 0;
    const char* p = "logit277=";
    while (*p) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, logit_277, EQ_LOG_STRING_LEN - pos);
    p = " max_logit=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, max_logit, EQ_LOG_STRING_LEN - pos);
    p = " LSE=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, log_sum_exp, EQ_LOG_STRING_LEN - pos);
    p = " pos=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_itoa_device(inputs + pos, position_idx, EQ_LOG_STRING_LEN - pos);
    inputs[pos] = '\0';
    
    pos = 0;
    p = "p277_uniform=";
    while (*p) expected[pos++] = *p++;
    pos += eq_ftoa_device(expected + pos, expected_uniform, EQ_LOG_STRING_LEN - pos);
    expected[pos] = '\0';
    
    pos = 0;
    p = "p277_actual=";
    while (*p) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, prob_277, EQ_LOG_STRING_LEN - pos);
    p = " ratio_to_uniform=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    float ratio = prob_277 / fmaxf(expected_uniform, 1e-10f);
    pos += eq_ftoa_device(actual + pos, ratio, EQ_LOG_STRING_LEN - pos);
    actual[pos] = '\0';
    
    enqueueEquationLog(d_buffer, d_state,
                      "TOKEN277_SOFTMAX",
                      "p(277) = exp(logit[277] - max - LSE)",
                      inputs, "", expected, actual,
                      batch_idx, -1, step_idx, EquationPhase::TOKEN277_SOFTMAX_PROB);
}

// Tracks: argmax analysis - when does 277 win?
__device__ __forceinline__ void logToken277ArgmaxAnalysis(
    EquationLogEntryDevice* d_buffer,
    EquationLogBufferState* d_state,
    int actual_argmax_token,    // Token that won argmax
    float actual_argmax_logit,  // Logit of winner
    float logit_277,            // Token 277's logit
    float gap_to_argmax,        // argmax_logit - logit_277
    int target_token,           // Ground truth target
    bool is_277_argmax,         // Did 277 win?
    bool is_277_target,         // Is 277 the correct answer?
    int position_idx,
    int batch_idx,
    int step_idx
) {
    char inputs[EQ_LOG_STRING_LEN];
    char expected[EQ_LOG_STRING_LEN];
    char actual[EQ_LOG_STRING_LEN];
    
    int pos = 0;
    const char* p = "argmax_tok=";
    while (*p) inputs[pos++] = *p++;
    pos += eq_itoa_device(inputs + pos, actual_argmax_token, EQ_LOG_STRING_LEN - pos);
    p = " argmax_logit=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, actual_argmax_logit, EQ_LOG_STRING_LEN - pos);
    p = " target=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_itoa_device(inputs + pos, target_token, EQ_LOG_STRING_LEN - pos);
    p = " pos=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_itoa_device(inputs + pos, position_idx, EQ_LOG_STRING_LEN - pos);
    inputs[pos] = '\0';
    
    pos = 0;
    p = "expect: 277_is_argmax=";
    while (*p) expected[pos++] = *p++;
    expected[pos++] = is_277_target ? '1' : '0';
    expected[pos] = '\0';
    
    pos = 0;
    p = "logit277=";
    while (*p) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, logit_277, EQ_LOG_STRING_LEN - pos);
    p = " gap=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, gap_to_argmax, EQ_LOG_STRING_LEN - pos);
    p = " 277_IS_ARGMAX=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    actual[pos++] = is_277_argmax ? '1' : '0';
    
    if (is_277_argmax && !is_277_target) {
        p = " [COLLAPSE_RISK]";
        while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    }
    actual[pos] = '\0';
    
    enqueueEquationLog(d_buffer, d_state,
                      "TOKEN277_ARGMAX",
                      "argmax(logits) == 277 ? gap = max_logit - logit[277]",
                      inputs, "", expected, actual,
                      batch_idx, -1, step_idx, EquationPhase::TOKEN277_ARGMAX_ANALYSIS);
}

// Tracks: gradient flow through W[277] updates
__device__ __forceinline__ void logToken277GradientFlow(
    EquationLogEntryDevice* d_buffer,
    EquationLogBufferState* d_state,
    float grad_w277_norm,       // ||grad_W[277]|| - gradient magnitude
    float grad_w277_sum,        // Σgrad_W[277] - should be ~0 if centered
    float grad_w277_mean,       // mean(grad_W[277])
    float grad_logits_277_sum,  // Σgrad_logits[:,277] across positions
    float hidden_mean_sum,      // Σmean(h) - centering check
    float expected_direction,   // Sign: neg=decrease, pos=increase
    int num_positions,
    int batch_idx,
    int step_idx)
{
    char inputs[EQ_LOG_STRING_LEN];
    char expected[EQ_LOG_STRING_LEN];
    char actual[EQ_LOG_STRING_LEN];
    
    int pos = 0;
    const char* p = "grad_logits277_sum=";
    while (*p) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, grad_logits_277_sum, EQ_LOG_STRING_LEN - pos);
    p = " hidden_bias=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, hidden_mean_sum, EQ_LOG_STRING_LEN - pos);
    p = " n_pos=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_itoa_device(inputs + pos, num_positions, EQ_LOG_STRING_LEN - pos);
    inputs[pos] = '\0';
    
    pos = 0;
    p = "grad_direction=";
    while (*p) expected[pos++] = *p++;
    pos += eq_ftoa_device(expected + pos, expected_direction, EQ_LOG_STRING_LEN - pos);
    p = " (neg=shrink)";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) expected[pos++] = *p++;
    expected[pos] = '\0';
    
    pos = 0;
    p = "||grad_W277||=";
    while (*p) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, grad_w277_norm, EQ_LOG_STRING_LEN - pos);
    p = " sum=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, grad_w277_sum, EQ_LOG_STRING_LEN - pos);
    p = " mean=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, grad_w277_mean, EQ_LOG_STRING_LEN - pos);
    
    if (fabsf(grad_w277_sum) > 0.01f) {
        p = " [CENTERING_ANOMALY]";
        while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    }
    actual[pos] = '\0';
    
    enqueueEquationLog(d_buffer, d_state,
                      "TOKEN277_GRAD",
                      "grad_W[277,d] = sum_t(h[t,d] * grad_logits[t,277])",
                      inputs, "", expected, actual,
                      batch_idx, -1, step_idx, EquationPhase::TOKEN277_GRADIENT_FLOW);
}

// Tracks: W[277] weight update via AdamW
__device__ __forceinline__ void logToken277WeightUpdate(
    EquationLogEntryDevice* d_buffer,
    EquationLogBufferState* d_state,
    float w277_norm_before,     // ||W[277]|| before update
    float w277_norm_after,      // ||W[277]|| after update
    float update_norm,          // ||W_new - W_old||
    float m277_norm,            // AdamW momentum ||m[277]||
    float v277_norm,            // AdamW variance ||v[277]||
    float lr,                   // Learning rate
    int batch_idx,
    int step_idx)
{
    char inputs[EQ_LOG_STRING_LEN];
    char expected[EQ_LOG_STRING_LEN];
    char actual[EQ_LOG_STRING_LEN];
    
    int pos = 0;
    const char* p = "||W277_pre||=";
    while (*p) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, w277_norm_before, EQ_LOG_STRING_LEN - pos);
    p = " ||m277||=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, m277_norm, EQ_LOG_STRING_LEN - pos);
    p = " ||v277||=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, v277_norm, EQ_LOG_STRING_LEN - pos);
    p = " lr=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, lr, EQ_LOG_STRING_LEN - pos);
    inputs[pos] = '\0';
    
    pos = 0;
    p = "||update||=";
    while (*p) expected[pos++] = *p++;
    pos += eq_ftoa_device(expected + pos, update_norm, EQ_LOG_STRING_LEN - pos);
    expected[pos] = '\0';
    
    pos = 0;
    p = "||W277_post||=";
    while (*p) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, w277_norm_after, EQ_LOG_STRING_LEN - pos);
    float delta_norm = w277_norm_after - w277_norm_before;
    p = " delta=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, delta_norm, EQ_LOG_STRING_LEN - pos);
    
    if (delta_norm > 0.0001f) {
        p = " [NORM_INCREASED]";
        while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    }
    actual[pos] = '\0';
    
    enqueueEquationLog(d_buffer, d_state,
                      "TOKEN277_UPDATE",
                      "W[277] = W[277] - lr * (m_hat/sqrt(v_hat+eps) + wd*W)",
                      inputs, "", expected, actual,
                      batch_idx, -1, step_idx, EquationPhase::TOKEN277_WEIGHT_UPDATE);
}

// Tracks: hidden state alignment with W[277] direction
__device__ __forceinline__ void logToken277HiddenAlignment(
    EquationLogEntryDevice* d_buffer,
    EquationLogBufferState* d_state,
    float avg_cosine,           // Mean cos(h_t, W[277]) across positions
    float min_cosine,           // Min cosine
    float max_cosine,           // Max cosine
    float std_cosine,           // Std dev of cosines
    float expected_random,      // 1/sqrt(d_model) ≈ 0.036 baseline
    int num_positions,
    int batch_idx,
    int step_idx)
{
    char inputs[EQ_LOG_STRING_LEN];
    char expected[EQ_LOG_STRING_LEN];
    char actual[EQ_LOG_STRING_LEN];
    
    int pos = 0;
    const char* p = "n_positions=";
    while (*p) inputs[pos++] = *p++;
    pos += eq_itoa_device(inputs + pos, num_positions, EQ_LOG_STRING_LEN - pos);
    inputs[pos] = '\0';
    
    pos = 0;
    p = "random_cos~";
    while (*p) expected[pos++] = *p++;
    pos += eq_ftoa_device(expected + pos, expected_random, EQ_LOG_STRING_LEN - pos);
    expected[pos] = '\0';
    
    pos = 0;
    p = "avg_cos=";
    while (*p) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, avg_cosine, EQ_LOG_STRING_LEN - pos);
    p = " min=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, min_cosine, EQ_LOG_STRING_LEN - pos);
    p = " max=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, max_cosine, EQ_LOG_STRING_LEN - pos);
    p = " std=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, std_cosine, EQ_LOG_STRING_LEN - pos);
    
    if (avg_cosine > 0.3f) {
        p = " [ALIGNMENT_COLLAPSE]";
        while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    }
    actual[pos] = '\0';
    
    enqueueEquationLog(d_buffer, d_state,
                      "TOKEN277_ALIGN",
                      "cos(h, W[277]) = (h . W[277]) / (||h|| * ||W[277]||)",
                      inputs, "", expected, actual,
                      batch_idx, -1, step_idx, EquationPhase::TOKEN277_HIDDEN_ALIGNMENT);
}

// ============================================================================
// MSE LOSS EQUATION LOGGING
// Formula: E = (1/2)(y - a₁⁽³⁾)² where a₁⁽³⁾ is the output layer activation
// Used for regression/numeric prediction heads (e.g., NumericHead in GRIM-text)
// ============================================================================

// Log MSE Loss forward computation: E = (1/2)(y - a)²
__device__ __forceinline__ void logMSELossEquation(
    EquationLogEntryDevice* d_buffer,
    EquationLogBufferState* d_state,
    float target_y,             // y (target/label value)
    float activation_a,         // a (output activation)
    float squared_error,        // (y - a)²
    float expected_loss,        // (1/2)(y - a)² computed
    float actual_loss,          // Actual loss value from system
    int sample_idx,             // Index in batch
    int batch_idx,
    int step_idx)
{
    char inputs[EQ_LOG_STRING_LEN];
    char expected[EQ_LOG_STRING_LEN];
    char actual[EQ_LOG_STRING_LEN];
    
    int pos = 0;
    const char* p = "y=";
    while (*p) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, target_y, EQ_LOG_STRING_LEN - pos);
    p = " a=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, activation_a, EQ_LOG_STRING_LEN - pos);
    float residual = target_y - activation_a;
    p = " (y-a)=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, residual, EQ_LOG_STRING_LEN - pos);
    p = " sample=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_itoa_device(inputs + pos, sample_idx, EQ_LOG_STRING_LEN - pos);
    inputs[pos] = '\0';
    
    pos = 0;
    p = "E=(1/2)(y-a)^2=";
    while (*p) expected[pos++] = *p++;
    pos += eq_ftoa_device(expected + pos, expected_loss, EQ_LOG_STRING_LEN - pos);
    p = " sq_err=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) expected[pos++] = *p++;
    pos += eq_ftoa_device(expected + pos, squared_error, EQ_LOG_STRING_LEN - pos);
    expected[pos] = '\0';
    
    pos = 0;
    p = "actual_loss=";
    while (*p) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, actual_loss, EQ_LOG_STRING_LEN - pos);
    float diff = fabsf(actual_loss - expected_loss);
    p = " diff=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, diff, EQ_LOG_STRING_LEN - pos);
    
    if (diff > 0.0001f) {
        p = " [MISMATCH]";
        while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    }
    if (isnan(actual_loss) || isinf(actual_loss)) {
        p = " [NAN/INF]";
        while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    }
    actual[pos] = '\0';
    
    enqueueEquationLog(d_buffer, d_state,
                      "MSE_LOSS_FWD",
                      "E = (1/2)(y - a)^2",
                      inputs, "", expected, actual,
                      batch_idx, -1, step_idx, EquationPhase::MSE_LOSS_FORWARD);
}

// Log MSE Loss backward gradient: dE/da = -(y - a) = (a - y)
__device__ __forceinline__ void logMSELossGradientEquation(
    EquationLogEntryDevice* d_buffer,
    EquationLogBufferState* d_state,
    float target_y,             // y (target value)
    float activation_a,         // a (output activation)
    float expected_grad,        // -(y - a) = (a - y)
    float actual_grad,          // Actual gradient from backward pass
    int sample_idx,
    int batch_idx,
    int step_idx)
{
    char inputs[EQ_LOG_STRING_LEN];
    char expected[EQ_LOG_STRING_LEN];
    char actual[EQ_LOG_STRING_LEN];
    
    int pos = 0;
    const char* p = "y=";
    while (*p) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, target_y, EQ_LOG_STRING_LEN - pos);
    p = " a=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, activation_a, EQ_LOG_STRING_LEN - pos);
    p = " sample=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_itoa_device(inputs + pos, sample_idx, EQ_LOG_STRING_LEN - pos);
    inputs[pos] = '\0';
    
    pos = 0;
    p = "dE/da=-(y-a)=";
    while (*p) expected[pos++] = *p++;
    pos += eq_ftoa_device(expected + pos, expected_grad, EQ_LOG_STRING_LEN - pos);
    expected[pos] = '\0';
    
    pos = 0;
    p = "actual_grad=";
    while (*p) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, actual_grad, EQ_LOG_STRING_LEN - pos);
    float diff = fabsf(actual_grad - expected_grad);
    p = " diff=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, diff, EQ_LOG_STRING_LEN - pos);
    
    if (diff > 0.0001f) {
        p = " [GRAD_MISMATCH]";
        while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    }
    actual[pos] = '\0';
    
    enqueueEquationLog(d_buffer, d_state,
                      "MSE_LOSS_BWD",
                      "dE/da = -(y - a) = (a - y)",
                      inputs, "", expected, actual,
                      batch_idx, -1, step_idx, EquationPhase::MSE_LOSS_BACKWARD);
}

// Tracks: per-position loss contribution when Token 277 is the target
__device__ __forceinline__ void logToken277LossContribution(
    EquationLogEntryDevice* d_buffer,
    EquationLogBufferState* d_state,
    float ce_loss_277,           // CE loss for this position (where 277 is target)
    float focal_weight,          // Focal loss weight (1-p_t)^gamma
    float prob_277,              // Probability assigned to token 277
    float log_prob_277,          // Log probability of token 277
    float grad_277_at_target,    // Actual gradient at this position
    float expected_grad,         // Expected gradient (p - 1 for CE)
    int token_idx,               // Position index in sequence
    int batch_idx,
    int step_idx
) {
    char inputs[EQ_LOG_STRING_LEN];
    char expected_str[EQ_LOG_STRING_LEN];
    char actual[EQ_LOG_STRING_LEN];
    
    int pos = 0;
    const char* p = "pos=";
    while (*p) inputs[pos++] = *p++;
    pos += eq_itoa_device(inputs + pos, token_idx, EQ_LOG_STRING_LEN - pos);
    p = " p(277)=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, prob_277, EQ_LOG_STRING_LEN - pos);
    p = " log_p=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, log_prob_277, EQ_LOG_STRING_LEN - pos);
    inputs[pos] = '\0';
    
    pos = 0;
    p = "grad_exp=";
    while (*p) expected_str[pos++] = *p++;
    pos += eq_ftoa_device(expected_str + pos, expected_grad, EQ_LOG_STRING_LEN - pos);
    p = " CE=-log_p=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) expected_str[pos++] = *p++;
    pos += eq_ftoa_device(expected_str + pos, -log_prob_277, EQ_LOG_STRING_LEN - pos);
    expected_str[pos] = '\0';
    
    pos = 0;
    p = "grad_act=";
    while (*p) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, grad_277_at_target, EQ_LOG_STRING_LEN - pos);
    p = " CE=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, ce_loss_277, EQ_LOG_STRING_LEN - pos);
    p = " fw=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, focal_weight, EQ_LOG_STRING_LEN - pos);
    
    if (grad_277_at_target >= 0.0f) {
        p = " [GRAD_SIGN_ERROR]";
        while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    }
    if (prob_277 > 0.5f) {
        p = " [HIGH_P277]";
        while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    }
    actual[pos] = '\0';
    
    enqueueEquationLog(d_buffer, d_state,
                      "TOKEN277_LOSS",
                      "CE_277 = -log(p(277)) * fw, grad = p(277) - 1",
                      inputs, "", expected_str, actual,
                      batch_idx, token_idx, step_idx, EquationPhase::TOKEN277_LOSS_CONTRIBUTION);
}

// Tracks: batch-level summary statistics for token 277
__device__ __forceinline__ void logToken277BatchSummary(
    EquationLogEntryDevice* d_buffer,
    EquationLogBufferState* d_state,
    float logit_277_mean,       // Mean logit[277] across positions
    float logit_277_std,        // Std dev of logit[277]
    float logit_277_min,        // Min logit[277]
    float logit_277_max,        // Max logit[277]
    int argmax_count_277,       // Positions where 277 is argmax
    int total_positions,        // Total positions in batch
    float avg_hidden_norm,      // Mean ||h|| across positions
    float weight_277_norm,      // Current ||W[277]||
    float avg_cosine_h_w277,    // Mean cos(h, W[277])
    int target_count_277,       // Positions where target=277
    int batch_idx,
    int step_idx)
{
    char inputs[EQ_LOG_STRING_LEN];
    char expected[EQ_LOG_STRING_LEN];
    char actual[EQ_LOG_STRING_LEN];
    
    // Inputs: decomposition components
    int pos = 0;
    const char* p = "||h||_avg=";
    while (*p) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, avg_hidden_norm, EQ_LOG_STRING_LEN - pos);
    p = " ||W277||=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, weight_277_norm, EQ_LOG_STRING_LEN - pos);
    p = " cos_avg=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) inputs[pos++] = *p++;
    pos += eq_ftoa_device(inputs + pos, avg_cosine_h_w277, EQ_LOG_STRING_LEN - pos);
    inputs[pos] = '\0';
    
    // Expected: 277 should be argmax only where it's the target
    pos = 0;
    p = "target_277=";
    while (*p) expected[pos++] = *p++;
    pos += eq_itoa_device(expected + pos, target_count_277, EQ_LOG_STRING_LEN - pos);
    p = "/";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) expected[pos++] = *p++;
    pos += eq_itoa_device(expected + pos, total_positions, EQ_LOG_STRING_LEN - pos);
    expected[pos] = '\0';
    
    // Actual statistics
    pos = 0;
    p = "logit277: mean=";
    while (*p) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, logit_277_mean, EQ_LOG_STRING_LEN - pos);
    p = " std=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    pos += eq_ftoa_device(actual + pos, logit_277_std, EQ_LOG_STRING_LEN - pos);
    p = " argmax_277=";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    pos += eq_itoa_device(actual + pos, argmax_count_277, EQ_LOG_STRING_LEN - pos);
    p = "/";
    while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    pos += eq_itoa_device(actual + pos, total_positions, EQ_LOG_STRING_LEN - pos);
    
    // Mode collapse indicator
    float argmax_rate = static_cast<float>(argmax_count_277) / fmaxf(static_cast<float>(total_positions), 1.0f);
    float target_rate = static_cast<float>(target_count_277) / fmaxf(static_cast<float>(total_positions), 1.0f);
    if (argmax_rate > target_rate + 0.05f) {  // 5% more argmax than expected
        p = " [COLLAPSE]";
        while (*p && pos < EQ_LOG_STRING_LEN - 1) actual[pos++] = *p++;
    }
    actual[pos] = '\0';
    
    enqueueEquationLog(d_buffer, d_state,
                      "TOKEN277_BATCH",
                      "Batch summary: argmax(logits)==277 frequency vs target frequency",
                      inputs, "", expected, actual,
                      batch_idx, -1, step_idx, EquationPhase::TOKEN277_ARGMAX_ANALYSIS);
}

#endif // __CUDACC__

// ============================================================================
// HOST-SIDE EQUATION LOGGER CLASS
// ============================================================================

class EquationLogger {
public:
    // Configuration
    std::string log_file_path_;
    bool enabled_ = false;
    int log_interval_ = 1;           // Log every N steps
    int flush_interval_ms_ = 1000;   // Flush to disk every N ms
    
    // Host-side buffers (pinned memory for async transfer)
    EquationLogEntryDevice* h_buffer_ = nullptr;
    EquationLogEntryDevice* d_buffer_ = nullptr;
    EquationLogBufferState* h_state_ = nullptr;
    EquationLogBufferState* d_state_ = nullptr;
    
    // CUDA resources
    cudaStream_t log_stream_ = nullptr;
    cudaEvent_t flush_event_ = nullptr;
    
    // Phase-ordered queue for host-side batching
    std::vector<EquationLogEntryDevice> pending_entries_;
    std::mutex pending_mutex_;
    
    // File output
    std::ofstream log_file_;
    std::mutex file_mutex_;
    
    // Statistics
    uint64_t total_logged_ = 0;
    uint64_t total_dropped_ = 0;
    
    // Timing
    std::chrono::steady_clock::time_point start_time_;
    
    EquationLogger() = default;
    
    ~EquationLogger() {
        shutdown();
    }
    
    // ========================================================================
    // INITIALIZATION
    // ========================================================================
    
    bool initialize(const std::string& log_path, 
                    bool enable = HyperParameters::DEFAULT_EQ_LOG_ENABLED, 
                    int interval = HyperParameters::DEFAULT_EQ_LOG_INTERVAL,
                    int buffer_size = EQ_LOG_BUFFER_SIZE) {
        if (!enable) {
            enabled_ = false;
            return true;
        }
        
        log_interval_ = interval;
        log_file_path_ = log_path;
        start_time_ = std::chrono::steady_clock::now();
        
        // Allocate pinned host buffer (for async transfers)
        cudaError_t err = cudaMallocHost(&h_buffer_, sizeof(EquationLogEntryDevice) * buffer_size);
        if (err != cudaSuccess) {
            std::cerr << "[EquationLogger] Failed to allocate pinned host buffer: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        memset(h_buffer_, 0, sizeof(EquationLogEntryDevice) * buffer_size);
        
        // Allocate device buffer
        err = cudaMalloc(&d_buffer_, sizeof(EquationLogEntryDevice) * buffer_size);
        if (err != cudaSuccess) {
            std::cerr << "[EquationLogger] Failed to allocate device buffer: " << cudaGetErrorString(err) << std::endl;
            cudaFreeHost(h_buffer_);
            return false;
        }
        cudaMemset(d_buffer_, 0, sizeof(EquationLogEntryDevice) * buffer_size);
        
        // Allocate state structures
        err = cudaMallocHost(&h_state_, sizeof(EquationLogBufferState));
        if (err != cudaSuccess) {
            cudaFree(d_buffer_);
            cudaFreeHost(h_buffer_);
            return false;
        }
        
        err = cudaMalloc(&d_state_, sizeof(EquationLogBufferState));
        if (err != cudaSuccess) {
            cudaFree(d_buffer_);
            cudaFreeHost(h_buffer_);
            cudaFreeHost(h_state_);
            return false;
        }
        
        // Initialize state
        h_state_->write_head = 0;
        h_state_->read_head = 0;
        h_state_->sequence_counter = 0;
        h_state_->entries_dropped = 0;
        h_state_->buffer_size = buffer_size;
        h_state_->flush_threshold = buffer_size / 2;  // Flush at 50% full
        
        cudaMemcpy(d_state_, h_state_, sizeof(EquationLogBufferState), cudaMemcpyHostToDevice);
        
        // Create dedicated stream for logging (non-blocking)
        err = cudaStreamCreateWithFlags(&log_stream_, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            std::cerr << "[EquationLogger] Failed to create log stream: " << cudaGetErrorString(err) << std::endl;
            // Continue without dedicated stream
            log_stream_ = nullptr;
        }
        
        // Create event for timing
        cudaEventCreate(&flush_event_);
        
        // Open log file
        log_file_.open(log_file_path_, std::ios::out | std::ios::app);
        if (!log_file_.is_open()) {
            std::cerr << "[EquationLogger] Failed to open log file: " << log_file_path_ << std::endl;
            // Continue without file output (console only)
        } else {
            // Write header
            log_file_ << "# GRIM-text Equation Log\n";
            log_file_ << "# Format: [EQUATION_TAG] phase | batch | layer | step | formula\n";
            log_file_ << "#         INPUTS: ...\n";
            log_file_ << "#         EXPECTED: ...\n";
            log_file_ << "#         ACTUAL: ...\n";
            log_file_ << "# Started: " << getCurrentTimestamp() << "\n\n";
        }
        
        pending_entries_.reserve(buffer_size);
        enabled_ = true;
        
        std::cout << "[EquationLogger] Initialized with " << buffer_size << " entry buffer" << std::endl;
        return true;
    }
    
    // ========================================================================
    // ASYNC FLUSH - Called periodically from training loop
    // ========================================================================
    
    void flushAsync() {
        if (!enabled_ || !d_buffer_ || !d_state_) return;
        
        cudaStream_t stream = log_stream_ ? log_stream_ : nullptr;
        
        // Async copy state to check how many entries pending
        cudaMemcpyAsync(h_state_, d_state_, sizeof(EquationLogBufferState), 
                        cudaMemcpyDeviceToHost, stream);
        
        // Record event for sync point
        if (flush_event_) {
            cudaEventRecord(flush_event_, stream);
        }
    }
    
    // ========================================================================
    // SYNC FLUSH - Called after flushAsync to process entries
    // ========================================================================
    
    void flushSync() {
        if (!enabled_) return;
        
        // Handle host-side entries FIRST (don't require CUDA state)
        bool has_pending_host_entries = false;
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            has_pending_host_entries = !pending_entries_.empty();
        }
        
        // If we only have host entries (no CUDA buffer initialized), flush them directly
        if (!d_buffer_ || !h_state_) {
            if (has_pending_host_entries) {
                std::lock_guard<std::mutex> lock(pending_mutex_);
                std::sort(pending_entries_.begin(), pending_entries_.end(),
                    [](const EquationLogEntryDevice& a, const EquationLogEntryDevice& b) {
                        if (a.step_idx != b.step_idx) return a.step_idx < b.step_idx;
                        if (a.batch_idx != b.batch_idx) return a.batch_idx < b.batch_idx;
                        return (int)a.phase < (int)b.phase;
                    });
                writeEntriesToFile();
                pending_entries_.clear();
            }
            return;
        }
        
        // Wait for async copy to complete
        if (flush_event_) {
            cudaEventSynchronize(flush_event_);
        } else if (log_stream_) {
            cudaStreamSynchronize(log_stream_);
        }
        
        uint64_t write_head = h_state_->write_head;
        uint64_t read_head = h_state_->read_head;
        int buffer_size = h_state_->buffer_size;
        
        // Even if no CUDA entries, we may have host entries to flush
        if (write_head <= read_head && !has_pending_host_entries) return;  // Nothing to flush
        
        // Calculate how many entries to read
        uint64_t pending = write_head - read_head;
        if (pending > (uint64_t)buffer_size) {
            // Buffer wrapped - we lost some entries
            total_dropped_ += pending - buffer_size;
            read_head = write_head - buffer_size;
        }
        
        int entries_to_read = (int)std::min(pending, (uint64_t)buffer_size);
        
        // Async copy buffer entries
        cudaStream_t stream = log_stream_ ? log_stream_ : nullptr;
        cudaMemcpyAsync(h_buffer_, d_buffer_, sizeof(EquationLogEntryDevice) * buffer_size,
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        // Process entries in phase order
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            
            for (int i = 0; i < entries_to_read; i++) {
                int idx = (int)((read_head + i) % buffer_size);
                if (h_buffer_[idx].valid) {
                    pending_entries_.push_back(h_buffer_[idx]);
                    total_logged_++;
                }
            }
            
            // Sort by step, then phase for ordered output
            std::sort(pending_entries_.begin(), pending_entries_.end(),
                [](const EquationLogEntryDevice& a, const EquationLogEntryDevice& b) {
                    if (a.step_idx != b.step_idx) return a.step_idx < b.step_idx;
                    if (a.batch_idx != b.batch_idx) return a.batch_idx < b.batch_idx;
                    return (int)a.phase < (int)b.phase;
                });
            
            // Write to file
            writeEntriesToFile();
            pending_entries_.clear();
        }
        
        // Update read head on device
        h_state_->read_head = write_head;
        cudaMemcpyAsync(d_state_, h_state_, sizeof(EquationLogBufferState),
                        cudaMemcpyHostToDevice, stream);
        
        // Clear valid flags on device
        cudaMemsetAsync(d_buffer_, 0, sizeof(EquationLogEntryDevice) * buffer_size, stream);
    }
    
    // ========================================================================
    // HOST-SIDE LOGGING (for non-CUDA code paths)
    // ========================================================================
    
    void logHost(
        const std::string& equation_name,
        const std::string& formula,
        const std::string& inputs,
        const std::string& outputs,
        const std::string& expected,
        const std::string& actual,
        int batch_idx,
        int layer_idx,
        int step_idx,
        EquationPhase phase
    ) {
        if (!enabled_) return;
        
        std::lock_guard<std::mutex> lock(pending_mutex_);
        
        EquationLogEntryDevice entry;
        memset(&entry, 0, sizeof(entry));
        
        strncpy(entry.equation_name, equation_name.c_str(), EQ_LOG_STRING_LEN - 1);
        strncpy(entry.formula, formula.c_str(), EQ_LOG_STRING_LEN - 1);
        strncpy(entry.inputs, inputs.c_str(), EQ_LOG_STRING_LEN - 1);
        strncpy(entry.outputs, outputs.c_str(), EQ_LOG_STRING_LEN - 1);
        strncpy(entry.expected, expected.c_str(), EQ_LOG_STRING_LEN - 1);
        strncpy(entry.actual, actual.c_str(), EQ_LOG_STRING_LEN - 1);
        
        entry.batch_idx = batch_idx;
        entry.layer_idx = layer_idx;
        entry.step_idx = step_idx;
        entry.phase = phase;
        entry.sequence_num = total_logged_++;
        entry.valid = 1;
        
        auto now = std::chrono::steady_clock::now();
        entry.timestamp_ms = std::chrono::duration<float, std::milli>(now - start_time_).count();
        
        pending_entries_.push_back(entry);
    }
    
    // ========================================================================
    // GETTERS FOR DEVICE POINTERS (pass to kernels)
    // ========================================================================
    
    EquationLogEntryDevice* getDeviceBuffer() const { return d_buffer_; }
    EquationLogBufferState* getDeviceState() const { return d_state_; }
    bool isEnabled() const { return enabled_; }
    
    // ========================================================================
    // STATISTICS
    // ========================================================================
    
    void printStats() const {
        std::cout << "[EquationLogger] Stats: logged=" << total_logged_ 
                  << " dropped=" << total_dropped_ << std::endl;
    }
    
    // ========================================================================
    // SHUTDOWN
    // ========================================================================
    
    void shutdown() {
        if (!enabled_) return;
        
        // Final flush
        flushAsync();
        flushSync();
        
        // Write summary
        if (log_file_.is_open()) {
            log_file_ << "\n# === END OF LOG ===\n";
            log_file_ << "# Total entries: " << total_logged_ << "\n";
            log_file_ << "# Dropped entries: " << total_dropped_ << "\n";
            log_file_.close();
        }
        
        // Cleanup CUDA resources
        if (flush_event_) { cudaEventDestroy(flush_event_); flush_event_ = nullptr; }
        if (log_stream_) { cudaStreamDestroy(log_stream_); log_stream_ = nullptr; }
        if (d_buffer_) { cudaFree(d_buffer_); d_buffer_ = nullptr; }
        if (d_state_) { cudaFree(d_state_); d_state_ = nullptr; }
        if (h_buffer_) { cudaFreeHost(h_buffer_); h_buffer_ = nullptr; }
        if (h_state_) { cudaFreeHost(h_state_); h_state_ = nullptr; }
        
        enabled_ = false;
        std::cout << "[EquationLogger] Shutdown complete. Logged " << total_logged_ << " entries." << std::endl;
    }

private:
    // ========================================================================
    // FILE OUTPUT
    // ========================================================================
    
    void writeEntriesToFile() {
        if (!log_file_.is_open() || pending_entries_.empty()) return;
        
        std::lock_guard<std::mutex> lock(file_mutex_);
        
        for (const auto& entry : pending_entries_) {
            // Format: [EQUATION_NAME] phase=X batch=Y layer=Z step=W
            log_file_ << "[" << entry.equation_name << "] "
                      << "phase=" << phaseToString(entry.phase)
                      << " batch=" << entry.batch_idx
                      << " layer=" << entry.layer_idx
                      << " step=" << entry.step_idx
                      << " @" << entry.timestamp_ms << "ms\n";
            
            log_file_ << "  FORMULA: " << entry.formula << "\n";
            log_file_ << "  INPUTS: " << entry.inputs << "\n";
            log_file_ << "  EXPECTED: " << entry.expected << "\n";
            log_file_ << "  ACTUAL: " << entry.actual << "\n";
            log_file_ << "\n";
        }
        
        log_file_.flush();
    }
    
    std::string getCurrentTimestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&time));
        return std::string(buf);
    }
};

// ============================================================================
// GLOBAL INSTANCE (optional singleton pattern)
// ============================================================================

inline EquationLogger& getEquationLogger() {
    static EquationLogger instance;
    return instance;
}

// Convenience macro for host-side logging
#define EQ_LOG_HOST(name, formula, inputs, outputs, expected, actual, batch, layer, step, phase) \
    do { \
        if (GRIM::getEquationLogger().isEnabled()) { \
            GRIM::getEquationLogger().logHost(name, formula, inputs, outputs, expected, actual, \
                                               batch, layer, step, phase); \
        } \
    } while(0)

} // namespace GRIM

// ============================================================================
// PYTORCH VERIFICATION INTEGRATION
// ============================================================================
// When GRIM_PYTORCH_VERIFY is defined, enables side-by-side comparison with
// PyTorch reference implementations via subprocess.
// 
// Usage:
//   1. Define GRIM_PYTORCH_VERIFY before including this header
//   2. Call PYTORCH_VERIFY_INIT("path/to/grim/root") at startup
//   3. Use EQ_LOG_AND_VERIFY_* macros instead of plain EQ_LOG_HOST
//   4. Call PYTORCH_VERIFY_SHUTDOWN() at cleanup
// 
// NOTE: This is SLOW (subprocess per verification) - use only for debugging!
// ============================================================================

#include "PyTorchVerify.hpp"

// ============================================================================
// COMBINED LOGGING + VERIFICATION MACROS
// These log the equation AND run PyTorch verification in one call
// ============================================================================

// RMSNorm: Log equation AND verify with PyTorch
#define EQ_LOG_AND_VERIFY_RMSNORM(x_host, gamma_host, output_host, batch_size, hidden_dim, eps, batch_idx, layer_idx, step_idx) \
    do { \
        /* Log equation */ \
        std::ostringstream _inputs, _expected, _actual; \
        float _x_rms = 0.0f, _out_rms = 0.0f; \
        for (int _i = 0; _i < (batch_size) * (hidden_dim); _i++) { _x_rms += (x_host)[_i] * (x_host)[_i]; } \
        _x_rms = std::sqrt(_x_rms / ((batch_size) * (hidden_dim))); \
        for (int _i = 0; _i < (batch_size) * (hidden_dim); _i++) { _out_rms += (output_host)[_i] * (output_host)[_i]; } \
        _out_rms = std::sqrt(_out_rms / ((batch_size) * (hidden_dim))); \
        _inputs << "x: shape=[" << (batch_size) << "," << (hidden_dim) << "] rms=" << _x_rms; \
        _expected << "output_rms ≈ 1.0"; \
        _actual << "output_rms = " << _out_rms; \
        EQ_LOG_HOST("RMSNORM_FWD", "y = x * gamma / sqrt(mean(x²) + eps)", \
                    _inputs.str(), "", _expected.str(), _actual.str(), \
                    batch_idx, layer_idx, step_idx, GRIM::EquationPhase::RMSNORM); \
        /* PyTorch verification */ \
        PYTORCH_VERIFY_RMSNORM(x_host, gamma_host, output_host, batch_size, hidden_dim, eps, batch_idx, layer_idx, step_idx); \
    } while(0)

// MatMul: Log equation AND verify with PyTorch
// transpose_b: If true, B is [N,K] and we compute A @ B^T
#define EQ_LOG_AND_VERIFY_MATMUL(A_host, B_host, C_host, M, K, N, op_name, batch_idx, layer_idx, step_idx, transpose_b) \
    do { \
        std::ostringstream _inputs, _actual; \
        float _a_rms = 0.0f, _c_rms = 0.0f; \
        for (int _i = 0; _i < (M) * (K); _i++) { _a_rms += (A_host)[_i] * (A_host)[_i]; } \
        _a_rms = std::sqrt(_a_rms / ((M) * (K))); \
        for (int _i = 0; _i < (M) * (N); _i++) { _c_rms += (C_host)[_i] * (C_host)[_i]; } \
        _c_rms = std::sqrt(_c_rms / ((M) * (N))); \
        _inputs << "A: [" << (M) << "x" << (K) << "] rms=" << _a_rms; \
        _actual << "C: [" << (M) << "x" << (N) << "] rms=" << _c_rms; \
        EQ_LOG_HOST(op_name, (transpose_b) ? "C = A @ B^T" : "C = A @ B", _inputs.str(), "", "", _actual.str(), \
                    batch_idx, layer_idx, step_idx, GRIM::EquationPhase::QKV_PROJECTION); \
        PYTORCH_VERIFY_MATMUL(A_host, B_host, C_host, M, K, N, op_name, batch_idx, layer_idx, step_idx, transpose_b); \
    } while(0)

// GELU: Log equation AND verify with PyTorch
#define EQ_LOG_AND_VERIFY_GELU(input_host, output_host, n, batch_idx, layer_idx, step_idx) \
    do { \
        std::ostringstream _inputs, _actual; \
        float _in_rms = 0.0f, _out_rms = 0.0f; \
        for (int _i = 0; _i < (n); _i++) { _in_rms += (input_host)[_i] * (input_host)[_i]; } \
        _in_rms = std::sqrt(_in_rms / (n)); \
        for (int _i = 0; _i < (n); _i++) { _out_rms += (output_host)[_i] * (output_host)[_i]; } \
        _out_rms = std::sqrt(_out_rms / (n)); \
        _inputs << "x: n=" << (n) << " rms=" << _in_rms; \
        _actual << "y: rms=" << _out_rms; \
        EQ_LOG_HOST("GELU_FWD", "y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x³)))", \
                    _inputs.str(), "", "", _actual.str(), \
                    batch_idx, layer_idx, step_idx, GRIM::EquationPhase::FFN_GELU); \
        PYTORCH_VERIFY_GELU(input_host, output_host, n, batch_idx, layer_idx, step_idx); \
    } while(0)

// Cross-entropy loss: Log equation AND verify with PyTorch
#define EQ_LOG_AND_VERIFY_CE_LOSS(logits_host, targets_host, loss_host, batch_size, vocab_size, batch_idx, step_idx) \
    do { \
        std::ostringstream _inputs, _actual; \
        _inputs << "logits: [" << (batch_size) << "x" << (vocab_size) << "]"; \
        _actual << "loss = " << *(loss_host); \
        EQ_LOG_HOST("CE_LOSS_FWD", "L = -log(softmax(logits)[target])", \
                    _inputs.str(), "", "", _actual.str(), \
                    batch_idx, -1, step_idx, GRIM::EquationPhase::LOSS_COMPUTATION); \
        PYTORCH_VERIFY_CE_LOSS(logits_host, targets_host, loss_host, batch_size, vocab_size, batch_idx, step_idx); \
    } while(0)

// Scaled dot-product attention: Log equation AND verify with PyTorch
#define EQ_LOG_AND_VERIFY_SDPA(Q_host, K_host, V_host, out_host, b, h, s, d, scale, batch_idx, layer_idx, step_idx) \
    do { \
        std::ostringstream _inputs, _actual; \
        _inputs << "Q,K,V: [" << (b) << "," << (h) << "," << (s) << "," << (d) << "] scale=" << (scale); \
        float _out_rms = 0.0f; \
        size_t _n = (size_t)(b) * (h) * (s) * (d); \
        for (size_t _i = 0; _i < _n; _i++) { _out_rms += (out_host)[_i] * (out_host)[_i]; } \
        _out_rms = std::sqrt(_out_rms / _n); \
        _actual << "output rms=" << _out_rms; \
        EQ_LOG_HOST("SDPA_FWD", "out = softmax(Q@K^T * scale) @ V", \
                    _inputs.str(), "", "", _actual.str(), \
                    batch_idx, layer_idx, step_idx, GRIM::EquationPhase::FLASH_ATTENTION_FWD); \
        PYTORCH_VERIFY_SDPA(Q_host, K_host, V_host, out_host, b, h, s, d, scale, batch_idx, layer_idx, step_idx); \
    } while(0)
