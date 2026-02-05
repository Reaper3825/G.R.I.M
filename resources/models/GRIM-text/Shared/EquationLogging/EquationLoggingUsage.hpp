/**
 * EquationLoggingUsage.hpp - Example integration patterns for equation logging
 * 
 * This file demonstrates how to use the EquationLogger in various contexts:
 * 1. From CUDA kernels (device-side logging)
 * 2. From host code (host-side logging)
 * 3. Integration with training loop
 */

#pragma once
#include "EquationLogging.hpp"

namespace GRIM {

// ============================================================================
// EXAMPLE 1: Logging from within a CUDA kernel
// ============================================================================
// Pattern: Use thread 0 to log after computation completes

/*
__global__ void myComputeKernel(
    const float* input, float* output, int n,
    // Pass log pointers to kernel
    EquationLogEntryDevice* eq_buffer,
    EquationLogBufferState* eq_state,
    int batch_idx, int layer_idx, int step_idx
) {
    // ... your computation here ...
    
    __shared__ float s_input_rms, s_output_rms;
    
    // Compute stats in parallel (omitted for brevity)
    
    // Thread 0 logs the equation
    if (threadIdx.x == 0 && blockIdx.x == 0 && eq_buffer && eq_state) {
        char inputs[EQ_LOG_STRING_LEN];
        char outputs[EQ_LOG_STRING_LEN];
        
        // Build input stats string
        int pos = 0;
        const char* p = "input: n=";
        while (*p) inputs[pos++] = *p++;
        // ... add integer n ...
        p = " rms=";
        while (*p) inputs[pos++] = *p++;
        pos += eq_ftoa_device(inputs + pos, s_input_rms, EQ_LOG_STRING_LEN - pos);
        inputs[pos] = '\0';
        
        // Build output stats string
        pos = 0;
        p = "output: rms=";
        while (*p) outputs[pos++] = *p++;
        pos += eq_ftoa_device(outputs + pos, s_output_rms, EQ_LOG_STRING_LEN - pos);
        outputs[pos] = '\0';
        
        enqueueEquationLog(
            eq_buffer, eq_state,
            "MY_KERNEL",                           // equation_name
            "output[i] = f(input[i])",             // formula
            inputs,                                 // inputs
            outputs,                                // outputs
            "expected rms ~ 1.0",                  // expected
            outputs,                                // actual (same as outputs)
            batch_idx, layer_idx, step_idx,
            EquationPhase::FFN_GELU                // phase
        );
    }
}
*/

// ============================================================================
// EXAMPLE 2: Using the convenience macros from kernels
// ============================================================================
// Requires global device pointers to be initialized first

/*
// In your kernel:
__global__ void myKernelWithMacro(...) {
    // ... computation ...
    
    EQ_LOG_IF_THREAD0(
        "GELU_FWD",
        "y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))",
        "input: rms=0.5",
        "output: rms=0.3",
        "expected output_rms < input_rms",
        "actual output_rms = 0.3",
        batch_idx, layer_idx, step_idx,
        EquationPhase::FFN_GELU
    );
}
*/

// ============================================================================
// EXAMPLE 3: Host-side logging (for CPU computations or wrappers)
// ============================================================================

/*
void myHostFunction(int batch_idx, int step_idx) {
    // Your computation...
    float computed_value = 3.14f;
    
    EQ_LOG_HOST(
        "CPU_COMPUTATION",
        "result = expensive_computation(input)",
        "input: batch=" + std::to_string(batch_idx),
        "output: value=" + std::to_string(computed_value),
        "expected: value > 0",
        "actual: value=" + std::to_string(computed_value),
        batch_idx, -1, step_idx,
        EquationPhase::LOSS_COMPUTATION
    );
}
*/

// ============================================================================
// EXAMPLE 4: Training loop integration
// ============================================================================

/*
class TrainingLoop {
    EquationLogger& eq_logger_;
    int flush_counter_ = 0;
    
public:
    TrainingLoop() : eq_logger_(getEquationLogger()) {
        // Initialize at start of training
        eq_logger_.initialize("equation_log.txt", true, 4096);
    }
    
    void trainBatch(int batch_idx, int step_idx) {
        // Forward pass
        forwardPass(batch_idx, step_idx);
        
        // Backward pass
        backwardPass(batch_idx, step_idx);
        
        // Optimizer step
        optimizerStep(batch_idx, step_idx);
        
        // Periodic async flush (every 10 batches)
        if (++flush_counter_ % 10 == 0) {
            eq_logger_.flushAsync();
        }
        
        // Sync flush less frequently (every 100 batches)
        if (flush_counter_ % 100 == 0) {
            eq_logger_.flushSync();
        }
    }
    
    void forwardPass(int batch_idx, int step_idx) {
        // Pass device pointers to kernels that need logging
        EquationLogEntryDevice* d_buf = eq_logger_.getDeviceBuffer();
        EquationLogBufferState* d_state = eq_logger_.getDeviceState();
        
        // Example: RMSNorm kernel call
        kernelRMSNorm<<<blocks, threads>>>(
            input, output, gamma, n, d_model,
            d_buf, d_state,  // Pass logging pointers
            batch_idx, 0, step_idx  // layer_idx = 0 for first layer
        );
    }
    
    void shutdown() {
        eq_logger_.shutdown();
    }
};
*/

// ============================================================================
// EXAMPLE 5: Inline helper for building common equation strings
// ============================================================================

class EquationStringBuilder {
public:
    // Build tensor stats string on host
    static std::string tensorStats(const std::string& name, int dim0, int dim1, 
                                    float min_val, float max_val, float rms_val) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s: shape=[%d,%d] min=%.6f max=%.6f rms=%.6f",
                 name.c_str(), dim0, dim1, min_val, max_val, rms_val);
        return std::string(buf);
    }
    
    // Build expected vs actual comparison
    static std::string comparison(const std::string& metric, float expected, float actual) {
        char buf[128];
        snprintf(buf, sizeof(buf), "%s: expected=%.6f actual=%.6f ratio=%.4f",
                 metric.c_str(), expected, actual, actual / (expected + 1e-10f));
        return std::string(buf);
    }
    
};

// ============================================================================
// EXAMPLE 6: Phase-ordered batch logging
// ============================================================================
// This shows how entries are automatically sorted by phase when flushed

/*
    // In your forward pass, log in any order:
    logRMSNorm(batch, layer=0, step);    // Phase: RMSNORM_PRE_ATTN
    logQKV(batch, layer=0, step);        // Phase: QKV_PROJECTION
    logAttention(batch, layer=0, step);  // Phase: FLASH_ATTENTION_FWD
    logRMSNorm(batch, layer=0, step);    // Phase: RMSNORM_PRE_FFN
    logFFN(batch, layer=0, step);        // Phase: FFN_LAYER1, FFN_GELU, FFN_LAYER2
    
    // When flushSync() is called, entries are sorted by:
    // 1. step_idx
    // 2. batch_idx
    // 3. phase (enum order)
    
    // So output will be ordered:
    // RMSNORM_PRE_ATTN (phase 2)
    // QKV_PROJECTION (phase 3)
    // FLASH_ATTENTION_FWD (phase 6)
    // RMSNORM_PRE_FFN (phase 8)
    // FFN_LAYER1 (phase 9)
    // FFN_GELU (phase 10)
    // FFN_LAYER2 (phase 11)
*/

} // namespace GRIM
