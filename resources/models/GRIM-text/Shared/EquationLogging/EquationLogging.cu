/**
 * EquationLogging.cu - CUDA implementation for equation logging system
 * 
 * This file provides:
 * 1. Device-side global pointers for the log buffer
 * 2. Kernel for initializing device pointers
 * 3. Helper kernels for building log strings on device
 */

#include "EquationLogging.hpp"
#include <cuda_runtime.h>
#include <cstdio>

namespace GRIM {

// ============================================================================
// DEVICE-SIDE GLOBAL POINTERS
// ============================================================================
// These are set by initEquationLoggerDevice() and used by EQ_LOG macros

__device__ EquationLogEntryDevice* g_eq_log_buffer = nullptr;
__device__ EquationLogBufferState* g_eq_log_state = nullptr;

// Host-side copies for passing to kernels
static EquationLogEntryDevice* h_d_buffer_ptr = nullptr;
static EquationLogBufferState* h_d_state_ptr = nullptr;

// ============================================================================
// INITIALIZATION KERNEL
// ============================================================================

__global__ void kernelSetEquationLogPointers(
    EquationLogEntryDevice* buffer,
    EquationLogBufferState* state
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_eq_log_buffer = buffer;
        g_eq_log_state = state;
    }
}

// Host function to initialize device pointers
void initEquationLoggerDevice(EquationLogEntryDevice* d_buffer, EquationLogBufferState* d_state) {
    h_d_buffer_ptr = d_buffer;
    h_d_state_ptr = d_state;
    
    kernelSetEquationLogPointers<<<1, 1>>>(d_buffer, d_state);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[EquationLogger] Failed to set device pointers: %s\n", cudaGetErrorString(err));
    }
}

// ============================================================================
// HELPER KERNEL: Build formatted tensor stats string
// ============================================================================
// Use this to build input/output strings like "shape=[7168,768] min=X max=Y rms=Z"

__device__ void buildTensorStatsString(
    char* output,
    int max_len,
    const char* name,
    int dim0,
    int dim1,
    float min_val,
    float max_val,
    float rms_val
) {
    int pos = 0;
    
    // Name
    const char* p = name;
    while (*p && pos < max_len - 1) output[pos++] = *p++;
    
    // Shape
    const char* shape_prefix = ": shape=[";
    p = shape_prefix;
    while (*p && pos < max_len - 1) output[pos++] = *p++;
    
    // dim0 (simple int to string)
    char buf[16];
    int val = dim0;
    int len = 0;
    if (val == 0) { buf[len++] = '0'; }
    else {
        char temp[16];
        int tlen = 0;
        while (val > 0) { temp[tlen++] = '0' + (val % 10); val /= 10; }
        while (tlen > 0) buf[len++] = temp[--tlen];
    }
    for (int i = 0; i < len && pos < max_len - 1; i++) output[pos++] = buf[i];
    
    output[pos++] = ',';
    
    // dim1
    val = dim1;
    len = 0;
    if (val == 0) { buf[len++] = '0'; }
    else {
        char temp[16];
        int tlen = 0;
        while (val > 0) { temp[tlen++] = '0' + (val % 10); val /= 10; }
        while (tlen > 0) buf[len++] = temp[--tlen];
    }
    for (int i = 0; i < len && pos < max_len - 1; i++) output[pos++] = buf[i];
    
    output[pos++] = ']';
    output[pos++] = ' ';
    
    // min=
    p = "min=";
    while (*p && pos < max_len - 1) output[pos++] = *p++;
    pos += eq_ftoa_device(output + pos, min_val, max_len - pos);
    
    output[pos++] = ' ';
    
    // max=
    p = "max=";
    while (*p && pos < max_len - 1) output[pos++] = *p++;
    pos += eq_ftoa_device(output + pos, max_val, max_len - pos);
    
    output[pos++] = ' ';
    
    // rms=
    p = "rms=";
    while (*p && pos < max_len - 1) output[pos++] = *p++;
    pos += eq_ftoa_device(output + pos, rms_val, max_len - pos);
    
    output[pos] = '\0';
}

// ============================================================================
// CONVENIENCE KERNEL: Log equation with tensor stats computation
// ============================================================================
// This kernel computes tensor stats AND logs in one shot
// Useful for logging without needing to precompute stats

template<int BLOCK_SIZE = 256>
__global__ void kernelLogTensorEquation(
    EquationLogEntryDevice* d_buffer,
    EquationLogBufferState* d_state,
    const float* tensor_data,
    int num_elements,
    const char* equation_name,
    const char* formula,
    const char* tensor_name,
    int dim0,
    int dim1,
    int batch_idx,
    int layer_idx,
    int step_idx,
    EquationPhase phase
) {
    __shared__ float s_min;
    __shared__ float s_max;
    __shared__ float s_sum_sq;
    __shared__ int s_count;
    
    if (threadIdx.x == 0) {
        s_min = FLT_MAX;
        s_max = -FLT_MAX;
        s_sum_sq = 0.0f;
        s_count = 0;
    }
    __syncthreads();
    
    // Parallel reduction for stats
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    float local_sum_sq = 0.0f;
    int local_count = 0;
    
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_elements; i += blockDim.x * gridDim.x) {
        float val = tensor_data[i];
        if (!isnan(val) && !isinf(val)) {
            local_min = fminf(local_min, val);
            local_max = fmaxf(local_max, val);
            local_sum_sq += val * val;
            local_count++;
        }
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_min = fminf(local_min, __shfl_down_sync(0xffffffff, local_min, offset));
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }
    
    // First thread of each warp updates shared memory
    if (threadIdx.x % 32 == 0) {
        atomicMin((int*)&s_min, __float_as_int(local_min));
        atomicMax((int*)&s_max, __float_as_int(local_max));
        atomicAdd(&s_sum_sq, local_sum_sq);
        atomicAdd(&s_count, local_count);
    }
    __syncthreads();
    
    // Thread 0 logs the result
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float rms = sqrtf(s_sum_sq / fmaxf((float)s_count, 1.0f));
        
        char inputs[EQ_LOG_STRING_LEN];
        buildTensorStatsString(inputs, EQ_LOG_STRING_LEN, tensor_name, 
                               dim0, dim1, s_min, s_max, rms);
        
        enqueueEquationLog(d_buffer, d_state,
                          equation_name, formula, inputs, 
                          "", "", "",  // outputs, expected, actual filled by caller
                          batch_idx, layer_idx, step_idx, phase);
    }
}

// ============================================================================
// NOTE: Specialized device logging functions (logRMSNormEquation, logAdamWEquation,
// logMSELossEquation, logMSELossGradientEquation, logFlashAttentionEquation, 
// logToken277* functions) are defined as __device__ __forceinline__ in 
// EquationLogging.hpp to allow cross-TU usage.
// ============================================================================

// ============================================================================
// HOST WRAPPER FUNCTIONS
// ============================================================================

extern "C" void initEquationLoggerDeviceHost(void* d_buffer, void* d_state) {
    initEquationLoggerDevice(
        static_cast<EquationLogEntryDevice*>(d_buffer),
        static_cast<EquationLogBufferState*>(d_state)
    );
}

} // namespace GRIM
  