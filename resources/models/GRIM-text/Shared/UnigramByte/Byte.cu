//======================================================//
//  Byte.cu
//  CUDA implementation of byte-level fallback tokenizer
//======================================================//

#include "Byte.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  CUDA Kernels
//======================================================//

// Kernel: Encode bytes to token IDs
__global__ void kernelByteEncode(
    const uint8_t* __restrict__ input,
    int* __restrict__ output,
    size_t length,
    int token_offset
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        // Direct mapping: byte value + offset = token ID
        output[idx] = static_cast<int>(input[idx]) + token_offset;
    }
}

// Kernel: Decode token IDs to bytes
__global__ void kernelByteDecode(
    const int* __restrict__ input,
    uint8_t* __restrict__ output,
    size_t count,
    int token_offset
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int token_id = input[idx];
        // Only decode valid byte tokens
        if (token_id >= token_offset && token_id < token_offset + 256) {
            output[idx] = static_cast<uint8_t>(token_id - token_offset);
        } else {
            output[idx] = 0xEF;  // Replacement character (first byte of U+FFFD in UTF-8)
        }
    }
}

// Kernel: Batch encode (single kernel handles all sequences)
__global__ void kernelByteBatchEncode(
    const uint8_t* const* __restrict__ inputs,
    const size_t* __restrict__ lengths,
    int** __restrict__ outputs,
    size_t batch_size,
    int token_offset
) {
    // Grid-stride loop over batch
    for (size_t batch_idx = blockIdx.y; batch_idx < batch_size; batch_idx += gridDim.y) {
        const uint8_t* input = inputs[batch_idx];
        int* output = outputs[batch_idx];
        size_t length = lengths[batch_idx];
        
        // Thread processes one element within sequence
        for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
             idx < length; 
             idx += blockDim.x * gridDim.x) {
            output[idx] = static_cast<int>(input[idx]) + token_offset;
        }
    }
}

// Kernel: Initialize lookup tables
__global__ void kernelInitByteTables(
    uint8_t* __restrict__ token_to_byte,
    int* __restrict__ byte_to_token,
    bool* __restrict__ is_continuation,
    int token_offset
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 256) {
        uint8_t byte_val = static_cast<uint8_t>(idx);
        
        token_to_byte[idx] = byte_val;
        byte_to_token[idx] = idx + token_offset;
        is_continuation[idx] = ((byte_val & 0xC0) == 0x80);  // 10xxxxxx
    }
}

//======================================================//
//  ByteEncoder Implementation
//======================================================//

ByteEncoder::ByteEncoder()
    : d_token_to_byte_(nullptr)
    , d_byte_to_token_(nullptr)
    , d_is_continuation_(nullptr)
    , gpu_initialized_(false) 
{
}

ByteEncoder::~ByteEncoder() {
    releaseGPU();
}

ByteEncoder::ByteEncoder(ByteEncoder&& other) noexcept
    : d_token_to_byte_(other.d_token_to_byte_)
    , d_byte_to_token_(other.d_byte_to_token_)
    , d_is_continuation_(other.d_is_continuation_)
    , gpu_initialized_(other.gpu_initialized_)
{
    other.d_token_to_byte_ = nullptr;
    other.d_byte_to_token_ = nullptr;
    other.d_is_continuation_ = nullptr;
    other.gpu_initialized_ = false;
}

ByteEncoder& ByteEncoder::operator=(ByteEncoder&& other) noexcept {
    if (this != &other) {
        releaseGPU();
        d_token_to_byte_ = other.d_token_to_byte_;
        d_byte_to_token_ = other.d_byte_to_token_;
        d_is_continuation_ = other.d_is_continuation_;
        gpu_initialized_ = other.gpu_initialized_;
        
        other.d_token_to_byte_ = nullptr;
        other.d_byte_to_token_ = nullptr;
        other.d_is_continuation_ = nullptr;
        other.gpu_initialized_ = false;
    }
    return *this;
}

void ByteEncoder::initGPU() {
    if (gpu_initialized_) return;
    
    cudaError_t err;
    
    // Allocate lookup tables
    err = cudaMalloc(&d_token_to_byte_, 256 * sizeof(uint8_t));
    if (err != cudaSuccess) {
        std::cerr << "[ByteEncoder] Failed to allocate token_to_byte: " 
                  << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMalloc(&d_byte_to_token_, 256 * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_token_to_byte_);
        d_token_to_byte_ = nullptr;
        std::cerr << "[ByteEncoder] Failed to allocate byte_to_token: "
                  << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMalloc(&d_is_continuation_, 256 * sizeof(bool));
    if (err != cudaSuccess) {
        cudaFree(d_token_to_byte_);
        cudaFree(d_byte_to_token_);
        d_token_to_byte_ = nullptr;
        d_byte_to_token_ = nullptr;
        std::cerr << "[ByteEncoder] Failed to allocate is_continuation: "
                  << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // INTENTIONAL: Stream 0 used for one-time initialization (cudaDeviceSynchronize follows at line 180)
    // Initialize tables with kernel
    kernelInitByteTables<<<1, 256, 0, 0>>>(
        d_token_to_byte_,
        d_byte_to_token_,
        d_is_continuation_,
        BYTE_TOKEN_OFFSET
    );
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "[ByteEncoder] Failed to initialize tables: "
                  << cudaGetErrorString(err) << std::endl;
        releaseGPU();
        return;
    }
    
    gpu_initialized_ = true;
}

void ByteEncoder::releaseGPU() {
    if (d_token_to_byte_) {
        cudaFree(d_token_to_byte_);
        d_token_to_byte_ = nullptr;
    }
    if (d_byte_to_token_) {
        cudaFree(d_byte_to_token_);
        d_byte_to_token_ = nullptr;
    }
    if (d_is_continuation_) {
        cudaFree(d_is_continuation_);
        d_is_continuation_ = nullptr;
    }
    gpu_initialized_ = false;
}

//--------------------------------------------------//
// CPU Interface
//--------------------------------------------------//

std::vector<int> ByteEncoder::encode(const std::string& text) const {
    return encode(reinterpret_cast<const uint8_t*>(text.data()), text.size());
}

std::vector<int> ByteEncoder::encode(const uint8_t* data, size_t length) const {
    std::vector<int> result(length);
    for (size_t i = 0; i < length; ++i) {
        result[i] = static_cast<int>(data[i]) + BYTE_TOKEN_OFFSET;
    }
    return result;
}

std::string ByteEncoder::decode(const std::vector<int>& token_ids) const {
    return decode(token_ids.data(), token_ids.size());
}

std::string ByteEncoder::decode(const int* token_ids, size_t count) const {
    std::string result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        int tid = token_ids[i];
        if (tid >= BYTE_TOKEN_OFFSET && tid < BYTE_TOKEN_OFFSET + 256) {
            result.push_back(static_cast<char>(tid - BYTE_TOKEN_OFFSET));
        }
    }
    return result;
}

bool ByteEncoder::isByteToken(int token_id) const {
    return token_id >= BYTE_TOKEN_OFFSET && token_id < BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE;
}

uint8_t ByteEncoder::tokenToByte(int token_id) const {
    if (!isByteToken(token_id)) return 0;
    return static_cast<uint8_t>(token_id - BYTE_TOKEN_OFFSET);
}

int ByteEncoder::byteToToken(uint8_t byte_value) const {
    return static_cast<int>(byte_value) + BYTE_TOKEN_OFFSET;
}

ByteToken ByteEncoder::getByteInfo(uint8_t byte_value) const {
    ByteToken info;
    info.byte_value = byte_value;
    info.token_id = byteToToken(byte_value);
    info.is_continuation = UTF8::isContinuation(byte_value);
    info.is_printable = UTF8::isPrintable(byte_value);
    return info;
}

std::string ByteEncoder::tokenToString(int token_id) const {
    if (!isByteToken(token_id)) {
        return "<non-byte>";
    }
    
    uint8_t byte_val = tokenToByte(token_id);
    std::ostringstream oss;
    
    if (UTF8::isPrintable(byte_val)) {
        oss << "'" << static_cast<char>(byte_val) << "'";
    } else if (byte_val == ' ') {
        oss << "<SP>";
    } else if (byte_val == '\n') {
        oss << "<LF>";
    } else if (byte_val == '\r') {
        oss << "<CR>";
    } else if (byte_val == '\t') {
        oss << "<TAB>";
    } else {
        oss << "<0x" << std::hex << std::setw(2) << std::setfill('0') 
            << static_cast<int>(byte_val) << ">";
    }
    
    return oss.str();
}

//--------------------------------------------------//
// GPU Interface
//--------------------------------------------------//

bool ByteEncoder::encodeGPU(const uint8_t* d_input, 
                            int* d_output, 
                            size_t length,
                            cudaStream_t stream) {
    if (length == 0) return true;
    
    const int threads = 256;
    const int blocks = (length + threads - 1) / threads;
    
    kernelByteEncode<<<blocks, threads, 0, stream>>>(
        d_input, d_output, length, BYTE_TOKEN_OFFSET
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ByteEncoder] GPU encode failed: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

bool ByteEncoder::decodeGPU(const int* d_input,
                            uint8_t* d_output,
                            size_t count,
                            cudaStream_t stream) {
    if (count == 0) return true;
    
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    
    kernelByteDecode<<<blocks, threads, 0, stream>>>(
        d_input, d_output, count, BYTE_TOKEN_OFFSET
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ByteEncoder] GPU decode failed: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

bool ByteEncoder::encodeBatchGPU(const uint8_t* const* d_inputs,
                                  const size_t* lengths,
                                  int** d_outputs,
                                  size_t batch_size,
                                  cudaStream_t stream) {
    if (batch_size == 0) return true;
    
    // Copy pointers to device
    uint8_t** d_input_ptrs;
    size_t* d_lengths;
    int** d_output_ptrs;
    
    cudaMalloc(&d_input_ptrs, batch_size * sizeof(uint8_t*));
    cudaMalloc(&d_lengths, batch_size * sizeof(size_t));
    cudaMalloc(&d_output_ptrs, batch_size * sizeof(int*));
    
    cudaMemcpyAsync(d_input_ptrs, d_inputs, batch_size * sizeof(uint8_t*), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_lengths, lengths, batch_size * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_output_ptrs, d_outputs, batch_size * sizeof(int*),
                    cudaMemcpyHostToDevice, stream);
    
    // Launch 2D grid: x for sequence position, y for batch
    dim3 threads(256, 1);
    dim3 blocks(64, std::min(batch_size, size_t(65535)));
    
    kernelByteBatchEncode<<<blocks, threads, 0, stream>>>(
        const_cast<const uint8_t* const*>(d_input_ptrs),
        d_lengths,
        d_output_ptrs,
        batch_size,
        BYTE_TOKEN_OFFSET
    );
    
    cudaError_t err = cudaGetLastError();
    
    cudaFree(d_input_ptrs);
    cudaFree(d_lengths);
    cudaFree(d_output_ptrs);
    
    if (err != cudaSuccess) {
        std::cerr << "[ByteEncoder] Batch encode failed: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

} // namespace Tokenizer
} // namespace GRIM
