//======================================================//
//  Byte.cu
//  CUDA implementation of byte-level fallback tokenizer
//======================================================//

#include "Byte.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <string>

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
//
// Per-slot output is one byte. Non-byte tokens (specials, atoms, unigram pieces)
// cannot be represented here, so we emit ASCII '?' (0x3F) as a single-byte,
// valid-UTF-8 placeholder. CPU decode() does the same so both paths agree.
__global__ void kernelByteDecode(
    const int* __restrict__ input,
    uint8_t* __restrict__ output,
    size_t count,
    int token_offset
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int token_id = input[idx];
        if (token_id >= token_offset && token_id < token_offset + 256) {
            output[idx] = static_cast<uint8_t>(token_id - token_offset);
        } else {
            output[idx] = static_cast<uint8_t>('?');
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

//======================================================//
//  ByteEncoder Implementation
//======================================================//

ByteEncoder::ByteEncoder() = default;
ByteEncoder::~ByteEncoder() = default;
ByteEncoder::ByteEncoder(ByteEncoder&&) noexcept = default;
ByteEncoder& ByteEncoder::operator=(ByteEncoder&&) noexcept = default;

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
    // Non-byte tokens (specials, atoms, unigram pieces) become ASCII '?' so the
    // output stays valid UTF-8 and matches kernelByteDecode's per-slot behavior.
    std::string result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        int tid = token_ids[i];
        if (tid >= BYTE_TOKEN_OFFSET && tid < BYTE_TOKEN_OFFSET + 256) {
            result.push_back(static_cast<char>(tid - BYTE_TOKEN_OFFSET));
        } else {
            result.push_back('?');
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
    
    // Whitespace specials must come before the printable branch: ' ' is
    // printable per UTF8::isPrintable, and the other whitespace bytes are not
    // printable but reach here as control bytes.
    if (byte_val == ' ') {
        oss << "<SP>";
    } else if (byte_val == '\n') {
        oss << "<LF>";
    } else if (byte_val == '\r') {
        oss << "<CR>";
    } else if (byte_val == '\t') {
        oss << "<TAB>";
    } else if (UTF8::isPrintable(byte_val)) {
        oss << "'" << static_cast<char>(byte_val) << "'";
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

// CONTRACT: `inputs` is a HOST array of `batch_size` device pointers (each
// pointing to a sequence's bytes in device memory). Same for `outputs` and
// `lengths`. The arrays themselves live on the host and are staged into device
// memory below; the data they reference must already be on the device.
//
// This call is ENQUEUE-ONLY: every allocation, copy, kernel launch, and free is
// stream-ordered on `stream`. The function returns as soon as the work is
// queued; the caller is responsible for synchronizing `stream` before consuming
// `outputs`. Errors observed before kernel launch are reported synchronously;
// post-launch errors surface on the next stream synchronization point.
//
// HOST-ARRAY LIFETIME: `inputs`, `lengths`, and `outputs` are read by
// cudaMemcpyAsync below. They MUST remain valid and unchanged until the
// caller synchronizes `stream`. Do not pass stack temporaries that go out of
// scope before the sync, and do not mutate these arrays before the sync. See
// the contract block in Byte.hpp on encodeBatchGPU().
bool ByteEncoder::encodeBatchGPU(const uint8_t* const* inputs,
                                  const size_t* lengths,
                                  int** outputs,
                                  size_t batch_size,
                                  cudaStream_t stream) {
    if (batch_size == 0) return true;

    if (!inputs || !lengths || !outputs) {
        throw std::runtime_error(
            "[ByteEncoder] encodeBatchGPU: inputs/lengths/outputs must be non-null host arrays "
            "(" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")");
    }

    // Stage host arrays of device pointers into device memory using
    // stream-ordered allocation so the staging buffers stay alive exactly as
    // long as the kernel needs them, and are freed in stream order afterwards.
    uint8_t** d_input_ptrs  = nullptr;
    size_t*   d_lengths     = nullptr;
    int**     d_output_ptrs = nullptr;

    auto cleanup_async = [&]() {
        if (d_input_ptrs)  cudaFreeAsync(d_input_ptrs,  stream);
        if (d_lengths)     cudaFreeAsync(d_lengths,     stream);
        if (d_output_ptrs) cudaFreeAsync(d_output_ptrs, stream);
    };

    auto fail = [&](const char* what, cudaError_t e) {
        std::cerr << "[ByteEncoder] encodeBatchGPU " << what << ": "
                  << cudaGetErrorString(e) << std::endl;
        cleanup_async();
        return false;
    };

    cudaError_t err;
    if ((err = cudaMallocAsync(&d_input_ptrs,  batch_size * sizeof(uint8_t*), stream)) != cudaSuccess)
        return fail("cudaMallocAsync(d_input_ptrs)", err);
    if ((err = cudaMallocAsync(&d_lengths,     batch_size * sizeof(size_t),   stream)) != cudaSuccess)
        return fail("cudaMallocAsync(d_lengths)", err);
    if ((err = cudaMallocAsync(&d_output_ptrs, batch_size * sizeof(int*),     stream)) != cudaSuccess)
        return fail("cudaMallocAsync(d_output_ptrs)", err);

    if ((err = cudaMemcpyAsync(d_input_ptrs, inputs, batch_size * sizeof(uint8_t*),
                               cudaMemcpyHostToDevice, stream)) != cudaSuccess)
        return fail("cudaMemcpyAsync(d_input_ptrs)", err);
    if ((err = cudaMemcpyAsync(d_lengths, lengths, batch_size * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream)) != cudaSuccess)
        return fail("cudaMemcpyAsync(d_lengths)", err);
    if ((err = cudaMemcpyAsync(d_output_ptrs, outputs, batch_size * sizeof(int*),
                               cudaMemcpyHostToDevice, stream)) != cudaSuccess)
        return fail("cudaMemcpyAsync(d_output_ptrs)", err);

    // Launch 2D grid: x for sequence position, y for batch
    dim3 threads(256, 1);
    dim3 blocks(64, static_cast<unsigned>(std::min(batch_size, size_t(65535))));

    kernelByteBatchEncode<<<blocks, threads, 0, stream>>>(
        const_cast<const uint8_t* const*>(d_input_ptrs),
        d_lengths,
        d_output_ptrs,
        batch_size,
        BYTE_TOKEN_OFFSET
    );

    // Check launch-time error BEFORE enqueuing frees so a launch failure is
    // reported. Any kernel runtime error will surface on the caller's next
    // synchronization of `stream`.
    err = cudaGetLastError();
    cleanup_async();

    if (err != cudaSuccess) {
        std::cerr << "[ByteEncoder] kernelByteBatchEncode launch failed: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

} // namespace Tokenizer
} // namespace GRIM
