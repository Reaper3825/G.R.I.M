//======================================================//
//  UnigramViterbi.cu
//  CUDA-backed RAII Viterbi segmentation for UnigramLM
//
//  Owns per-run launch validation and path materialization.
//  Durable device buffers stay in UnigramGpuMemory;
//  learned vocab/trie state stays in UnigramLM.
//======================================================//

#include "UnigramViterbi.hpp"
#include "UnigramGpuMemory.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <climits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace GRIM {
namespace Tokenizer {

namespace {

static void requireCallerLabel(const char* caller) {
    if (caller == nullptr) {
        throw std::runtime_error("UnigramViterbiSession requires a non-null caller label at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (caller[0] == '\0') {
        throw std::runtime_error("UnigramViterbiSession requires a non-empty caller label at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}

__device__ static bool initializeCudaErrorCode(int* error_code) {
    if (error_code == nullptr) {
        asm("trap;");
        return false;
    }
    *error_code = kUnigramViterbiCudaOk;
    return true;
}

__device__ static void setCudaErrorCode(int* error_code, int code) {
    *error_code = code;
}

static void requireCudaSuccess(cudaError_t err, const char* label, const char* caller) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(caller) + ": " + label + " failed: " +
                                 cudaGetErrorString(err) + " at " + std::string(__FILE__) +
                                 ":" + std::to_string(__LINE__));
    }
}

static void requireCudaWorkspacePointer(const void* ptr, const char* label, const char* caller) {
    if (ptr == nullptr) {
        throw std::runtime_error(std::string(caller) + ": CUDA Viterbi workspace pointer " +
                                 label + " is NULL; caller MUST initialize tokenizer GPU state before encoding at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}

static void requireCudaKernelOk(UnigramGpuMemory& gpu, const char* label, const char* caller) {
    int host_error_code = -1;
    requireCudaSuccess(cudaMemcpy(&host_error_code,
                                  gpu.d_viterbi_error_code,
                                  sizeof(int),
                                  cudaMemcpyDeviceToHost),
                       label,
                       caller);
    if (host_error_code != kUnigramViterbiCudaOk) {
        throw std::runtime_error(std::string(caller) + ": " + label +
                                 " reported error_code=" + std::to_string(host_error_code) +
                                 " (" + unigramViterbiCudaErrorName(host_error_code) + ") at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}

} // namespace

//======================================================//
//  CUDA Kernels
//======================================================//

__global__ void kernelViterbiForward(
    const char* __restrict__ text,
    size_t length,
    const int* __restrict__ trie_children,    // [num_nodes * 256]
    const int* __restrict__ trie_token_ids,   // [num_nodes]
    const float* __restrict__ trie_scores,    // [num_nodes]
    int num_trie_nodes,
    float* __restrict__ viterbi_scores,       // [length + 1]
    int* __restrict__ viterbi_prev,           // [length + 1]
    int* __restrict__ viterbi_tokens,         // [length + 1]
    bool* __restrict__ selected_fallback,     // [length], cleared here; selected path is marked by backtrack
    int unk_id,
    bool enable_byte_fallback,
    int* __restrict__ error_code
) {
    // Single thread processes positions SEQUENTIALLY to maintain Viterbi invariants.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (!initializeCudaErrorCode(error_code)) return;
    if (text == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullText); return; }
    if (trie_children == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullTrieChildren); return; }
    if (trie_token_ids == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullTrieTokenIds); return; }
    if (trie_scores == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullTrieScores); return; }
    if (viterbi_scores == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullViterbiScores); return; }
    if (viterbi_prev == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullViterbiPrev); return; }
    if (viterbi_tokens == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullViterbiTokens); return; }
    if (num_trie_nodes <= 0) { setCudaErrorCode(error_code, kUnigramViterbiCudaEmptyTrie); return; }
    
    // Initialize all DP states before forward relaxation.
    for (size_t i = 0; i <= length; ++i) {
        viterbi_scores[i] = -1e30f;
        viterbi_prev[i] = -1;
        viterbi_tokens[i] = unk_id;
    }
    viterbi_scores[0] = 0.0f;
    viterbi_tokens[0] = -1;

    if (selected_fallback != nullptr) {
        for (size_t i = 0; i < length; ++i) {
            selected_fallback[i] = false;
        }
    }
    
    // Viterbi pass: from each reachable start position, walk the uploaded
    // forward trie over text[pos], text[pos + 1], ... and relax end states.
    for (size_t pos = 0; pos < length; ++pos) {
        if (viterbi_scores[pos] < -1e20f) continue;
        
        unsigned char cur_byte = static_cast<unsigned char>(text[pos]);

        int node = 0;
        for (size_t len = 1; len <= MAX_PIECE_LENGTH && pos + len <= length; ++len) {
            unsigned char c = static_cast<unsigned char>(text[pos + len - 1]);
            
            int child = trie_children[node * 256 + c];
            if (child < 0) break;
            if (child >= num_trie_nodes) { setCudaErrorCode(error_code, kUnigramViterbiCudaTrieChildOutOfRange); return; }
            node = child;
            
            int token_id = trie_token_ids[node];
            if (token_id >= 0) {
                float score = viterbi_scores[pos] + trie_scores[node];
                
                if (score > viterbi_scores[pos + len]) {
                    viterbi_scores[pos + len] = score;
                    viterbi_prev[pos + len] = static_cast<int>(pos);
                    viterbi_tokens[pos + len] = token_id;
                }
            }
        }
        
        float fallback_score = viterbi_scores[pos] + UNKNOWN_SCORE;
        if (fallback_score > viterbi_scores[pos + 1]) {
            viterbi_scores[pos + 1] = fallback_score;
            viterbi_prev[pos + 1] = static_cast<int>(pos);

            if (enable_byte_fallback) {
                viterbi_tokens[pos + 1] = static_cast<int>(cur_byte) + BYTE_TOKEN_OFFSET;
            } else {
                viterbi_tokens[pos + 1] = unk_id;
            }
        }
    }
}

__global__ void kernelViterbiBacktrack(
    size_t length,
    const int* __restrict__ viterbi_prev,
    const int* __restrict__ viterbi_tokens,
    int* __restrict__ output_tokens,
    int* __restrict__ output_count,
    int max_tokens,
    bool* __restrict__ selected_fallback,
    int* __restrict__ error_code
) {
    // Single thread does backtracking.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (!initializeCudaErrorCode(error_code)) return;
    if (viterbi_prev == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullViterbiPrev); return; }
    if (viterbi_tokens == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullViterbiTokens); return; }
    if (output_tokens == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullOutputTokens); return; }
    if (output_count == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullOutputCount); return; }
    if (max_tokens <= 0) { setCudaErrorCode(error_code, kUnigramViterbiCudaInvalidMaxTokens); return; }
    if (length > static_cast<size_t>(INT_MAX)) { setCudaErrorCode(error_code, kUnigramViterbiCudaBacktrackLengthTooLarge); return; }

    *output_count = 0;
    
    if (selected_fallback != nullptr) {
        for (size_t i = 0; i < length; ++i) {
            selected_fallback[i] = false;
        }
    }
    
    int count = 0;
    int pos = static_cast<int>(length);
    int safety_counter = 0;
    while (pos > 0) {
        ++safety_counter;
        if (safety_counter > static_cast<int>(length)) { setCudaErrorCode(error_code, kUnigramViterbiCudaBacktrackSafetyLimit); return; }
        int prev_pos = viterbi_prev[pos];
        if (prev_pos < 0 || prev_pos >= pos) { setCudaErrorCode(error_code, kUnigramViterbiCudaInvalidBackpointer); return; }
        count++;
        if (count > max_tokens) { setCudaErrorCode(error_code, kUnigramViterbiCudaOutputBufferTooSmall); return; }
        pos = prev_pos;
    }
    
    *output_count = count;
    pos = static_cast<int>(length);
    int write_idx = count - 1;
    safety_counter = 0;
    while (pos > 0) {
        ++safety_counter;
        if (safety_counter > static_cast<int>(length)) { setCudaErrorCode(error_code, kUnigramViterbiCudaBacktrackSafetyLimit); return; }
        if (write_idx < 0) { setCudaErrorCode(error_code, kUnigramViterbiCudaOutputBufferTooSmall); return; }
        int prev_pos = viterbi_prev[pos];
        if (prev_pos < 0 || prev_pos >= pos) { setCudaErrorCode(error_code, kUnigramViterbiCudaInvalidBackpointer); return; }
        int token_id = viterbi_tokens[pos];

        if (selected_fallback != nullptr &&
            token_id >= BYTE_TOKEN_OFFSET && token_id < BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE) {
            if (prev_pos != pos - 1) { setCudaErrorCode(error_code, kUnigramViterbiCudaByteFallbackSpanInvalid); return; }
            selected_fallback[prev_pos] = true;
        }

        output_tokens[write_idx] = token_id;
        pos = prev_pos;
        write_idx--;
    }
}

__global__ void kernelTrieLookup(
    const char* __restrict__ text,
    size_t length,
    size_t start_pos,
    const int* __restrict__ trie_children,
    const int* __restrict__ trie_token_ids,
    const float* __restrict__ trie_scores,
    int num_trie_nodes,
    int* __restrict__ match_token,
    int* __restrict__ match_length,
    float* __restrict__ match_score,
    int* __restrict__ error_code
) {
    // Single thread traverses trie from start_pos.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (!initializeCudaErrorCode(error_code)) return;
    if (text == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullText); return; }
    if (trie_children == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullTrieChildren); return; }
    if (trie_token_ids == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullTrieTokenIds); return; }
    if (trie_scores == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullTrieScores); return; }
    if (match_token == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullMatchToken); return; }
    if (match_length == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullMatchLength); return; }
    if (match_score == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullMatchScore); return; }
    if (num_trie_nodes <= 0) { setCudaErrorCode(error_code, kUnigramViterbiCudaEmptyTrie); return; }
    
    int node = 0;
    int best_token = -1;
    int best_length = 0;
    float best_score = -1e30f;
    
    for (size_t i = start_pos; i < length && (i - start_pos) < MAX_PIECE_LENGTH; ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        int child = trie_children[node * 256 + c];
        
        if (child < 0) break;
        if (child >= num_trie_nodes) { setCudaErrorCode(error_code, kUnigramViterbiCudaTrieChildOutOfRange); return; }
        node = child;
        
        int token_id = trie_token_ids[node];
        if (token_id >= 0) {
            float score = trie_scores[node];
            if (score > best_score) {
                best_token = token_id;
                best_length = static_cast<int>(i - start_pos + 1);
                best_score = score;
            }
        }
    }
    
    *match_token = best_token;
    *match_length = best_length;
    *match_score = best_score;
}

//======================================================//
//  UnigramViterbiSession Implementation
//======================================================//

UnigramViterbiSession::UnigramViterbiSession(const UnigramLM& model,
                                             const std::string& normalized_text,
                                             const char* caller) {
    requireCallerLabel(caller);
    CudaResult result = runCuda(model, normalized_text, caller);
    path_score_ = result.path_score;
    tokens_ = std::move(result.tokens);
}

UnigramViterbiSession::CudaResult UnigramViterbiSession::runCuda(
    const UnigramLM& model,
    const std::string& normalized_text,
    const char* caller) {
    requireCallerLabel(caller);

    CudaResult result;
    const size_t n = normalized_text.size();
    if (n == 0) {
        return result;
    }
    if (n > static_cast<size_t>(INT_MAX)) {
        throw std::runtime_error(std::string(caller) +
                                 ": normalized text length exceeds CUDA Viterbi int backtrack limit: length=" +
                                 std::to_string(n) + " at " + std::string(__FILE__) + ":" +
                                 std::to_string(__LINE__));
    }
    if (model.trie_.empty()) {
        throw std::runtime_error(std::string(caller) +
                                 ": trie_ is empty; caller MUST call UnigramLM::buildTrie() before CUDA Viterbi segmentation at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (!model.gpu_) {
        throw std::runtime_error(std::string(caller) +
                                 ": UnigramLM.gpu_ is NULL; object was moved from or not constructed at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    UnigramGpuMemory& gpu = *model.gpu_;
    std::lock_guard<std::mutex> lock(gpu.viterbi_workspace_mutex);

    if (!gpu.initialized) {
        throw std::runtime_error(std::string(caller) +
                                 ": CUDA Viterbi requested before UnigramLM::initGPU(); production tokenization requires uploaded trie state at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (gpu.uploaded_trie_generation != model.trie_generation_) {
        throw std::runtime_error(std::string(caller) +
                                 ": CUDA Viterbi trie upload is stale: uploaded_generation=" +
                                 std::to_string(gpu.uploaded_trie_generation) +
                                 ", live_generation=" + std::to_string(model.trie_generation_) +
                                 "; caller MUST call UnigramLM::initGPU() after buildTrie()/score mutation at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (n > gpu.workspace_max_length) {
        throw std::runtime_error(std::string(caller) +
                                 ": normalized text length=" + std::to_string(n) +
                                 " exceeds CUDA Viterbi workspace_max_length=" +
                                 std::to_string(gpu.workspace_max_length) + " at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    requireCudaWorkspacePointer(gpu.d_viterbi_text, "d_viterbi_text", caller);
    requireCudaWorkspacePointer(gpu.d_trie_children, "d_trie_children", caller);
    requireCudaWorkspacePointer(gpu.d_trie_token_ids, "d_trie_token_ids", caller);
    requireCudaWorkspacePointer(gpu.d_trie_scores, "d_trie_scores", caller);
    requireCudaWorkspacePointer(gpu.d_viterbi_scores, "d_viterbi_scores", caller);
    requireCudaWorkspacePointer(gpu.d_viterbi_prev, "d_viterbi_prev", caller);
    requireCudaWorkspacePointer(gpu.d_viterbi_tokens, "d_viterbi_tokens", caller);
    requireCudaWorkspacePointer(gpu.d_viterbi_output_tokens, "d_viterbi_output_tokens", caller);
    requireCudaWorkspacePointer(gpu.d_viterbi_output_count, "d_viterbi_output_count", caller);
    requireCudaWorkspacePointer(gpu.d_viterbi_selected_fallback, "d_viterbi_selected_fallback", caller);
    requireCudaWorkspacePointer(gpu.d_viterbi_error_code, "d_viterbi_error_code", caller);
    if (gpu.num_nodes <= 0) {
        throw std::runtime_error(std::string(caller) +
                                 ": CUDA Viterbi num_nodes is invalid: " +
                                 std::to_string(gpu.num_nodes) + " at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    requireCudaSuccess(cudaMemcpy(gpu.d_viterbi_text,
                                  normalized_text.data(),
                                  n,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy d_viterbi_text",
                       caller);

    size_t kernel_length = n;
    int kernel_unk_id = UNK_TOKEN_ID;
    bool kernel_enable_byte_fallback = model.enable_byte_fallback_;
    void* forward_args[] = {
        &gpu.d_viterbi_text,
        &kernel_length,
        &gpu.d_trie_children,
        &gpu.d_trie_token_ids,
        &gpu.d_trie_scores,
        &gpu.num_nodes,
        &gpu.d_viterbi_scores,
        &gpu.d_viterbi_prev,
        &gpu.d_viterbi_tokens,
        &gpu.d_viterbi_selected_fallback,
        &kernel_unk_id,
        &kernel_enable_byte_fallback,
        &gpu.d_viterbi_error_code
    };
    requireCudaSuccess(cudaLaunchKernel(reinterpret_cast<const void*>(&kernelViterbiForward),
                                        dim3(1),
                                        dim3(1),
                                        forward_args,
                                        0,
                                        nullptr),
                       "kernelViterbiForward launch",
                       caller);
    requireCudaSuccess(cudaDeviceSynchronize(), "kernelViterbiForward sync", caller);
    requireCudaKernelOk(gpu, "kernelViterbiForward status", caller);

    requireCudaSuccess(cudaMemcpy(&result.path_score,
                                  gpu.d_viterbi_scores + n,
                                  sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy CUDA Viterbi path score",
                       caller);

    int max_tokens = static_cast<int>(n);
    size_t backtrack_length = n;
    void* backtrack_args[] = {
        &backtrack_length,
        &gpu.d_viterbi_prev,
        &gpu.d_viterbi_tokens,
        &gpu.d_viterbi_output_tokens,
        &gpu.d_viterbi_output_count,
        &max_tokens,
        &gpu.d_viterbi_selected_fallback,
        &gpu.d_viterbi_error_code
    };
    requireCudaSuccess(cudaLaunchKernel(reinterpret_cast<const void*>(&kernelViterbiBacktrack),
                                        dim3(1),
                                        dim3(1),
                                        backtrack_args,
                                        0,
                                        nullptr),
                       "kernelViterbiBacktrack launch",
                       caller);
    requireCudaSuccess(cudaDeviceSynchronize(), "kernelViterbiBacktrack sync", caller);
    requireCudaKernelOk(gpu, "kernelViterbiBacktrack status", caller);

    int output_count = -1;
    requireCudaSuccess(cudaMemcpy(&output_count,
                                  gpu.d_viterbi_output_count,
                                  sizeof(int),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy CUDA Viterbi output count",
                       caller);
    if (output_count < 0 || output_count > max_tokens) {
        throw std::runtime_error(std::string(caller) +
                                 ": CUDA Viterbi output_count=" + std::to_string(output_count) +
                                 " is outside [0," + std::to_string(max_tokens) + "] at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    result.tokens.resize(static_cast<size_t>(output_count));
    if (output_count > 0) {
        requireCudaSuccess(cudaMemcpy(result.tokens.data(),
                                      gpu.d_viterbi_output_tokens,
                                      static_cast<size_t>(output_count) * sizeof(int),
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy CUDA Viterbi output tokens",
                           caller);
    }
    return result;
}

} // namespace Tokenizer
} // namespace GRIM