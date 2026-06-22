//======================================================//
//  KvCacheState_GPU.cu
//  Allocation + bookkeeping for the session-scoped KV cache.
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "KvCacheState_GPU.hpp"

#include <stdexcept>
#include <string>

#include "../CudaAllocUtils.hpp"
#include "../PBM/PositionalBiasMethod.hpp"

namespace GRIM {

namespace {

template <typename T>
void freeDevice(T*& ptr) {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

__nv_bfloat16* allocBf16(size_t elems, const char* label) {
    __nv_bfloat16* ptr = nullptr;
    GRIM::CudaAlloc::cudaMallocOrThrow(reinterpret_cast<void**>(&ptr),
                                       elems * sizeof(__nv_bfloat16), label);
    cudaMemset(ptr, 0, elems * sizeof(__nv_bfloat16));
    return ptr;
}

} // namespace

void KvCacheState::ensureAllocated(int num_layers_in,
                                   int n_heads_in,
                                   int n_kv_heads_in,
                                   int head_dim_in,
                                   int rotary_dim_in,
                                   int cache_max_seq_in,
                                   const float* rope_inv_freq,
                                   cudaStream_t stream) {
    if (num_layers_in <= 0 || n_heads_in <= 0 || n_kv_heads_in <= 0 ||
        head_dim_in <= 0 || cache_max_seq_in <= 0) {
        throw std::runtime_error("KvCacheState::ensureAllocated: invalid geometry");
    }
    if (rotary_dim_in <= 0 || (rotary_dim_in & 1) != 0 || rotary_dim_in > head_dim_in) {
        throw std::runtime_error("KvCacheState::ensureAllocated: invalid rotary_dim=" +
                                 std::to_string(rotary_dim_in) + " for head_dim=" +
                                 std::to_string(head_dim_in));
    }
    if (!rope_inv_freq) {
        throw std::runtime_error("KvCacheState::ensureAllocated: rope_inv_freq is NULL "
                                 "(PBM state not initialized?)");
    }

    // Idempotent when geometry is unchanged — the rotary tables and caches are
    // constant for the model across sessions.
    if (allocated &&
        num_layers == num_layers_in && n_heads == n_heads_in &&
        n_kv_heads == n_kv_heads_in && head_dim == head_dim_in &&
        rotary_dim == rotary_dim_in && cache_max_seq == cache_max_seq_in) {
        return;
    }

    release();

    num_layers = num_layers_in;
    n_heads = n_heads_in;
    n_kv_heads = n_kv_heads_in;
    head_dim = head_dim_in;
    rotary_dim = rotary_dim_in;
    cache_max_seq = cache_max_seq_in;

    const size_t kv_elems_per_layer =
        static_cast<size_t>(cache_max_seq) * n_kv_heads * head_dim;  // batch = 1
    k_cache.assign(static_cast<size_t>(num_layers), nullptr);
    v_cache.assign(static_cast<size_t>(num_layers), nullptr);
    for (int l = 0; l < num_layers; ++l) {
        k_cache[static_cast<size_t>(l)] = allocBf16(kv_elems_per_layer, "kv_cache_k");
        v_cache[static_cast<size_t>(l)] = allocBf16(kv_elems_per_layer, "kv_cache_v");
    }

    // Device fill counter.
    GRIM::CudaAlloc::cudaMallocOrThrow(reinterpret_cast<void**>(&d_cache_seqlens),
                                       sizeof(int), "kv_cache_seqlens");

    // Fused-rotary cos/sin tables: [cache_max_seq, rotary_dim/2] bf16, identical to
    // GRIM's training-time interleaved (GPT-J) RoPE because they are built from the
    // same rope_inv_freq.
    const size_t table_elems =
        static_cast<size_t>(cache_max_seq) * static_cast<size_t>(rotary_dim / 2);
    rotary_cos = allocBf16(table_elems, "kv_cache_rotary_cos");
    rotary_sin = allocBf16(table_elems, "kv_cache_rotary_sin");
    PBM::launchBuildRotaryCosSinTables(
        rope_inv_freq, rotary_cos, rotary_sin, cache_max_seq, rotary_dim,
        /*is_bf16=*/true, stream);

    // Shared per-forward scratch, sized for the maximum q_len (= cache_max_seq).
    const size_t q_scratch_elems =
        static_cast<size_t>(cache_max_seq) * n_heads * head_dim;
    const size_t kv_scratch_elems =
        static_cast<size_t>(cache_max_seq) * n_kv_heads * head_dim;
    const size_t lse_scratch_elems =
        static_cast<size_t>(cache_max_seq) * n_heads;
    scratch_q = allocBf16(q_scratch_elems, "kv_cache_scratch_q");
    scratch_knew = allocBf16(kv_scratch_elems, "kv_cache_scratch_knew");
    scratch_vnew = allocBf16(kv_scratch_elems, "kv_cache_scratch_vnew");
    scratch_out = allocBf16(q_scratch_elems, "kv_cache_scratch_out");
    GRIM::CudaAlloc::cudaMallocOrThrow(reinterpret_cast<void**>(&scratch_lse),
                                       lse_scratch_elems * sizeof(float),
                                       "kv_cache_scratch_lse");

    host_seqlen = 0;
    cudaError_t err = cudaMemsetAsync(d_cache_seqlens, 0, sizeof(int), stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("KvCacheState::ensureAllocated: cache_seqlens zero failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("KvCacheState::ensureAllocated: sync failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    allocated = true;
}

void KvCacheState::beginSession(cudaStream_t stream) {
    if (!allocated) {
        throw std::runtime_error("KvCacheState::beginSession: not allocated");
    }
    setSeqlen(0, stream);
}

void KvCacheState::advance(int n, cudaStream_t stream) {
    if (n < 0) {
        throw std::runtime_error("KvCacheState::advance: negative n");
    }
    setSeqlen(host_seqlen + n, stream);
}

void KvCacheState::setSeqlen(int v, cudaStream_t stream) {
    if (!allocated) {
        throw std::runtime_error("KvCacheState::setSeqlen: not allocated");
    }
    if (v < 0 || v > cache_max_seq) {
        throw std::runtime_error("KvCacheState::setSeqlen: value " + std::to_string(v) +
                                 " out of range [0, " + std::to_string(cache_max_seq) + "]");
    }
    host_seqlen = v;
    cudaError_t err = cudaMemcpyAsync(d_cache_seqlens, &host_seqlen, sizeof(int),
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("KvCacheState::setSeqlen: upload failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    // The next forward reads d_cache_seqlens from the device immediately; make the
    // tiny upload visible before the caller launches attention.
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("KvCacheState::setSeqlen: sync failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

KvCacheLayerView KvCacheState::layerView(int layer_idx) const {
    if (!allocated) {
        throw std::runtime_error("KvCacheState::layerView: not allocated");
    }
    if (layer_idx < 0 || layer_idx >= num_layers) {
        throw std::runtime_error("KvCacheState::layerView: layer_idx=" +
                                 std::to_string(layer_idx) + " out of range [0, " +
                                 std::to_string(num_layers) + ")");
    }
    KvCacheLayerView view{};
    view.k_cache = k_cache[static_cast<size_t>(layer_idx)];
    view.v_cache = v_cache[static_cast<size_t>(layer_idx)];
    view.cache_seqlens = d_cache_seqlens;
    view.rotary_cos = rotary_cos;
    view.rotary_sin = rotary_sin;
    view.cache_max_seq = cache_max_seq;
    view.rotary_dim = rotary_dim;
    view.scratch_q = scratch_q;
    view.scratch_knew = scratch_knew;
    view.scratch_vnew = scratch_vnew;
    view.scratch_out = scratch_out;
    view.scratch_lse = scratch_lse;
    return view;
}

void KvCacheState::release() {
    for (auto*& p : k_cache) freeDevice(p);
    for (auto*& p : v_cache) freeDevice(p);
    k_cache.clear();
    v_cache.clear();
    freeDevice(d_cache_seqlens);
    freeDevice(rotary_cos);
    freeDevice(rotary_sin);
    freeDevice(scratch_q);
    freeDevice(scratch_knew);
    freeDevice(scratch_vnew);
    freeDevice(scratch_out);
    freeDevice(scratch_lse);
    allocated = false;
    host_seqlen = 0;
    num_layers = n_heads = n_kv_heads = head_dim = rotary_dim = cache_max_seq = 0;
}

} // namespace GRIM
