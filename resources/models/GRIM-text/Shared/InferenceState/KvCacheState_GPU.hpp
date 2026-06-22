//======================================================//
//  KvCacheState_GPU.hpp
//  Session-scoped KV cache for autoregressive decode.
//
//  Owns, per encoder layer, the bf16 key/value capacity buffers that the
//  FlashAttention KV-cache decode kernel (flash_attn_fwd_kvcache_rotary) reads
//  and appends to in-place. Also owns the shared fused-rotary cos/sin tables and
//  a single device cache-fill counter (cache_seqlens) plus a host mirror.
//
//  Lifetime: allocated lazily on first prefill, reused across decode steps within
//  one generation session, and re-zeroed (buffers kept) at session reset. This is
//  generation session state, owned by GenerationState — NOT training state and
//  NOT a forward-call sink.
//
//  Layouts (batch is always 1 for inference):
//    k_cache/v_cache[layer] : [1, cache_max_seq, n_kv_heads, head_dim]  (bf16)
//    rotary_cos/rotary_sin  : [cache_max_seq, rotary_dim/2]             (bf16)
//    cache_seqlens          : device int[1] = fill BEFORE the next forward
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace GRIM {

// Per-layer view handed to the cached attention facade for one forward call.
// All pointers are borrowed from the owning KvCacheState; the view owns nothing.
struct KvCacheLayerView {
    __nv_bfloat16* k_cache = nullptr;         // [1, cache_max_seq, n_kv_heads, head_dim] (in/out)
    __nv_bfloat16* v_cache = nullptr;         // [1, cache_max_seq, n_kv_heads, head_dim] (in/out)
    int* cache_seqlens = nullptr;             // device int[1] — fill BEFORE this call
    const __nv_bfloat16* rotary_cos = nullptr; // [cache_max_seq, rotary_dim/2]
    const __nv_bfloat16* rotary_sin = nullptr; // [cache_max_seq, rotary_dim/2]
    int cache_max_seq = 0;
    int rotary_dim = 0;

    // Shared scratch (sized for the maximum q_len = cache_max_seq), reused across
    // every layer within one forward — layers run sequentially on one stream.
    __nv_bfloat16* scratch_q = nullptr;       // [1, q_len, n_heads, head_dim]
    __nv_bfloat16* scratch_knew = nullptr;    // [1, q_len, n_kv_heads, head_dim]
    __nv_bfloat16* scratch_vnew = nullptr;    // [1, q_len, n_kv_heads, head_dim]
    __nv_bfloat16* scratch_out = nullptr;     // [1, q_len, n_heads, head_dim]
    float* scratch_lse = nullptr;             // [1, n_heads, q_len]

    bool valid() const {
        return k_cache && v_cache && cache_seqlens && rotary_cos && rotary_sin &&
               scratch_q && scratch_knew && scratch_vnew && scratch_out && scratch_lse &&
               cache_max_seq > 0 && rotary_dim > 0;
    }
};

struct KvCacheState {
    bool allocated = false;
    int num_layers = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int rotary_dim = 0;
    int cache_max_seq = 0;

    std::vector<__nv_bfloat16*> k_cache;      // size num_layers
    std::vector<__nv_bfloat16*> v_cache;      // size num_layers
    int* d_cache_seqlens = nullptr;           // device int[1]
    int host_seqlen = 0;                      // host mirror of d_cache_seqlens
    __nv_bfloat16* rotary_cos = nullptr;
    __nv_bfloat16* rotary_sin = nullptr;

    // Shared per-forward scratch (see KvCacheLayerView).
    __nv_bfloat16* scratch_q = nullptr;
    __nv_bfloat16* scratch_knew = nullptr;
    __nv_bfloat16* scratch_vnew = nullptr;
    __nv_bfloat16* scratch_out = nullptr;
    float* scratch_lse = nullptr;

    KvCacheState() = default;
    ~KvCacheState() { release(); }
    KvCacheState(const KvCacheState&) = delete;
    KvCacheState& operator=(const KvCacheState&) = delete;

    // Lazily allocate all device buffers and build the fused-rotary cos/sin tables
    // from rope_inv_freq (device [rotary_dim/2], borrowed from PBMState). Idempotent
    // when the geometry is unchanged; reallocates if geometry differs.
    void ensureAllocated(int num_layers,
                         int n_heads,
                         int n_kv_heads,
                         int head_dim,
                         int rotary_dim,
                         int cache_max_seq,
                         const float* rope_inv_freq,
                         cudaStream_t stream);

    // Host-only reset (no device op). Safe to call from GenerationState::resetSession()
    // which has no stream. The device counter is re-uploaded by beginSession().
    void resetSession() { host_seqlen = 0; }

    // Start a generation session: host_seqlen = 0 and upload it to the device.
    void beginSession(cudaStream_t stream);

    // host_seqlen += n and upload (called once per forward, after all layers).
    void advance(int n, cudaStream_t stream);

    // host_seqlen = v and upload (speculative-decode cache rollback / set).
    void setSeqlen(int v, cudaStream_t stream);

    int currentSeqlen() const { return host_seqlen; }
    int remainingCapacity() const { return cache_max_seq - host_seqlen; }

    // Build a per-layer view for the cached attention facade.
    KvCacheLayerView layerView(int layer_idx) const;

    void release();
};

} // namespace GRIM

#endif // USE_CUDA
