//======================================================//
//  GenerationState_GPU.hpp
//  Explicit owner for autoregressive inference/generation state
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_bf16.h>

#include "../TrainingState/DeviceAllocation_GPU.hpp"

namespace GRIM {

struct GenerationState {
    struct KVCacheShape {
        int num_layers = 0;
        int num_kv_heads = 0;
        int head_dim = 0;
        int capacity_tokens = 0;

        bool valid() const {
            return num_layers > 0 && num_kv_heads > 0 && head_dim > 0 && capacity_tokens > 0;
        }

        void requireValid(const char* caller) const {
            if (!valid()) {
                throw std::runtime_error(std::string(caller) +
                    ": invalid KV cache shape layers=" + std::to_string(num_layers) +
                    " kv_heads=" + std::to_string(num_kv_heads) +
                    " head_dim=" + std::to_string(head_dim) +
                    " capacity_tokens=" + std::to_string(capacity_tokens));
            }
        }

        std::size_t elementsPerLayer() const {
            requireValid("GenerationState::KVCacheShape::elementsPerLayer");
            return static_cast<std::size_t>(num_kv_heads) *
                   static_cast<std::size_t>(capacity_tokens) *
                   static_cast<std::size_t>(head_dim);
        }

        std::size_t bytesPerLayer() const {
            return elementsPerLayer() * sizeof(__nv_bfloat16);
        }
    };

    struct KVCacheBuffers {
        KVCacheShape shape;
        std::vector<DeviceAllocation> k;    // BF16, one per encoder layer
        std::vector<DeviceAllocation> v;    // BF16, one per encoder layer
        DeviceAllocation softmax_lse;       // FP32 FlashAttention decode LSE scratch

        bool allocated() const {
            return shape.valid() && k.size() == static_cast<std::size_t>(shape.num_layers) &&
                   v.size() == static_cast<std::size_t>(shape.num_layers) &&
                   !k.empty() && !v.empty() && static_cast<bool>(softmax_lse);
        }

        int capacityTokens() const {
            shape.requireValid("GenerationState::KVCacheBuffers::capacityTokens");
            return shape.capacity_tokens;
        }

        std::size_t bytesPerLayer() const {
            return shape.bytesPerLayer();
        }
    };

    struct DecodeScratch {
        DeviceAllocation q_bf16;        // [num_heads * head_dim] BF16
        DeviceAllocation attn_out_bf16; // [num_heads * head_dim] BF16
        DeviceAllocation attn_out_fp32; // [d_model] FP32

        bool allocated() const {
            return static_cast<bool>(q_bf16) &&
                   static_cast<bool>(attn_out_bf16) &&
                   static_cast<bool>(attn_out_fp32);
        }
    };

    int kv_cache_len = 0;      // Valid tokens in KV cache for current generation session
    KVCacheBuffers kv_cache;
    DecodeScratch decode_scratch;

    bool kvReady() const {
        return kv_cache.allocated() && decode_scratch.allocated();
    }

    void resetSession() {
        kv_cache_len = 0;
    }
};

} // namespace GRIM

#endif // USE_CUDA
