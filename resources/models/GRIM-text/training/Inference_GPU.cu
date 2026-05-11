//======================================================//
//  Inference_GPU.cu
//  Inference using the shared mode-explicit forward primitive
//  
//  Single inference entry point: executeInferenceForward_()
//  All public inference methods copy data to cached_* tensors
//  then call this private method.
//
//  Rule 20: Payload-authored inference only - uses autograd forward only
//  Rule 26: One inference path, not two
//======================================================//

#include <vector>
#include <algorithm>
#include <cstdint>
#include <string>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdexcept>

#include "../GRIM/grim_language_model_cuda.hpp"
#include "../Shared/Forward/ModelForward_GPU.hpp"
#include "../Layers/Encoding/Encoding_GPU.hpp"
#include "../Layers/FlashAttention/Flash_Attention_Kernal.hpp"
#include "../Shared/TensorConversion/TensorConversion.hpp"
#include "../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.hpp"
#include "../Shared/Execution/DecodeTimeNumPolicy.hpp"
#include "../Shared/CudaAllocUtils.hpp"
#include "../Shared/Batching/BatchPayload.hpp"
#include "../Shared/Batching/BatchDeviceBindings.hpp"

namespace GRIM {

namespace {

constexpr int kScratchTextFeatureDim = 16;

ScratchBlockLayer::RowLocalAtomView buildDecodeExecutionAtomView(
    ScratchBlockLayer& scratch_block,
    const HyperParameters::LanguageModelConfig& cfg,
    TrainingState& ts,
    cudaStream_t stream,
    int token_pos)
{
    auto throwCuda = [](const char* what, cudaError_t err) {
        throw std::runtime_error(std::string("buildDecodeExecutionAtomView: ") +
                                 what + ": " + cudaGetErrorString(err));
    };

    if (!ts.cached_token_ids_tensor.data) {
        throw std::runtime_error("buildDecodeExecutionAtomView: cached_token_ids_tensor is NULL");
    }
    if (!ts.cached_token_numeric_values.data) {
        throw std::runtime_error("buildDecodeExecutionAtomView: cached_token_numeric_values is NULL");
    }
    if (!ts.cached_token_text_features.data) {
        throw std::runtime_error("buildDecodeExecutionAtomView: cached_token_text_features is NULL");
    }
    if (!ts.cached_token_atom_mask.data) {
        throw std::runtime_error("buildDecodeExecutionAtomView: cached_token_atom_mask is NULL");
    }
    if (!ts.cached_token_atom_flags.data) {
        throw std::runtime_error("buildDecodeExecutionAtomView: cached_token_atom_flags is NULL");
    }
    if (!ts.cached_token_to_slot_map.data) {
        throw std::runtime_error("buildDecodeExecutionAtomView: cached_token_to_slot_map is NULL during decode-time execution");
    }

    auto dummy_hidden = Tensor::zeros({1, cfg.d_model}, stream, "decode_exec_atom_dummy_hidden");
    const bool exec_first_type_only =
        cfg.execution_block_enabled && cfg.scratch_block_execution_first_type_only;

    scratch_block.runForwardKernels(
        dummy_hidden.data,
        1,
        reinterpret_cast<const int*>(ts.cached_token_ids_tensor.data) + token_pos,
        ts.cached_token_numeric_values.data + token_pos,
        reinterpret_cast<const uint16_t*>(ts.cached_token_text_features.data) +
            static_cast<size_t>(token_pos) * kScratchTextFeatureDim,
        reinterpret_cast<const uint8_t*>(ts.cached_token_atom_mask.data) + token_pos,
        reinterpret_cast<const uint32_t*>(ts.cached_token_atom_flags.data) + token_pos,
        reinterpret_cast<const int32_t*>(ts.cached_token_to_slot_map.data) + token_pos,
        stream,
        exec_first_type_only);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throwCuda("ScratchBlock decode atom path", err);

    auto view = scratch_block.extractRowLocalAtomView(0, 1, stream);

    int32_t slot_id = -1;
    err = cudaMemcpyAsync(&slot_id,
                          reinterpret_cast<const int32_t*>(ts.cached_token_to_slot_map.data) + token_pos,
                          sizeof(int32_t),
                          cudaMemcpyDeviceToHost,
                          stream);
    if (err != cudaSuccess) throwCuda("cudaMemcpyAsync(slot_id)", err);
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) throwCuda("cudaStreamSynchronize(slot_id)", err);

    if (slot_id >= 0 && view.num_atoms == 0) {
        throw std::runtime_error(
            "buildDecodeExecutionAtomView: decode-time execution token is slot-bound but ScratchBlock produced no row-local atom view");
    }

    return view;
}

Batching::BatchDeviceBindings buildInferencePrefillBindings(TrainingState& ts, int seq_len) {
    if (seq_len <= 0) {
        throw std::runtime_error("buildInferencePrefillBindings: seq_len <= 0");
    }
    Batching::BatchDeviceBindings bindings;
    bindings.batch_size = 1;
    bindings.max_seq_len = seq_len;
    bindings.d_input_ids = reinterpret_cast<int*>(ts.cached_token_ids_tensor.data);
    bindings.d_seq_lengths = reinterpret_cast<int*>(ts.cached_seq_lengths_tensor.data);
    bindings.d_numeric_values = ts.cached_token_numeric_values.data;
    bindings.d_text_features = reinterpret_cast<uint16_t*>(ts.cached_token_text_features.data);
    bindings.d_atom_mask = reinterpret_cast<uint8_t*>(ts.cached_token_atom_mask.data);
    bindings.d_atom_flags = reinterpret_cast<uint32_t*>(ts.cached_token_atom_flags.data);
    bindings.d_token_to_slot_map = reinterpret_cast<int32_t*>(ts.cached_token_to_slot_map.data);

    if (!bindings.d_input_ids) {
        throw std::runtime_error("buildInferencePrefillBindings: cached_token_ids_tensor.data is NULL");
    }
    if (!bindings.d_seq_lengths) {
        throw std::runtime_error("buildInferencePrefillBindings: cached_seq_lengths_tensor.data is NULL");
    }
    if (!bindings.d_token_to_slot_map) {
        throw std::runtime_error("buildInferencePrefillBindings: cached_token_to_slot_map.data is NULL");
    }
    return bindings;
}

bool sequenceCoupledGeometryRequiresFullPrefill(
    const HyperParameters::LanguageModelConfig& cfg) {
    return cfg.center_encoder_residuals ||
           cfg.lm_head_center_hidden_states ||
           cfg.project_out_pc1;
}

void updateDecodeSelectorAfterPrefill(LanguageModel& model,
                                      const HyperParameters::LanguageModelConfig& cfg,
                                      TrainingState& ts,
                                      GenerationState& gen,
                                      cudaStream_t stream,
                                      int seq_len,
                                      const char* caller) {
    if (seq_len <= 0) {
        throw std::runtime_error(std::string(caller) + ": seq_len <= 0 while updating decode selector");
    }

    // For the selector, exec_block_active = "execution block system is
    // configured and memory is live." Trace structures are irrelevant for a
    // full prefill pass; decode-step execution traces are used only by the KV
    // single-token path.
    const bool selector_can_run = cfg.execution_block_enabled
        && model.getExecutionBlockLayer() != nullptr
        && model.getScratchBlockLayer() != nullptr
        && model.getScratchBlockLayer()->isEnabled()
        && gen.has_exec_memory;

    const float* last_hidden = ts.cached_encoder_output.data
        ? ts.cached_encoder_output.data + static_cast<size_t>(seq_len - 1) * cfg.d_model
        : nullptr;

    if (!last_hidden && cfg.selector_enabled) {
        throw std::runtime_error(
            std::string(caller) + ": selector_enabled but cached_encoder_output is NULL "
            "after prefill — cannot evaluate decode-time slot selector");
    }

    DecodeTimeResolveResult sel = resolveDecodeTimeNumSlotSelectionOrMask(
        model.getDecodeTimeSlotSelectorLayer(),
        model.getDecodeTimeNumPolicy(),
        cfg.selector_enabled,
        selector_can_run,
        gen.has_exec_memory,
        gen.exec_memory,
        last_hidden,
        stream,
        ts.cublas_handle.get());

    gen.decode_selector.valid          = sel.valid;
    gen.decode_selector.status         = static_cast<uint8_t>(sel.status);
    gen.decode_selector.selected_slot  = sel.selected_slot;
    gen.decode_selector.selected_value = sel.selected_value;
}

}  // namespace

//======================================================//
//  executeInferenceForward_ - THE single inference forward path
//  Assumes all data already in cached_* tensors.
//  Builds an explicit inference prefill request, runs forward, returns last-token logits.
//  When populate_kv_cache=true, extracts per-layer K,V from autograd
//  intermediates and converts to BF16 BSHD format in KV cache buffers.
//======================================================//
Vector LanguageModel::executeInferenceForward_(int seq_len, bool populate_kv_cache) {
    if (!training_state_.initialized) {
        throw std::runtime_error("executeInferenceForward_: training state not initialized");
    }
    if (seq_len <= 0 || seq_len > config_.max_seq_len) {
        throw std::runtime_error("executeInferenceForward_: seq_len=" + std::to_string(seq_len) +
                                 " out of range [1, " + std::to_string(config_.max_seq_len) + "]");
    }

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();

    if (!training_state_.cached_seq_lengths_tensor.data) {
        throw std::runtime_error("executeInferenceForward_: cached_seq_lengths_tensor.data is NULL");
    }
    int h_seq_len = seq_len;
    cudaError_t len_cp = cudaMemcpyAsync(training_state_.cached_seq_lengths_tensor.data,
                                         &h_seq_len,
                                         sizeof(int),
                                         cudaMemcpyHostToDevice,
                                         stream);
    if (len_cp != cudaSuccess) {
        throw std::runtime_error("executeInferenceForward_: cudaMemcpyAsync(seq_len) failed: " +
                                 std::string(cudaGetErrorString(len_cp)));
    }

    Batching::BatchDeviceBindings bindings = buildInferencePrefillBindings(training_state_, seq_len);
    Batching::BatchPayload payload = Batching::buildInferenceStagedPayload(
        seq_len, config_.vocab_size, static_cast<size_t>(config_.max_cached_seq_len),
        generation_state_.has_exec_memory);

    return executeInferenceForward_(payload, bindings, populate_kv_cache);
}

Vector LanguageModel::executeInferenceForward_(
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    bool populate_kv_cache) {
    if (!training_state_.initialized) {
        throw std::runtime_error("executeInferenceForward_: training state not initialized");
    }
    payload.validate("executeInferenceForward_");
    if (!payload.isInference()) {
        throw std::runtime_error("executeInferenceForward_: payload mode is training; inference forward requires an inference payload");
    }
    if (payload.batch_size != 1) {
        throw std::runtime_error("executeInferenceForward_: inference payload batch_size must be 1");
    }
    if (payload.max_seq_len <= 0 || payload.max_seq_len > config_.max_seq_len) {
        throw std::runtime_error("executeInferenceForward_: payload max_seq_len=" + std::to_string(payload.max_seq_len) +
                                 " out of range [1, " + std::to_string(config_.max_seq_len) + "]");
    }
    if (bindings.batch_size != payload.batch_size || bindings.max_seq_len != payload.max_seq_len) {
        throw std::runtime_error(
            "executeInferenceForward_: BatchDeviceBindings geometry (" +
            std::to_string(bindings.batch_size) + "x" + std::to_string(bindings.max_seq_len) +
            ") does not match payload (" + std::to_string(payload.batch_size) + "x" +
            std::to_string(payload.max_seq_len) + ")");
    }

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();

    Forward::ModelForwardRequest request{};
    request.config = &config_;
    request.runtime_state = &training_state_;
    request.gpu_encoder = &getGpuEncoder();
    request.embedding_layer = getEmbeddingLayer();
    request.lm_head = getLmHeadLayer();
    request.scratch_block = getScratchBlockLayer();
    request.reasoning_head = getReasoningHeadLayer();
    request.execution_block = getExecutionBlockLayer();
    request.cublas_handle = training_state_.cublas_handle.get();
    request.stream = stream;
    request.payload = &payload;
    request.bindings = &bindings;
    request.step = 0;
    request.mode = Forward::ModelForwardMode::InferencePrefill;

    Forward::ModelForwardResult result = Forward::executeModelForward(request);
    if (!result.success) {
        throw std::runtime_error("executeInferenceForward_: forward failed - " + result.error_message);
    }

    // Extract last token logits
    if (!training_state_.cached_logits_tensor.data) {
        throw std::runtime_error("executeInferenceForward_: cached_logits_tensor not initialized");
    }
    Vector logits(config_.vocab_size);
    const size_t last_token_offset = static_cast<size_t>(payload.max_seq_len - 1) * config_.vocab_size;
    cudaMemcpyAsync(logits.data.data(),
                    training_state_.cached_logits_tensor.data + last_token_offset,
                    config_.vocab_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    // Persist ExecutionMemory so decode-time execution state survives across
    // autoregressive forwardStep calls.
    if (!training_state_.autograd_intermediates.exec_memories.empty()) {
        generation_state_.exec_memory =
            std::move(training_state_.autograd_intermediates.exec_memories[0]);
        generation_state_.has_exec_memory = true;
    }

    // Populate KV cache from autograd intermediates (prefill mode).
    // Must happen BEFORE clear() which destroys the intermediate tensors.
    auto& gen = generation_state_;
    if (populate_kv_cache && gen.kv_cache.k.empty()) {
        throw std::runtime_error("executeInferenceForward_: populate_kv_cache=true but KV cache not allocated — "
                                 "call ensureKVCacheAllocated() before generation");
    }
    if (populate_kv_cache && !gen.kv_cache.k.empty()) {
        const auto& kv_shape = gen.kv_cache.shape;
        kv_shape.requireValid("executeInferenceForward_ KV cache population");
        const int num_kv_heads = kv_shape.num_kv_heads;
        const int head_dim = kv_shape.head_dim;
        const auto& layers = training_state_.autograd_intermediates.layer_intermediates.layers;
        const int num_layers = static_cast<int>(layers.size());

        if (num_layers != kv_shape.num_layers) {
            throw std::runtime_error(
                "executeInferenceForward_: KV prefill expected " + std::to_string(kv_shape.num_layers) +
                " layer intermediate snapshots but ModelForward produced " + std::to_string(num_layers) +
                " — InferencePrefill mode MUST preserve per-layer K/V tensors until KV copy completes");
        }

        for (int i = 0; i < num_layers; ++i) {
            const auto& layer_ints = layers[i];
            if (!layer_ints.K_bhsd.data || !layer_ints.V_bhsd.data) {
                throw std::runtime_error("executeInferenceForward_: layer " + std::to_string(i) +
                                         " K_bhsd/V_bhsd is NULL during KV cache population");
            }
            // K_bhsd: [1, nkv, seq_len, hd] BHSD FP32 (post-RoPE)
            // V_bhsd: [1, nkv, seq_len, hd] BHSD FP32
            // Convert to BSHD BF16 → [1, seq_len, nkv, hd] in cache
            TensorConversion::convert_BHSD_to_BSHD_bf16(
                layer_ints.K_bhsd.data,
                gen.kv_cache.k[i].as<__nv_bfloat16>(),
                1, num_kv_heads, payload.max_seq_len, head_dim, stream);
            TensorConversion::convert_BHSD_to_BSHD_bf16(
                layer_ints.V_bhsd.data,
                gen.kv_cache.v[i].as<__nv_bfloat16>(),
                1, num_kv_heads, payload.max_seq_len, head_dim, stream);
        }
    }

    // Free all autograd intermediates — inference never runs backward.
    // Without this, grad_fn chains and cached tensors leak across generation steps.
    training_state_.autograd_intermediates.clear();

    return logits;
}

Vector LanguageModel::forwardInit(const Batching::BatchPayload& prompt_payload) {
    prompt_payload.validate("forwardInit(BatchPayload)");
    if (!prompt_payload.isInferencePrefill()) {
        throw std::runtime_error("forwardInit(BatchPayload): payload must be InferencePrefill");
    }
    if (prompt_payload.batch_size != 1) {
        throw std::runtime_error("forwardInit(BatchPayload): batch_size must be 1");
    }
    const int seq_len = prompt_payload.max_seq_len;
    if (seq_len <= 0) {
        throw std::runtime_error("forwardInit(BatchPayload): seq_len <= 0");
    }
    if (seq_len > config_.max_seq_len) {
        throw std::runtime_error("forwardInit(BatchPayload): seq_len exceeds max_seq_len");
    }

    const auto bindings = uploadBatchToDevice(prompt_payload);

    // Prefill: populate KV cache from autograd intermediates.
    // Commit kv_cache_len AFTER success — if forward throws, the cache
    // must not claim it holds tokens that never completed the forward path.
    Vector logits = executeInferenceForward_(prompt_payload, bindings, /*populate_kv_cache=*/true);
    generation_state_.kv_cache_len = seq_len;

    cudaStream_t post_stream = training_state_.stream_ctrl.getPrimaryStream();
    if (generation_state_.has_exec_memory
        && generation_state_.trace_state_by_row.empty()) {
        generation_state_.trace_state_by_row.resize(1);
        generation_state_.trace_state_by_row[0] = Tensor::zeros(
            {1, config_.d_model}, post_stream, "trace_state_decode");
        generation_state_.trace_state_by_row[0].requires_grad = false;
    }
    if (generation_state_.has_exec_memory
        && generation_state_.execution_trace_by_row.empty()) {
        generation_state_.execution_trace_by_row.resize(1);
        generation_state_.execution_trace_by_row[0].clear();
    }

    updateDecodeSelectorAfterPrefill(*this, config_, training_state_, generation_state_,
                                     post_stream, seq_len, "forwardInit(BatchPayload)");

    return logits;
}

//======================================================//
//  forwardStep - Decode phase for one appended token
//  Uses KV-cache decode only when the config is sequence-local. Configs with
//  sequence-coupled centering/projection rerun the full current sequence.
//======================================================//
Vector LanguageModel::forwardStep(int new_token, float numeric_value, uint8_t atom_mask,
                                  int32_t new_token_slot_id) {
    if (!training_state_.initialized) {
        throw std::runtime_error("forwardStep: training state not initialized");
    }
    if (generation_state_.kv_cache_len == 0) {
        throw std::runtime_error("forwardStep: call forwardInit first");
    }

    const int token_pos = generation_state_.kv_cache_len;
    const int new_seq_len = token_pos + 1;
    
    if (new_seq_len > config_.max_seq_len) {
        throw std::runtime_error("forwardStep: sequence exceeds max_seq_len");
    }
    if (!generation_state_.kvReady()) {
        throw std::runtime_error("forwardStep: KV cache not allocated — call initInferenceState first");
    }

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    
    // Append new token to device buffer (needed for embedding lookup)
    int* token_ids_ptr = reinterpret_cast<int*>(training_state_.cached_token_ids_tensor.data);
    cudaMemcpyAsync(token_ids_ptr + token_pos,
                    &new_token, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    
    // Append numeric side-channel
    cudaMemcpyAsync(training_state_.cached_token_numeric_values.data + token_pos,
                    &numeric_value, sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(reinterpret_cast<uint8_t*>(training_state_.cached_token_atom_mask.data) + token_pos,
                    &atom_mask, sizeof(uint8_t),
                    cudaMemcpyHostToDevice, stream);

    if (!training_state_.cached_token_text_features.data) {
        throw std::runtime_error("forwardStep: cached_token_text_features.data is NULL");
    }
    cudaMemsetAsync(reinterpret_cast<uint16_t*>(training_state_.cached_token_text_features.data) +
                    static_cast<size_t>(token_pos) * Batching::BatchPayload::kTextFeatureDim,
                    0,
                    static_cast<size_t>(Batching::BatchPayload::kTextFeatureDim) * sizeof(uint16_t),
                    stream);

    if (!training_state_.cached_token_atom_flags.data) {
        throw std::runtime_error("forwardStep: cached_token_atom_flags.data is NULL");
    }
    uint32_t zero_atom_flags = 0;
    cudaMemcpyAsync(reinterpret_cast<uint32_t*>(training_state_.cached_token_atom_flags.data) + token_pos,
                    &zero_atom_flags, sizeof(uint32_t),
                    cudaMemcpyHostToDevice, stream);

    if (training_state_.cached_token_to_slot_map.data) {
        auto* slot_dst = reinterpret_cast<int32_t*>(training_state_.cached_token_to_slot_map.data) + token_pos;
        cudaError_t serr = cudaMemcpyAsync(slot_dst, &new_token_slot_id, sizeof(int32_t),
                                           cudaMemcpyHostToDevice, stream);
        if (serr != cudaSuccess) {
            throw std::runtime_error(std::string("forwardStep slot map: ") + cudaGetErrorString(serr));
        }
    }

    // Sequence-coupled geometry cannot be evaluated from a single decoded row:
    // - encoder residual centering changes the residual stream mean over the
    //   whole sequence, invalidating KV-only layer updates;
    // - LM-head hidden centering needs all rows to avoid zeroing the current row;
    // - PC1 projection needs the sequence matrix, not a single vector.
    // Re-run the full current sequence for these configs. Plain configs keep the
    // fast KV path below.
    if (sequenceCoupledGeometryRequiresFullPrefill(config_)) {
        Vector logits = executeInferenceForward_(new_seq_len, /*populate_kv_cache=*/false);
        generation_state_.kv_cache_len = new_seq_len;
        updateDecodeSelectorAfterPrefill(*this, config_, training_state_, generation_state_,
                                         stream, new_seq_len, "forwardStep(full-prefill)");
        return logits;
    }

    // Commit kv_cache_len AFTER success — if decode throws, the cache
    // must not claim it holds a token that never completed the forward path.
    Vector logits = executeDecodeForward_(token_pos);
    generation_state_.kv_cache_len = new_seq_len;
    return logits;
}

//======================================================//
//  executeDecodeForward_ - KV-cached single-token decode
//  Processes one token through all encoder layers using
//  cached K,V from prior tokens. O(n) per layer instead of O(n²).
//======================================================//
Vector LanguageModel::executeDecodeForward_(int token_pos) {
    if (!training_state_.initialized) {
        throw std::runtime_error("executeDecodeForward_: training state not initialized");
    }
    if (!generation_state_.kvReady()) {
        throw std::runtime_error("executeDecodeForward_: KV cache not allocated");
    }

    using namespace TensorContract;
    namespace ag = autograd;

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    const auto& cfg = config_;
    auto& ts = training_state_;
    auto& gen = generation_state_;

    const int d_model = cfg.d_model;
    const int num_heads = cfg.num_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_dim = cfg.head_dim;
    const int num_layers = cfg.num_layers;
    const float rms_eps = cfg.rms_epsilon;
    const int seqlen_k = token_pos + 1;  // K cache has [0..token_pos] inclusive
    const HyperParameters::EncoderLayerConstructionHP encoder_hp =
        HyperParameters::encoderLayerConstructionHP(cfg);
    const HyperParameters::EncoderSelfAttentionHP attention_hp =
        HyperParameters::encoderSelfAttentionHP(encoder_hp);

    // Set cuBLAS handle for autograd matmul calls
    ag::set_autograd_cublas_handle(ts.cublas_handle.get());

    // ── Step 1: Embedding lookup for the single new token ──
    EmbeddingLayer* emb_layer = getEmbeddingLayer();
    if (!emb_layer) {
        throw std::runtime_error("executeDecodeForward_: embedding layer is NULL");
    }
    Tensor emb_view = emb_layer->tokenWeights().detach(stream);

    // Token ID is at device offset token_pos
    const int* token_id_ptr = reinterpret_cast<const int*>(ts.cached_token_ids_tensor.data) + token_pos;

    Tensor hidden = ag::embedding(emb_view, token_id_ptr, 1, stream, 1.0f);
    // hidden: [1, d_model]

    // ── Step 2: PBM validation ──
    const PBM::PBMSpec& pbm_spec = getPBMSpec();
    if (!pbm_spec.valid || !pbm_spec.rope_inv_freq || !pbm_spec.alibi_slopes) {
        throw std::runtime_error("executeDecodeForward_: PBM (RoPE/ALiBi) not initialized");
    }

    // ── ExecutionBlock setup ──
    ExecutionBlockLayer* exec_block = getExecutionBlockLayer();
    ScratchBlockLayer* scratch_block = getScratchBlockLayer();
    const bool exec_block_configured = cfg.execution_block_enabled && exec_block != nullptr;
    if (exec_block_configured && (!scratch_block || !scratch_block->isEnabled())) {
        throw std::runtime_error(
            "executeDecodeForward_: execution_block_enabled requires an enabled ScratchBlock for row-local decode atom views");
    }
    const bool exec_block_active = cfg.execution_block_enabled
        && exec_block != nullptr
        && scratch_block != nullptr
        && scratch_block->isEnabled()
        && gen.has_exec_memory
        && !gen.trace_state_by_row.empty()
        && !gen.execution_trace_by_row.empty();

    int exec_layer = -1;
    int exec_K = 0;
    if (exec_block_active) {
        exec_layer = cfg.execution_block_layer;
        if (exec_layer < 0) exec_layer = num_layers - 2;
        if (exec_layer < 0) exec_layer = 0;
        if (exec_layer >= num_layers) exec_layer = num_layers - 1;
        exec_K = cfg.execution_block_num_steps;
    }

    // ── Step 3: Process through all encoder layers ──
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        auto* enc = getGpuEncoder().getLayer(layer_idx);
        if (!enc) {
            throw std::runtime_error("executeDecodeForward_: encoder layer " +
                                     std::to_string(layer_idx) + " is NULL");
        }

        // 3a. RMSNorm1
        Tensor rms1_view = enc->rms1Gamma().detach(stream);
        Tensor ln1_out = ag::rms_norm(hidden, rms1_view, rms_eps, stream);

        // 3b. QKV projection: [1, d_model] @ W_qkv^T → [1, qkv_dim]
        Tensor wqkv_view = enc->attnWqkv().detach(stream);
        Tensor qkv = ag::matmul(ln1_out, wqkv_view, stream,
                                nullptr, nullptr, true);
        if (enc->attnBqkv().data) {
            Tensor bqkv_view = enc->attnBqkv().detach(stream);
            qkv = ag::broadcast_add(qkv, bqkv_view, stream);
        }

        // 3c. Split QKV → Q [1,nh,1,hd], K [1,nkv,1,hd], V [1,nkv,1,hd]
        Batching::BatchPayload inf_payload = Batching::buildInferenceDecodePayload(cfg.vocab_size);
        auto [Q, K, V] = ag::split_and_reshape_qkv(
            qkv, inf_payload, attention_hp, stream);

        // 3d. RoPE rotation with position offset
        // For S=1, Q is [1,nh,1,hd] and K is [1,nkv,1,hd]
        PBM::launchRoPERotationGQA(
            Q.data, K.data, pbm_spec.rope_inv_freq,
            1, num_heads, num_kv_heads, 1, head_dim,
            pbm_spec.rotary_dim, stream, token_pos);

        // 3e. Convert K,V to BF16 and write to KV cache at position token_pos
        // For S=1: BHSD [1,H,1,D] = BSHD [1,1,H,D] — same contiguous layout [H*D]
        // Cache layout: [kv_capacity, nkv, hd] BSHD
        const size_t kv_slot_offset = static_cast<size_t>(token_pos) * num_kv_heads * head_dim;
        auto* k_cache = gen.kv_cache.k[layer_idx].as<__nv_bfloat16>();
        auto* v_cache = gen.kv_cache.v[layer_idx].as<__nv_bfloat16>();

        TensorConversion::convert_BHSD_to_BSHD_bf16(
            K.data, k_cache + kv_slot_offset,
            1, num_kv_heads, 1, head_dim, stream);
        TensorConversion::convert_BHSD_to_BSHD_bf16(
            V.data, v_cache + kv_slot_offset,
            1, num_kv_heads, 1, head_dim, stream);

        // 3f. Convert Q to BF16 for flash attention
        auto* q_bf16 = gen.decode_scratch.q_bf16.as<__nv_bfloat16>();
        TensorConversion::convert_BHSD_to_BSHD_bf16(
            Q.data, q_bf16, 1, num_heads, 1, head_dim, stream);

        // 3g. Flash attention with KV cache
        // Q: [1, 1, nh, hd] BSHD BF16 (seqlen_q=1)
        // K cache: [1, kv_capacity, nkv, hd] BSHD BF16 (seqlen_k=token_pos+1)
        auto* attn_out_bf16 = gen.decode_scratch.attn_out_bf16.as<__nv_bfloat16>();
        flash_attn_fwd_kvcache(
            q_bf16,
            k_cache,
            v_cache,
            attn_out_bf16,
            gen.kv_cache.softmax_lse.as<float>(),
            pbm_spec.alibi_slopes,
            1,              // batch=1
            1,              // seqlen_q=1
            seqlen_k,       // cached tokens including current
            num_heads,
            num_kv_heads,
            head_dim,
            true,           // causal
            true,           // is_bf16
            stream);

        // 3h. Convert attention output BF16 → FP32
        // [1,1,nh,hd] BSHD → [1,nh,1,hd] BHSD (same memory for S=1)
        TensorConversion::convert_BSHD_bf16_to_BHSD(
            attn_out_bf16, gen.decode_scratch.attn_out_fp32.as<float>(),
            1, 1, num_heads, head_dim, stream);

        // 3i. Wrap attention output as non-owning Tensor [1, d_model]
        Tensor attn_flat;
        attn_flat.data = gen.decode_scratch.attn_out_fp32.as<float>();
        attn_flat.shape = TensorShape::make_BSM(1, d_model);
        attn_flat.requires_grad = false;
        attn_flat.owns_data = false;
        attn_flat.stream = stream;

        // 3j. Output projection: [1, d_model] @ W_o^T
        Tensor wo_view = enc->attnWo().detach(stream);
        Tensor proj = ag::matmul(attn_flat, wo_view, stream,
                                 nullptr, nullptr, true);
        if (enc->attnBo().data) {
            Tensor bo_view = enc->attnBo().detach(stream);
            proj = ag::broadcast_add(proj, bo_view, stream);
        }

        // 3k. LayerScale + residual
        if (cfg.use_layer_scale) {
            Tensor& gamma1 = enc->layerScale1();
            if (!gamma1.data) {
                throw std::runtime_error("executeDecodeForward_: use_layer_scale=true but layerScale1 is NULL at layer " +
                                         std::to_string(layer_idx));
            }
            gamma1.shape.require("executeDecodeForward_ layerScale1");
            if (!gamma1.shape.is_2d_layout()) {
                throw std::runtime_error("executeDecodeForward_: layerScale1 must be a 2D [1,d_model] gamma vector at layer " +
                                         std::to_string(layer_idx));
            }
            const auto gamma1_dims = gamma1.shape.as_2d();
            if (gamma1_dims.rows != 1 || gamma1_dims.cols != d_model) {
                throw std::runtime_error("executeDecodeForward_: layerScale1 must have shape [1,d_model] at layer " +
                                         std::to_string(layer_idx) + ". expected=[1," + std::to_string(d_model) +
                                         "] got=[" + std::to_string(gamma1_dims.rows) + "," +
                                         std::to_string(gamma1_dims.cols) + "]");
            }
            Tensor ls1_view = gamma1.detach(stream);
            proj = ag::layer_scale(proj, ls1_view, stream);
        }
        hidden = ag::add(hidden, proj, stream);
        // Skip center_columns for S=1 (centering a single vector is a no-op)

        // 3l. RMSNorm2
        Tensor rms2_view = enc->rms2Gamma().detach(stream);
        Tensor ln2_out = ag::rms_norm(hidden, rms2_view, rms_eps, stream);

        // 3m. FFN (SwiGLU)
        ForwardIntermediates ffn_ints;
        auto* ffn_layer = enc->getFfnLayer();
        if (!ffn_layer) {
            throw std::runtime_error("executeDecodeForward_: FFN layer is NULL at layer " +
                                     std::to_string(layer_idx));
        }
        Tensor ffn_out = ffn_layer->forward(ln2_out, ffn_ints,
                            stream, ts.cublas_handle.get(),
                            0, false, layer_idx);

        // 3n. LayerScale + residual
        if (cfg.use_layer_scale) {
            Tensor& gamma2 = enc->layerScale2();
            if (!gamma2.data) {
                throw std::runtime_error("executeDecodeForward_: use_layer_scale=true but layerScale2 is NULL at layer " +
                                         std::to_string(layer_idx));
            }
            gamma2.shape.require("executeDecodeForward_ layerScale2");
            if (!gamma2.shape.is_2d_layout()) {
                throw std::runtime_error("executeDecodeForward_: layerScale2 must be a 2D [1,d_model] gamma vector at layer " +
                                         std::to_string(layer_idx));
            }
            const auto gamma2_dims = gamma2.shape.as_2d();
            if (gamma2_dims.rows != 1 || gamma2_dims.cols != d_model) {
                throw std::runtime_error("executeDecodeForward_: layerScale2 must have shape [1,d_model] at layer " +
                                         std::to_string(layer_idx) + ". expected=[1," + std::to_string(d_model) +
                                         "] got=[" + std::to_string(gamma2_dims.rows) + "," +
                                         std::to_string(gamma2_dims.cols) + "]");
            }
            Tensor ls2_view = gamma2.detach(stream);
            ffn_out = ag::layer_scale(ffn_out, ls2_view, stream);
        }
        hidden = ag::add(hidden, ffn_out, stream);
        // Skip center_columns for S=1

        // ── ExecutionBlock: K execution steps at exec_layer ──
        if (exec_block_active && layer_idx == exec_layer) {
            const float T = cfg.execution_block_temp_start;
            const int32_t* slot_ptr = ts.cached_token_to_slot_map.data
                ? reinterpret_cast<const int32_t*>(ts.cached_token_to_slot_map.data) + token_pos
                : nullptr;
            auto row_atom_view = buildDecodeExecutionAtomView(
                *scratch_block, cfg, ts, stream, token_pos);

            // Bootstrap the new token's slot binding into execution memory.
            // During prefill, bootstrap runs inside AutogradTraining. During
            // decode, the single new token's numeric_value + slot_id must be
            // injected here so the execution block can operate on it.
            // WS7: Fail loud if bootstrap data is missing while execution is active.
            if (!slot_ptr || !ts.cached_token_numeric_values.data) {
                throw std::runtime_error(
                    "Inference: decode-time execution is active but slot map or numeric values "
                    "are missing for bootstrap at token_pos " + std::to_string(token_pos));
            }
            exec_block->bootstrapMemoryFromSlotMap(
                gen.exec_memory,
                ts.cached_token_numeric_values.data + token_pos,
                slot_ptr,
                1, stream);

            // Construct the shared single-token inference decode payload plus a
            // single-token BatchDeviceBindings for decode-time execution.
            // Decode is NOT a full training step — there is no host-side payload
            // upload, just the row-local slot map / atom mask already living in
            // TrainingState's cached buffers.
            GRIM::Batching::BatchPayload decode_payload =
                GRIM::Batching::buildInferenceDecodePayload(cfg.vocab_size);

            GRIM::Batching::BatchDeviceBindings decode_bindings;
            decode_bindings.batch_size  = 1;
            decode_bindings.max_seq_len = 1;
            decode_bindings.d_seq_lengths = reinterpret_cast<int*>(ts.cached_seq_lengths_tensor.data);
            decode_bindings.d_token_to_slot_map = const_cast<int32_t*>(slot_ptr);
            decode_bindings.d_atom_mask =
                ts.cached_token_atom_mask.data
                    ? reinterpret_cast<uint8_t*>(ts.cached_token_atom_mask.data) + token_pos
                    : nullptr;

            ExecutionBlockStepOutput last_step_diag;
            for (int step = 0; step < exec_K; ++step) {
                ExecutionBlockStepOutput step_diag;
                exec_block->executeStep(
                    hidden, gen.exec_memory,
                    reinterpret_cast<const int*>(row_atom_view.atom_positions.data),
                    row_atom_view.num_atoms,
                    decode_payload, decode_bindings, 0,
                    step, T, stream,
                    &step_diag,
                    gen.trace_state_by_row[0],
                    gen.execution_trace_by_row[0]);
                gen.execution_trace_by_row[0].push_back(step_diag.record);
                if (step == exec_K - 1) {
                    last_step_diag = std::move(step_diag);
                }
            }

        }

        // ── ExecutionBlock: cross-attention read at layers >= exec_layer ──
        if (exec_block_active && layer_idx >= exec_layer) {
            exec_block->crossAttentionRead(
                hidden, gen.exec_memory,
                1, stream, 0, 1);
        }
    }

    // ── Decode-time slot selector: evaluate selection before LM head ──
    {
        DecodeTimeResolveResult sel = resolveDecodeTimeNumSlotSelectionOrMask(
            getDecodeTimeSlotSelectorLayer(),
            getDecodeTimeNumPolicy(),
            cfg.selector_enabled,
            exec_block_active,
            gen.has_exec_memory,
            gen.exec_memory,
            hidden.data,
            stream,
            ts.cublas_handle.get());
        gen.decode_selector.valid          = sel.valid;
        gen.decode_selector.status         = static_cast<uint8_t>(sel.status);
        gen.decode_selector.selected_slot  = sel.selected_slot;
        gen.decode_selector.selected_value = sel.selected_value;
    }

    // ── Step 4: LM Head (RMSNorm → optional centering → W @ hidden^T) ──
    LMHeadLayer* lm_head = getLmHeadLayer();
    if (!lm_head) {
        throw std::runtime_error("executeDecodeForward_: LM head is NULL");
    }
    if (!ts.cached_seq_lengths_tensor.data) {
        throw std::runtime_error("executeDecodeForward_: cached_seq_lengths_tensor.data is NULL");
    }
    const int decode_seq_len = 1;
    cudaError_t len_cp = cudaMemcpyAsync(ts.cached_seq_lengths_tensor.data,
                                         &decode_seq_len,
                                         sizeof(int),
                                         cudaMemcpyHostToDevice,
                                         stream);
    if (len_cp != cudaSuccess) {
        throw std::runtime_error("executeDecodeForward_: cudaMemcpyAsync(seq_len) failed: " +
                                 std::string(cudaGetErrorString(len_cp)));
    }

    Tensor centered_hidden;
    Tensor logits_tensor = lm_head->forward(
        hidden,
        centered_hidden,
        reinterpret_cast<int*>(ts.cached_seq_lengths_tensor.data),
        1,
        1,
        stream,
        ts.cublas_handle.get());

    // ── Step 5: Extract logits to host ──
    Vector logits(cfg.vocab_size);
    if (!logits_tensor.data) {
        throw std::runtime_error("executeDecodeForward_: logits_tensor.data is NULL after LM head");
    }
    // logits_tensor is [1, vocab_size] — copy the single row
    cudaMemcpyAsync(logits.data.data(), logits_tensor.data,
                    cfg.vocab_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return logits;
}

//======================================================//
//  ensureKVCacheAllocated - Allocate KV cache + decode scratch
//  buffers if they haven't been allocated yet.
//
//  During pure inference, initInferenceState() handles this.
//  During training-time sample generation, callers must explicitly request
//  generation state allocation through this helper before forwardInit/forwardStep.
//
//  Safe to call repeatedly — early-returns if already allocated.
//  Cleanup is already handled by TrainingState::cleanup().
//======================================================//
void LanguageModel::ensureKVCacheAllocated() {
    if (!training_state_.initialized) {
        throw std::runtime_error("ensureKVCacheAllocated: training state not initialized");
    }

    // Already allocated? Nothing to do.
    auto& gen = generation_state_;
    if (gen.kvReady()) {
        return;
    }

    const auto& cfg = getConfig();
    if (cfg.num_kv_heads <= 0) {
        throw std::runtime_error("ensureKVCacheAllocated: cfg.num_kv_heads must be > 0");
    }
    if (!HyperParameters::isValidGQAConfig(cfg.num_heads, cfg.num_kv_heads)) {
        throw std::runtime_error("ensureKVCacheAllocated: invalid GQA config num_heads=" +
            std::to_string(cfg.num_heads) + " num_kv_heads=" + std::to_string(cfg.num_kv_heads));
    }
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_dim = cfg.head_dim;
    const int n_layers = cfg.num_layers;
    const int kv_cap = std::min(cfg.max_seq_len, cfg.max_cached_seq_len);

    if (kv_cap <= 0) {
        throw std::runtime_error("ensureKVCacheAllocated: config provides no positive generation KV capacity");
    }

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();

    // ---- Per-layer KV cache (BF16, BSHD layout for FlashAttention) ----
    const size_t kv_elems_per_layer = static_cast<size_t>(num_kv_heads) * kv_cap * head_dim;
    const size_t kv_bytes_per_layer = kv_elems_per_layer * sizeof(__nv_bfloat16);

    gen.kv_cache.shape.num_layers = n_layers;
    gen.kv_cache.shape.num_kv_heads = num_kv_heads;
    gen.kv_cache.shape.head_dim = head_dim;
    gen.kv_cache.shape.capacity_tokens = kv_cap;
    gen.kv_cache.k.resize(n_layers);
    gen.kv_cache.v.resize(n_layers);

    for (int l = 0; l < n_layers; ++l) {
        gen.kv_cache.k[l].allocate(kv_bytes_per_layer, "kv_cache_k");
        gen.kv_cache.v[l].allocate(kv_bytes_per_layer, "kv_cache_v");
        cudaMemsetAsync(gen.kv_cache.k[l], 0, kv_bytes_per_layer, stream);
        cudaMemsetAsync(gen.kv_cache.v[l], 0, kv_bytes_per_layer, stream);
    }

    // ---- Softmax LSE scratch: [num_heads, generation KV capacity rounded] ----
    const int kv_cap_rounded = ((kv_cap + 127) / 128) * 128;
    const size_t lse_bytes = static_cast<size_t>(cfg.num_heads) * kv_cap_rounded * sizeof(float);
    gen.kv_cache.softmax_lse.allocate(lse_bytes, "kv_cache_lse");
    cudaMemsetAsync(gen.kv_cache.softmax_lse, 0, lse_bytes, stream);

    // ---- Decode scratch buffers (tiny, reused per layer per decode step) ----
    const size_t q_bf16_bytes = static_cast<size_t>(cfg.num_heads) * head_dim * sizeof(__nv_bfloat16);
    gen.decode_scratch.q_bf16.allocate(q_bf16_bytes, "decode_q_bf16");
    gen.decode_scratch.attn_out_bf16.allocate(q_bf16_bytes, "decode_attn_out_bf16");
    gen.decode_scratch.attn_out_fp32.allocate(static_cast<size_t>(cfg.d_model) * sizeof(float), "decode_attn_out_fp32");

    // ---- Single-token buffers for incremental generation ----
    using TC = TensorContract::TensorShape;
    if (!gen.single_token_embedding.data) {
        gen.single_token_embedding = Tensor::zeros(
            TC::make_BSM(1, cfg.d_model), false, stream, "single_token_embedding_kv");
    }
    if (!gen.single_token_hidden.data) {
        gen.single_token_hidden = Tensor::zeros(
            TC::make_BSM(1, cfg.d_model), false, stream, "single_token_hidden_kv");
    }
    if (!gen.single_token_logits.data) {
        gen.single_token_logits = Tensor::zeros(
            TC::make_BSM(1, cfg.vocab_size), false, stream, "single_token_logits_kv");
    }

    const size_t total_kv_bytes = n_layers * 2 * kv_bytes_per_layer + lse_bytes;
    std::cout << "[ensureKVCacheAllocated] Allocated KV cache: " << n_layers << " layers, "
              << (total_kv_bytes / 1024.0 / 1024.0) << " MB (BF16)" << std::endl;
}

void LanguageModel::resetKVCache() {
    if (training_state_.initialized) {
        auto& gen = generation_state_;
        gen.resetSession();

        // Zero BF16 KV cache buffers using GenerationState's shaped owner.
        const size_t kv_bytes_per_layer = gen.kv_cache.bytesPerLayer();
        cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();

        for (size_t l = 0; l < gen.kv_cache.k.size(); ++l) {
            if (gen.kv_cache.k[l])
                cudaMemsetAsync(gen.kv_cache.k[l], 0, kv_bytes_per_layer, stream);
            if (gen.kv_cache.v[l])
                cudaMemsetAsync(gen.kv_cache.v[l], 0, kv_bytes_per_layer, stream);
        }
    }
}

//======================================================//
//  getKVCacheLength - Query current sequence length
//======================================================//
int LanguageModel::getKVCacheLength() const {
    return generation_state_.kv_cache_len;
}

} // namespace GRIM
