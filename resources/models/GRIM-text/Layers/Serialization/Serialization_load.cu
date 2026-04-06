#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Common/grim_model_serialization_version.hpp"
#include "Serialization_GPU.hpp"

namespace {

constexpr auto kLogModule = GRIM::Logging::ModuleId::Checkpoint;

template <typename... Args>
std::string Msg(Args&&... args) {
    std::ostringstream oss;
    (oss << ... << args);
    return oss.str();
}

} // namespace

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                                             \
    do {                                                                                                             \
        cudaError_t err__ = (call);                                                                                  \
        if (err__ != cudaSuccess) {                                                                                  \
            GRIM::Logging::EmitModuleError(kLogModule, Msg("[CUDA] ", __FILE__, ':', __LINE__, ": ",                 \
                                         cudaGetErrorString(err__)));                                                \
            throw std::runtime_error("CUDA failure");                                                                 \
        }                                                                                                            \
    } while (0)
#endif

namespace GRIM {

SerializationLayer::SerializationLayer(SerializationConfig config) : config_(std::move(config)) {}

void SerializationLayer::setConfig(const SerializationConfig& config) {
    config_ = config;
}

bool SerializationLayer::load(SerializationLoadRequest& request) {
    const auto& cfg = request.config;
    const auto& req = request.capabilities;

    if (cfg.vocab_size <= 0 || cfg.d_model <= 0 || cfg.num_layers <= 0) {
        Logging::EmitModuleError(kLogModule, "[load] Invalid model dimensions");
        return false;
    }
    if (request.encoder_layers.size() != static_cast<std::size_t>(cfg.num_layers)) {
        Logging::EmitModuleError(kLogModule, Msg("[load] Encoder layer mismatch (expected ",
                                            cfg.num_layers, ", got ", request.encoder_layers.size(), ")"));
        return false;
    }

    // ─── Step 1: Read file ───
    std::ifstream file(request.path, std::ios::binary | std::ios::ate);
    if (!file) {
        Logging::EmitModuleError(kLogModule, Msg("[load] Failed to open: ", request.path));
        return false;
    }
    const auto file_size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(file_size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
        Logging::EmitModuleError(kLogModule, "[load] Read failed");
        return false;
    }
    file.close();

    // ─── Step 2: Verify FlatBuffer ───
    auto emitFlatBufferLoadDiag = [&](const std::string& issue) {
        Logging::EmitModuleError(kLogModule, "[load] FlatBuffer: " + issue);
    };
    std::vector<std::string> issues;

    if (file_size < 8) {
        issues.push_back("buffer too small (need >= 8 bytes), got " + std::to_string(file_size));
    }
    if (file_size >= 4) {
        uint32_t first4 = 0;
        std::memcpy(&first4, buffer.data(), 4);
        if (first4 == 0x474D5254u) {
            issues.push_back("first 4 bytes = 0x474D5254 (GRMT magic) — looks like training data .grmt, not a checkpoint");
        }
        if (first4 >= file_size) {
            issues.push_back("root_offset=" + std::to_string(first4) +
                             " points outside buffer (size=" + std::to_string(file_size) + ")");
        }
    }
    if (file_size >= 8) {
        const char* expected_id = "GRMT";
        const char* file_id = reinterpret_cast<const char*>(buffer.data()) + 4;
        if (std::memcmp(file_id, expected_id, 4) != 0) {
            std::ostringstream ss;
            char observed_id[5] = { file_id[0], file_id[1], file_id[2], file_id[3], '\0' };
            uint32_t observed_raw = 0;
            std::memcpy(&observed_raw, file_id, 4);
            ss << "file_identifier wrong: expected \"GRMT\", got \""
               << observed_id << "\" (0x" << std::hex << observed_raw << std::dec << ")";
            issues.push_back(ss.str());
        }
    }

    flatbuffers::Verifier verifier(buffer.data(), buffer.size());
    if (!GRIMTransformer::VerifyTransformerModelBuffer(verifier)) {
        emitFlatBufferLoadDiag("verification failed");
        for (const auto& s : issues) emitFlatBufferLoadDiag(s);
        // Scan for zero-filled regions which indicate incomplete filesystem flush
        if (file_size >= 4096) {
            // Check if the last 4KB is all zeros (strong indicator of unflushed write)
            bool tail_all_zero = true;
            for (std::size_t i = file_size - 4096; i < file_size; ++i) {
                if (buffer[i] != 0) { tail_all_zero = false; break; }
            }
            if (tail_all_zero) {
                emitFlatBufferLoadDiag("last 4096 bytes are all zeros — likely incomplete filesystem flush (Lustre/NFS)");
            }
            // Check middle region for large zero spans
            std::size_t mid = file_size / 2;
            bool mid_zero = true;
            for (std::size_t i = mid; i < std::min(mid + 4096, file_size); ++i) {
                if (buffer[i] != 0) { mid_zero = false; break; }
            }
            if (mid_zero) {
                emitFlatBufferLoadDiag("middle region contains 4096+ zero bytes — likely partial write corruption");
            }
        }
        emitFlatBufferLoadDiag("file_size=" + std::to_string(file_size) +
                               " path=" + request.path + " — structure invalid");
        return false;
    }

    const auto* model_fb = GRIMTransformer::GetTransformerModel(buffer.data());
    if (!model_fb) {
        Logging::EmitModuleError(kLogModule, "[load] Failed to parse FlatBuffer");
        return false;
    }

    // ─── Step 3: Version check ───
    if (model_fb->version() != GRIM_MODEL_VERSION) {
        Logging::EmitModuleError(kLogModule, Msg("[load] Version mismatch (",
                                            model_fb->version(), " != ", GRIM_MODEL_VERSION, ")"));
        return false;
    }

    // ─── Step 4: VALIDATE — before any GPU writes ───
    if (!validate_checkpoint_capabilities(model_fb, cfg, req, request)) {
        return false;
    }

    Logging::EmitModuleInfo(kLogModule, Msg("[load] GQA: num_heads=",
        cfg.num_heads, " num_kv_heads=", cfg.num_kv_heads));

    // ─── Step 5 would be: if false → return (handled above) ───

    // ─── Step 6: LOAD — deterministic upload, gated by requires_* only ───

    auto upload_device_vector = [](const std::vector<float>& host,
                                   const DeviceWriteView& view,
                                   const char* label) -> bool {
        if (host.empty()) return true;
        if (!view.ptr) {
            Logging::EmitModuleError(kLogModule, Msg("[load] Missing destination for ", label));
            return false;
        }
        if (view.count != host.size()) {
            Logging::EmitModuleError(kLogModule, Msg("[load] Size mismatch for ", label,
                                                " (dest=", view.count, ", src=", host.size(), ")"));
            return false;
        }
        CUDA_CHECK(cudaMemcpy(view.ptr, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
        return true;
    };

    // ─── Embeddings ───
    const auto* fb_embeddings = model_fb->embeddings();
    const auto* fb_token_vec = fb_embeddings->token_embeddings();
    std::vector<float> token_host(fb_token_vec->begin(), fb_token_vec->end());
    const int vocab_size = static_cast<int>(fb_embeddings->vocab_size());
    const int d_model = static_cast<int>(fb_embeddings->d_model());

    if (cfg.use_gpu) {
        if (!request.gpu_embedding.token_embeddings.ptr) {
            Logging::EmitModuleError(kLogModule, "[load] GPU embedder not initialized");
            return false;
        }
        if (!upload_device_vector(token_host, request.gpu_embedding.token_embeddings, "token embeddings"))
            return false;
    }
    if (request.cpu_embedding.set_tokens) {
        request.cpu_embedding.set_tokens(token_host, vocab_size, d_model);
    }

    if (fb_embeddings->use_rms_norm() && fb_embeddings->rms_gamma()) {
        std::vector<float> rms_gamma_host(fb_embeddings->rms_gamma()->begin(), fb_embeddings->rms_gamma()->end());
        if (cfg.use_gpu && request.gpu_embedding.has_rms_norm) {
            if (!upload_device_vector(rms_gamma_host, request.gpu_embedding.rms_gamma, "embedding rms gamma"))
                return false;
        }
        if (request.cpu_embedding.set_rms_gamma) {
            request.cpu_embedding.set_rms_gamma(rms_gamma_host);
        }
    }

    // ─── Encoder layers ───
    const auto* fb_layers = model_fb->encoder_layers();
    for (int layer_idx = 0; layer_idx < cfg.num_layers; ++layer_idx) {
        const auto* fb_layer = fb_layers->Get(layer_idx);
        const auto& layer_view = request.encoder_layers[layer_idx];

        const auto* fb_attn = fb_layer->attention();
        std::vector<float> h_W_qkv(fb_attn->w_qkv_data()->begin(), fb_attn->w_qkv_data()->end());
        std::vector<float> h_b_qkv(fb_attn->b_qkv_data()->begin(), fb_attn->b_qkv_data()->end());
        std::vector<float> h_W_o(fb_attn->w_o_data()->begin(), fb_attn->w_o_data()->end());
        std::vector<float> h_b_o(fb_attn->b_o_data()->begin(), fb_attn->b_o_data()->end());
        if (!upload_device_vector(h_W_qkv, layer_view.attn_w_qkv, "attn.W_qkv") ||
            !upload_device_vector(h_b_qkv, layer_view.attn_b_qkv, "attn.b_qkv") ||
            !upload_device_vector(h_W_o, layer_view.attn_w_o, "attn.W_o") ||
            !upload_device_vector(h_b_o, layer_view.attn_b_o, "attn.b_o"))
            return false;

        const auto* fb_ffn = fb_layer->ffn();
        std::vector<float> h_W_gate;
        if (fb_ffn->w_gate_data())
            h_W_gate.assign(fb_ffn->w_gate_data()->begin(), fb_ffn->w_gate_data()->end());
        std::vector<float> h_W1(fb_ffn->w1_data()->begin(), fb_ffn->w1_data()->end());
        std::vector<float> h_W2(fb_ffn->w2_data()->begin(), fb_ffn->w2_data()->end());
        std::vector<float> h_b2(fb_ffn->b2_data()->begin(), fb_ffn->b2_data()->end());
        if ((!h_W_gate.empty() && !upload_device_vector(h_W_gate, layer_view.ffn_w_gate, "ffn.W_gate")) ||
            !upload_device_vector(h_W1, layer_view.ffn_w1, "ffn.W1") ||
            !upload_device_vector(h_W2, layer_view.ffn_w2, "ffn.W2") ||
            !upload_device_vector(h_b2, layer_view.ffn_b2, "ffn.b2"))
            return false;

        std::vector<float> h_rms1(fb_layer->rms1()->gamma()->begin(), fb_layer->rms1()->gamma()->end());
        std::vector<float> h_rms2(fb_layer->rms2()->gamma()->begin(), fb_layer->rms2()->gamma()->end());
        if (!upload_device_vector(h_rms1, layer_view.rms1_gamma, "rms1.gamma") ||
            !upload_device_vector(h_rms2, layer_view.rms2_gamma, "rms2.gamma"))
            return false;

        if (layer_view.layer_scale1.ptr) {
            std::vector<float> h_ls1(fb_layer->layer_scale1()->begin(), fb_layer->layer_scale1()->end());
            if (!upload_device_vector(h_ls1, layer_view.layer_scale1, "layer_scale1"))
                return false;
        }
        if (layer_view.layer_scale2.ptr) {
            std::vector<float> h_ls2(fb_layer->layer_scale2()->begin(), fb_layer->layer_scale2()->end());
            if (!upload_device_vector(h_ls2, layer_view.layer_scale2, "layer_scale2"))
                return false;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // ─── LM head ───
    const auto* fb_lm_head = model_fb->lm_head();
    if (fb_lm_head) {
        if (request.lm_head.projection.ptr && fb_lm_head->projection_data()) {
            std::vector<float> lm_proj_host(fb_lm_head->projection_data()->begin(), fb_lm_head->projection_data()->end());
            if (!upload_device_vector(lm_proj_host, request.lm_head.projection, "LM head projection"))
                return false;
        } else if (!cfg.tie_embeddings) {
            Logging::EmitModuleError(kLogModule, "[load] LM head projection missing (tie_embeddings=false)");
            return false;
        }

        if (cfg.use_bias && request.lm_head.bias.ptr && fb_lm_head->bias_data()) {
            std::vector<float> lm_bias_host(fb_lm_head->bias_data()->begin(), fb_lm_head->bias_data()->end());
            if (!upload_device_vector(lm_bias_host, request.lm_head.bias, "LM head bias"))
                return false;
        } else if (cfg.use_bias) {
            Logging::EmitModuleError(kLogModule, "[load] LM head bias missing (use_bias=true)");
            return false;
        }
    }

    // ─── ScratchBlock (gated by requires_scratch_block) ───
    if (req.requires_scratch_block) {
        const auto* fb_sb = model_fb->scratch_block();
        if (request.scratch_block.atom_type_embeddings.ptr && fb_sb->atom_type_embeddings()) {
            std::vector<float> sb_ate(fb_sb->atom_type_embeddings()->begin(), fb_sb->atom_type_embeddings()->end());
            if (!upload_device_vector(sb_ate, request.scratch_block.atom_type_embeddings, "ScratchBlock atom_type_embeddings"))
                return false;
        }
        if (request.scratch_block.atom_projection.ptr && fb_sb->atom_projection()) {
            std::vector<float> sb_ap(fb_sb->atom_projection()->begin(), fb_sb->atom_projection()->end());
            if (!upload_device_vector(sb_ap, request.scratch_block.atom_projection, "ScratchBlock atom_projection"))
                return false;
        }
        request.report.scratch_block_loaded = true;
        Logging::EmitModuleInfo(kLogModule, Msg("[load] ScratchBlock: atom_types=",
            fb_sb->num_atom_types(), " atom_dim=", fb_sb->atom_embedding_dim()));
    }

    // ─── ReasoningHead (gated by requires_reasoning_head) ───
    if (req.requires_reasoning_head) {
        const auto* fb_rh = model_fb->reasoning_head();
        std::vector<float> rh_w_op(fb_rh->w_op_data()->begin(), fb_rh->w_op_data()->end());
        if (!upload_device_vector(rh_w_op, request.reasoning_head.w_op, "ReasoningHead W_op"))
            return false;
        std::vector<float> rh_b_op(fb_rh->b_op_data()->begin(), fb_rh->b_op_data()->end());
        if (!upload_device_vector(rh_b_op, request.reasoning_head.b_op, "ReasoningHead b_op"))
            return false;
        std::vector<float> rh_w_arg1(fb_rh->w_arg1_data()->begin(), fb_rh->w_arg1_data()->end());
        if (!upload_device_vector(rh_w_arg1, request.reasoning_head.w_arg1, "ReasoningHead w_arg1"))
            return false;
        std::vector<float> rh_w_arg2(fb_rh->w_arg2_data()->begin(), fb_rh->w_arg2_data()->end());
        if (!upload_device_vector(rh_w_arg2, request.reasoning_head.w_arg2, "ReasoningHead w_arg2"))
            return false;
        request.report.reasoning_head_loaded = true;
        Logging::EmitModuleInfo(kLogModule, Msg("[load] ReasoningHead: num_ops=",
            fb_rh->num_ops(), " d_total=", fb_rh->d_total()));
    }

    // ─── ExecutionBlock (gated by requires_execution_block) ───
    if (req.requires_execution_block) {
        const auto* fb_eb = model_fb->execution_block();
        auto ul = [&](const flatbuffers::Vector<float>* src, const DeviceWriteView& dst, const char* name) -> bool {
            if (!src) return true;  // Field absent in checkpoint — skip (e.g., older schema)
            std::vector<float> buf(src->begin(), src->end());
            return upload_device_vector(buf, dst, name);
        };
        const auto& eb = request.execution_block;
        bool eb_ok = true;
        eb_ok = eb_ok && ul(fb_eb->w_decode_1_data(), eb.w_decode_1, "EB w_decode_1");
        eb_ok = eb_ok && ul(fb_eb->b_decode_1_data(), eb.b_decode_1, "EB b_decode_1");
        eb_ok = eb_ok && ul(fb_eb->w_decode_2_data(), eb.w_decode_2, "EB w_decode_2");
        eb_ok = eb_ok && ul(fb_eb->w_arg1_select_data(), eb.w_arg1_select, "EB w_arg1_select");
        eb_ok = eb_ok && ul(fb_eb->w_arg2_select_data(), eb.w_arg2_select, "EB w_arg2_select");
        eb_ok = eb_ok && ul(fb_eb->w_op_select_data(), eb.W_op_select, "EB W_op_select");
        eb_ok = eb_ok && ul(fb_eb->w_key_proj_data(), eb.W_key_proj, "EB W_key_proj");
        eb_ok = eb_ok && ul(fb_eb->w_write_query_data(), eb.W_write_query, "EB W_write_query");
        eb_ok = eb_ok && ul(fb_eb->w_write_key_data(), eb.W_write_key, "EB W_write_key");
        eb_ok = eb_ok && ul(fb_eb->alpha_data(), eb.alpha, "EB alpha");
        eb_ok = eb_ok && ul(fb_eb->beta_data(), eb.beta, "EB beta");
        eb_ok = eb_ok && ul(fb_eb->step_embeddings_data(), eb.step_embeddings, "EB step_embeddings");
        eb_ok = eb_ok && ul(fb_eb->type_num_embed_data(), eb.type_num_embed, "EB type_num_embed");
        eb_ok = eb_ok && ul(fb_eb->w_value_to_emb_data(), eb.W_value_to_emb, "EB W_value_to_emb");
        eb_ok = eb_ok && ul(fb_eb->b_value_to_emb_data(), eb.b_value_to_emb, "EB b_value_to_emb");
        eb_ok = eb_ok && ul(fb_eb->w_inject_gate_data(), eb.w_inject_gate, "EB w_inject_gate");
        eb_ok = eb_ok && ul(fb_eb->w_q_read_data(), eb.W_Q_read, "EB W_Q_read");
        eb_ok = eb_ok && ul(fb_eb->w_k_read_data(), eb.W_K_read, "EB W_K_read");
        eb_ok = eb_ok && ul(fb_eb->w_v_read_data(), eb.W_V_read, "EB W_V_read");
        eb_ok = eb_ok && ul(fb_eb->w_o_read_data(), eb.W_O_read, "EB W_O_read");
        eb_ok = eb_ok && ul(fb_eb->w_gate_read_data(), eb.W_gate_read, "EB W_gate_read");
        eb_ok = eb_ok && ul(fb_eb->tau_data(), eb.tau, "EB tau");
        eb_ok = eb_ok && ul(fb_eb->e_slot_data(), eb.E_slot, "EB E_slot");
        eb_ok = eb_ok && ul(fb_eb->e_op_data(), eb.E_op, "EB E_op");
        eb_ok = eb_ok && ul(fb_eb->w_scal_data(), eb.W_scal, "EB W_scal");
        eb_ok = eb_ok && ul(fb_eb->b_scal_data(), eb.b_scal, "EB b_scal");
        eb_ok = eb_ok && ul(fb_eb->w_trace_data(), eb.W_trace, "EB W_trace");
        eb_ok = eb_ok && ul(fb_eb->b_trace_data(), eb.b_trace, "EB b_trace");
        eb_ok = eb_ok && ul(fb_eb->w_reason_gate_data(), eb.W_reason_gate, "EB W_reason_gate");
        if (!eb_ok) return false;
        request.report.execution_block_loaded = true;
        Logging::EmitModuleInfo(kLogModule, "[load] ExecutionBlock v2 weights loaded");
    }

    // ─── DecodeTimeSlotSelector (gated by requires_slot_selector) ───
    if (req.requires_slot_selector) {
        const auto* fb_ss = model_fb->slot_selector();
        if (!fb_ss) {
            Logging::EmitModuleError(kLogModule, "[load] FATAL: SlotSelector required but missing in checkpoint");
            return false;
        }
        auto ul = [&](const flatbuffers::Vector<float>* src, const DeviceWriteView& dst, const char* name) -> bool {
            if (!src) return true;  // Field absent in checkpoint — skip
            std::vector<float> buf(src->begin(), src->end());
            return upload_device_vector(buf, dst, name);
        };
        const auto& ss = request.slot_selector;
        bool ss_ok = true;
        ss_ok = ss_ok && ul(fb_ss->w_q_select_data(), ss.w_q_select, "SS w_q_select");
        ss_ok = ss_ok && ul(fb_ss->w_k_select_data(), ss.w_k_select, "SS w_k_select");
        ss_ok = ss_ok && ul(fb_ss->null_key_select_data(), ss.null_key_select, "SS null_key_select");
        ss_ok = ss_ok && ul(fb_ss->null_logit_bias_data(), ss.null_logit_bias, "SS null_logit_bias");
        if (!ss_ok) return false;
        request.report.slot_selector_loaded = true;
        Logging::EmitModuleInfo(kLogModule, "[load] SlotSelector weights loaded");
    }

    // ─── final_rms_gamma (gated by requires_final_rms_gamma) ───
    if (req.requires_final_rms_gamma) {
        const auto* fb_frg = model_fb->final_rms_gamma();
        std::vector<float> frg_data(fb_frg->begin(), fb_frg->end());
        if (!upload_device_vector(frg_data, request.final_rms_gamma, "final_rms_gamma"))
            return false;
        Logging::EmitModuleInfo(kLogModule, Msg("[load] final_rms_gamma: size=", frg_data.size()));
    }

    // ─── Step 7: Final load verification (safety) ───
    if (req.requires_reasoning_head && !request.report.reasoning_head_loaded) {
        Logging::EmitModuleError(kLogModule, "[load] FATAL: ReasoningHead required but not loaded");
        return false;
    }
    if (req.requires_scratch_block && !request.report.scratch_block_loaded) {
        Logging::EmitModuleError(kLogModule, "[load] FATAL: ScratchBlock required but not loaded");
        return false;
    }
    if (req.requires_execution_block && !request.report.execution_block_loaded) {
        Logging::EmitModuleError(kLogModule, "[load] FATAL: ExecutionBlock required but not loaded");
        return false;
    }
    if (req.requires_slot_selector && !request.report.slot_selector_loaded) {
        Logging::EmitModuleError(kLogModule, "[load] FATAL: SlotSelector required but not loaded");
        return false;
    }

    Logging::EmitModuleInfo(kLogModule, "[load] Model loaded successfully");
    return true;
}

} // namespace GRIM
