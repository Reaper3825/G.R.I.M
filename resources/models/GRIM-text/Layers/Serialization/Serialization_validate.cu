#include "Serialization_validate.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"

#include <sstream>
#include <string>

namespace {

constexpr auto kLogModule = GRIM::Logging::ModuleId::Checkpoint;

template <typename... Args>
std::string Msg(Args&&... args) {
    std::ostringstream oss;
    (oss << ... << args);
    return oss.str();
}

bool check_fb_vec_size(const flatbuffers::Vector<float>* vec,
                       std::size_t expected,
                       const char* field_name) {
    if (!vec) {
        GRIM::Logging::EmitModuleError(kLogModule,
            Msg("[load] FATAL: missing required checkpoint field: ", field_name));
        return false;
    }
    if (static_cast<std::size_t>(vec->size()) != expected) {
        GRIM::Logging::EmitModuleError(kLogModule,
            Msg("[load] FATAL: ", field_name, " size mismatch: checkpoint=",
                vec->size(), " expected=", expected));
        return false;
    }
    return true;
}

bool cross_check_view(const GRIM::DeviceWriteView& view,
                      std::size_t contract_numel,
                      const char* field_name) {
    if (!view.ptr) return true;
    if (view.count != contract_numel) {
        GRIM::Logging::EmitModuleError(kLogModule,
            Msg("[load] FATAL: DeviceWriteView.count drift for ", field_name,
                ": view=", view.count, " contract=", contract_numel,
                " — model allocation does not match tensor contract"));
        return false;
    }
    return true;
}

} // namespace

namespace GRIM {

bool validate_checkpoint_capabilities(
    const GRIMTransformer::TransformerModel* model_fb,
    const SerializationModelConfigView& cfg,
    const CheckpointCapabilityRequirements& req,
    const SerializationLoadRequest& load_req)
{
    bool ok = true;

    const auto* fb_config = model_fb->config();
    if (!fb_config) {
        Logging::EmitModuleError(kLogModule, "[load] FATAL: checkpoint has no config table");
        return false;
    }

    // ─── Core dimension contract ───
    if (static_cast<int>(fb_config->vocab_size()) != cfg.vocab_size) {
        Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: vocab_size mismatch: checkpoint=",
            fb_config->vocab_size(), " model=", cfg.vocab_size));
        ok = false;
    }
    if (static_cast<int>(fb_config->d_model()) != cfg.d_model) {
        Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: d_model mismatch: checkpoint=",
            fb_config->d_model(), " model=", cfg.d_model));
        ok = false;
    }
    if (static_cast<int>(fb_config->num_layers()) != cfg.num_layers) {
        Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: num_layers mismatch: checkpoint=",
            fb_config->num_layers(), " model=", cfg.num_layers));
        ok = false;
    }
    if (static_cast<int>(fb_config->num_heads()) != cfg.num_heads) {
        Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: num_heads mismatch: checkpoint=",
            fb_config->num_heads(), " model=", cfg.num_heads));
        ok = false;
    }
    if (static_cast<int>(fb_config->num_kv_heads()) != cfg.num_kv_heads) {
        Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: num_kv_heads mismatch: checkpoint=",
            fb_config->num_kv_heads(), " model=", cfg.num_kv_heads));
        ok = false;
    }
    if (!ok) return false;

    // ─── Contract-derived sizes (TensorContract rules) ───
    const std::size_t d_model = static_cast<std::size_t>(cfg.d_model);
    const std::size_t d_ff    = static_cast<std::size_t>(cfg.d_ff);
    const int hd = cfg.head_dim();
    const std::size_t total_qkv_dim = static_cast<std::size_t>(cfg.total_qkv_dim());
    const std::size_t qkv_weight_size = total_qkv_dim * d_model;
    const std::size_t embed_numel = static_cast<std::size_t>(cfg.vocab_size) * d_model;
    (void)hd;

    // ─── Embeddings (always required) ───
    const auto* fb_emb = model_fb->embeddings();
    if (!fb_emb) {
        Logging::EmitModuleError(kLogModule, "[load] FATAL: embeddings table missing");
        return false;
    }
    if (!fb_emb->token_embeddings() ||
        static_cast<std::size_t>(fb_emb->token_embeddings()->size()) != embed_numel) {
        Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: token_embeddings size mismatch: expected=",
            embed_numel, " checkpoint=",
            fb_emb->token_embeddings() ? fb_emb->token_embeddings()->size() : 0u));
        return false;
    }

    // ─── Encoder layers (always required) ───
    const auto* fb_layers = model_fb->encoder_layers();
    if (!fb_layers || static_cast<int>(fb_layers->size()) != cfg.num_layers) {
        Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: encoder_layers count mismatch: expected=",
            cfg.num_layers, " checkpoint=", fb_layers ? static_cast<int>(fb_layers->size()) : 0));
        return false;
    }
    for (int i = 0; i < cfg.num_layers; ++i) {
        const auto* fl = fb_layers->Get(i);
        if (!fl || !fl->attention() || !fl->ffn() || !fl->rms1() || !fl->rms2()) {
            Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: encoder layer ", i, " missing sub-tables"));
            return false;
        }
        const auto* fa = fl->attention();
        if (!check_fb_vec_size(fa->w_qkv_data(), qkv_weight_size, "attn.W_qkv") ||
            !check_fb_vec_size(fa->b_qkv_data(), total_qkv_dim, "attn.b_qkv") ||
            !check_fb_vec_size(fa->w_o_data(), d_model * d_model, "attn.W_o") ||
            !check_fb_vec_size(fa->b_o_data(), d_model, "attn.b_o")) {
            Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: attention size error in layer ", i));
            return false;
        }
        const auto* ff = fl->ffn();
        if (!check_fb_vec_size(ff->w1_data(), d_model * d_ff, "ffn.W1") ||
            !check_fb_vec_size(ff->w2_data(), d_ff * d_model, "ffn.W2") ||
            !check_fb_vec_size(ff->b2_data(), d_model, "ffn.b2")) {
            Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: FFN size error in layer ", i));
            return false;
        }
        if (!check_fb_vec_size(fl->rms1()->gamma(), d_model, "rms1.gamma") ||
            !check_fb_vec_size(fl->rms2()->gamma(), d_model, "rms2.gamma")) {
            Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: RMSNorm gamma size error in layer ", i));
            return false;
        }
        // LayerScale: if model destination exists, checkpoint must provide it
        if (i < static_cast<int>(load_req.encoder_layers.size())) {
            const auto& lv = load_req.encoder_layers[i];
            if (lv.layer_scale1.ptr) {
                if (!fl->layer_scale1() || fl->layer_scale1()->size() == 0) {
                    Logging::EmitModuleError(kLogModule,
                        Msg("[load] FATAL: layer_scale1 required for layer ", i,
                            " but missing in checkpoint"));
                    return false;
                }
            }
            if (lv.layer_scale2.ptr) {
                if (!fl->layer_scale2() || fl->layer_scale2()->size() == 0) {
                    Logging::EmitModuleError(kLogModule,
                        Msg("[load] FATAL: layer_scale2 required for layer ", i,
                            " but missing in checkpoint"));
                    return false;
                }
            }
        }
    }

    // ─── NumericHead (removed from runtime; reject legacy checkpoints that carry weights) ───
    if (const auto* fb_nh = model_fb->numeric_head()) {
        const bool has_weights = fb_nh->projection_data() && fb_nh->projection_data()->size() > 0;
        if (has_weights) {
            Logging::EmitModuleError(kLogModule,
                "[load] FATAL: checkpoint contains NumericHead weights; this build removed NumericHead "
                "(execution-first). Train or export a checkpoint without NumericHead.");
            return false;
        }
    }

    // ─── ReasoningHead ───
    if (req.requires_reasoning_head) {
        const auto* fb_rh = model_fb->reasoning_head();
        if (!fb_rh) {
            Logging::EmitModuleError(kLogModule, "[load] FATAL: ReasoningHead required but missing in checkpoint");
            return false;
        }
        const int nops = static_cast<int>(fb_rh->num_ops());
        const int dt   = static_cast<int>(fb_rh->d_total());
        if (load_req.reasoning_head.num_ops != nops || load_req.reasoning_head.d_total != dt) {
            Logging::EmitModuleError(kLogModule,
                Msg("[load] FATAL: ReasoningHead shape mismatch: checkpoint num_ops=", nops,
                    " d_total=", dt, " model num_ops=", load_req.reasoning_head.num_ops,
                    " d_total=", load_req.reasoning_head.d_total));
            return false;
        }
        const std::size_t wop_sz = static_cast<std::size_t>(nops) * dt;
        if (!check_fb_vec_size(fb_rh->w_op_data(), wop_sz, "ReasoningHead W_op")) return false;
        if (!check_fb_vec_size(fb_rh->b_op_data(), static_cast<std::size_t>(nops), "ReasoningHead b_op")) return false;
        if (!check_fb_vec_size(fb_rh->w_arg1_data(), static_cast<std::size_t>(dt), "ReasoningHead w_arg1")) return false;
        if (!check_fb_vec_size(fb_rh->w_arg2_data(), static_cast<std::size_t>(dt), "ReasoningHead w_arg2")) return false;
        if (!cross_check_view(load_req.reasoning_head.w_op, wop_sz, "ReasoningHead W_op")) return false;
    }

    // ─── ScratchBlock ───
    if (req.requires_scratch_block) {
        const auto* fb_sb = model_fb->scratch_block();
        if (!fb_sb) {
            Logging::EmitModuleError(kLogModule, "[load] FATAL: ScratchBlock required but missing in checkpoint");
            return false;
        }
        if (!fb_sb->enabled()) {
            Logging::EmitModuleError(kLogModule, "[load] FATAL: ScratchBlock required but checkpoint has enabled=false");
            return false;
        }
        if (fb_sb->atom_type_embeddings()) {
            const std::size_t ate_numel = static_cast<std::size_t>(fb_sb->num_atom_types()) *
                                          static_cast<std::size_t>(fb_sb->atom_embedding_dim());
            if (static_cast<std::size_t>(fb_sb->atom_type_embeddings()->size()) != ate_numel) {
                Logging::EmitModuleError(kLogModule,
                    Msg("[load] FATAL: ScratchBlock atom_type_embeddings size mismatch: checkpoint=",
                        fb_sb->atom_type_embeddings()->size(), " expected=", ate_numel));
                return false;
            }
            if (!cross_check_view(load_req.scratch_block.atom_type_embeddings, ate_numel,
                                  "ScratchBlock atom_type_embeddings")) return false;
        }
        if (fb_sb->atom_projection()) {
            const std::size_t ap_expected = static_cast<std::size_t>(fb_sb->atom_embedding_dim()) * d_model;
            if (static_cast<std::size_t>(fb_sb->atom_projection()->size()) != ap_expected) {
                Logging::EmitModuleError(kLogModule,
                    Msg("[load] FATAL: ScratchBlock atom_projection size mismatch: checkpoint=",
                        fb_sb->atom_projection()->size(), " expected=", ap_expected));
                return false;
            }
            if (!cross_check_view(load_req.scratch_block.atom_projection, ap_expected,
                                  "ScratchBlock atom_projection")) return false;
        }
    }

    // ─── ExecutionBlock ───
    if (req.requires_execution_block) {
        const auto* fb_eb = model_fb->execution_block();
        if (!fb_eb) {
            Logging::EmitModuleError(kLogModule, "[load] FATAL: ExecutionBlock required but missing in checkpoint");
            return false;
        }
        // Each EB field must exist and match the model's Tensor::numel() via DeviceWriteView.count
        auto eb_field = [&](const flatbuffers::Vector<float>* src, const DeviceWriteView& dst, const char* name) -> bool {
            if (!src) {
                Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: missing required EB field: ", name));
                return false;
            }
            if (!dst.ptr) {
                Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: null model destination for EB field: ", name));
                return false;
            }
            if (static_cast<std::size_t>(src->size()) != dst.count) {
                Logging::EmitModuleError(kLogModule,
                    Msg("[load] FATAL: EB field ", name, " size mismatch: checkpoint=",
                        src->size(), " model_numel=", dst.count));
                return false;
            }
            return true;
        };
        const auto& eb = load_req.execution_block;
        ok = true;
        ok = ok && eb_field(fb_eb->w_decode_1_data(), eb.w_decode_1, "EB w_decode_1");
        ok = ok && eb_field(fb_eb->b_decode_1_data(), eb.b_decode_1, "EB b_decode_1");
        ok = ok && eb_field(fb_eb->w_decode_2_data(), eb.w_decode_2, "EB w_decode_2");
        ok = ok && eb_field(fb_eb->w_arg1_select_data(), eb.w_arg1_select, "EB w_arg1_select");
        ok = ok && eb_field(fb_eb->w_arg2_select_data(), eb.w_arg2_select, "EB w_arg2_select");
        ok = ok && eb_field(fb_eb->w_op_select_data(), eb.W_op_select, "EB W_op_select");
        ok = ok && eb_field(fb_eb->w_key_proj_data(), eb.W_key_proj, "EB W_key_proj");
        ok = ok && eb_field(fb_eb->w_write_query_data(), eb.W_write_query, "EB W_write_query");
        ok = ok && eb_field(fb_eb->w_write_key_data(), eb.W_write_key, "EB W_write_key");
        ok = ok && eb_field(fb_eb->alpha_data(), eb.alpha, "EB alpha");
        ok = ok && eb_field(fb_eb->beta_data(), eb.beta, "EB beta");
        ok = ok && eb_field(fb_eb->step_embeddings_data(), eb.step_embeddings, "EB step_embeddings");
        ok = ok && eb_field(fb_eb->type_num_embed_data(), eb.type_num_embed, "EB type_num_embed");
        ok = ok && eb_field(fb_eb->w_value_to_emb_data(), eb.W_value_to_emb, "EB W_value_to_emb");
        ok = ok && eb_field(fb_eb->b_value_to_emb_data(), eb.b_value_to_emb, "EB b_value_to_emb");
        ok = ok && eb_field(fb_eb->w_inject_gate_data(), eb.w_inject_gate, "EB w_inject_gate");
        ok = ok && eb_field(fb_eb->w_q_read_data(), eb.W_Q_read, "EB W_Q_read");
        ok = ok && eb_field(fb_eb->w_k_read_data(), eb.W_K_read, "EB W_K_read");
        ok = ok && eb_field(fb_eb->w_v_read_data(), eb.W_V_read, "EB W_V_read");
        ok = ok && eb_field(fb_eb->w_o_read_data(), eb.W_O_read, "EB W_O_read");
        ok = ok && eb_field(fb_eb->w_gate_read_data(), eb.W_gate_read, "EB W_gate_read");
        ok = ok && eb_field(fb_eb->tau_data(), eb.tau, "EB tau");
        if (!ok) return false;
    }

    // ─── final_rms_gamma ───
    if (req.requires_final_rms_gamma) {
        const auto* fb_frg = model_fb->final_rms_gamma();
        if (!fb_frg) {
            Logging::EmitModuleError(kLogModule, "[load] FATAL: final_rms_gamma required but missing in checkpoint");
            return false;
        }
        if (static_cast<std::size_t>(fb_frg->size()) != d_model) {
            Logging::EmitModuleError(kLogModule,
                Msg("[load] FATAL: final_rms_gamma size mismatch: checkpoint=",
                    fb_frg->size(), " expected=", d_model));
            return false;
        }
        if (!cross_check_view(load_req.final_rms_gamma, d_model, "final_rms_gamma")) return false;
    }

    return true;
}

} // namespace GRIM
