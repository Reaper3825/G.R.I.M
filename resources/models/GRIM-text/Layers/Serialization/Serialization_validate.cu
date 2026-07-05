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
        // Biases are presence-driven: a checkpoint only carries attn/ffn biases
        // when the model allocated bias tensors (use_bias). When the model has no
        // bias destination (ptr==null) the checkpoint stores an empty vector, so
        // requiring a full-size bias here would wrongly reject every bias-less
        // checkpoint. Mirror the LM-head/layer-scale presence-driven contract.
        const bool has_layer_view = i < static_cast<int>(load_req.encoder_layers.size());
        const auto* lv = has_layer_view ? &load_req.encoder_layers[i] : nullptr;
        const bool model_has_attn_bias = lv && (lv->attn_b_qkv.ptr || lv->attn_b_o.ptr);
        const bool model_has_ffn_bias = lv && lv->ffn_b2.ptr;

        const auto* fa = fl->attention();
        if (!check_fb_vec_size(fa->w_qkv_data(), qkv_weight_size, "attn.W_qkv") ||
            !check_fb_vec_size(fa->w_o_data(), d_model * d_model, "attn.W_o")) {
            Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: attention size error in layer ", i));
            return false;
        }
        if (model_has_attn_bias &&
            (!check_fb_vec_size(fa->b_qkv_data(), total_qkv_dim, "attn.b_qkv") ||
             !check_fb_vec_size(fa->b_o_data(), d_model, "attn.b_o"))) {
            Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: attention bias size error in layer ", i));
            return false;
        }
        const auto* ff = fl->ffn();
        if (!check_fb_vec_size(ff->w1_data(), d_model * d_ff, "ffn.W1") ||
            !check_fb_vec_size(ff->w2_data(), d_ff * d_model, "ffn.W2")) {
            Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: FFN size error in layer ", i));
            return false;
        }
        if (model_has_ffn_bias &&
            !check_fb_vec_size(ff->b2_data(), d_model, "ffn.b2")) {
            Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: FFN bias size error in layer ", i));
            return false;
        }
        if (!check_fb_vec_size(fl->rms1()->gamma(), d_model, "rms1.gamma") ||
            !check_fb_vec_size(fl->rms2()->gamma(), d_model, "rms2.gamma")) {
            Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: RMSNorm gamma size error in layer ", i));
            return false;
        }
        // LayerScale: if model destination exists, checkpoint must provide the full
        // per-channel gamma vector. The checkpoint contract is d_model elements,
        // not "whatever the caller's view count says".
        if (i < static_cast<int>(load_req.encoder_layers.size())) {
            const auto& lv = load_req.encoder_layers[i];
            if (lv.layer_scale1.ptr) {
                if (!cross_check_view(lv.layer_scale1, d_model, "layer_scale1")) {
                    return false;
                }
                if (!check_fb_vec_size(fl->layer_scale1(), d_model, "layer_scale1")) {
                    Logging::EmitModuleError(kLogModule,
                        Msg("[load] FATAL: layer_scale1 vector error in layer ", i,
                            " — standard LayerScale requires d_model channels"));
                    return false;
                }
            }
            if (lv.layer_scale2.ptr) {
                if (!cross_check_view(lv.layer_scale2, d_model, "layer_scale2")) {
                    return false;
                }
                if (!check_fb_vec_size(fl->layer_scale2(), d_model, "layer_scale2")) {
                    Logging::EmitModuleError(kLogModule,
                        Msg("[load] FATAL: layer_scale2 vector error in layer ", i,
                            " — standard LayerScale requires d_model channels"));
                    return false;
                }
            }
        }
    }

    // ─── NumberEncoder ───
    if (req.requires_number_encoder) {
        const auto* fb_ne = model_fb->number_encoder();
        if (!fb_ne) {
            Logging::EmitModuleError(kLogModule, "[load] FATAL: NumberEncoder required but missing in checkpoint");
            return false;
        }
        auto ne_field = [&](const flatbuffers::Vector<float>* src, const DeviceWriteView& dst, const char* name) -> bool {
            if (!src) {
                Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: missing required NumberEncoder field: ", name));
                return false;
            }
            if (!dst.ptr) {
                Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: null model destination for NumberEncoder field: ", name));
                return false;
            }
            if (static_cast<std::size_t>(src->size()) != dst.count) {
                Logging::EmitModuleError(kLogModule,
                    Msg("[load] FATAL: NumberEncoder field ", name, " size mismatch: checkpoint=",
                        src->size(), " model_numel=", dst.count));
                return false;
            }
            return true;
        };
        // use_bias-gated hidden biases are optional: absent on both sides is valid;
        // present-in-checkpoint-only (model dropped the bias) is skipped; a missing
        // checkpoint field for a bias the model expects is fatal.
        auto ne_field_opt = [&](const flatbuffers::Vector<float>* src, const DeviceWriteView& dst, const char* name) -> bool {
            const bool have_src = (src != nullptr);
            const bool have_dst = (dst.ptr != nullptr);
            if (have_src && have_dst) return ne_field(src, dst, name);
            if (!have_src && !have_dst) return true;
            if (!have_src && have_dst) {
                Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: NumberEncoder bias destination present but missing in checkpoint: ", name));
                return false;
            }
            return true;  // checkpoint has bias the model dropped (use_bias gated off): skip
        };
        const auto& ne = load_req.number_encoder;
        ok = true;
        ok = ok && ne_field(fb_ne->digit_emb_data(), ne.digit_emb, "digit_emb");
        ok = ok && ne_field(fb_ne->pow10_emb_data(), ne.pow10_emb, "pow10_emb");
        ok = ok && ne_field(fb_ne->w_c1_data(), ne.W_c1, "W_c1");
        ok = ok && ne_field_opt(fb_ne->b_c1_data(), ne.b_c1, "b_c1");
        ok = ok && ne_field(fb_ne->w_c2_data(), ne.W_c2, "W_c2");
        ok = ok && ne_field(fb_ne->w_g1_data(), ne.W_g1, "W_g1");
        ok = ok && ne_field_opt(fb_ne->b_g1_data(), ne.b_g1, "b_g1");
        ok = ok && ne_field(fb_ne->w_g2_data(), ne.W_g2, "W_g2");
        if (!ok) return false;
    }

    // ─── Arg/option selector ───
    if (req.requires_arg_selector) {
        const auto* fb_sel = model_fb->arg_selector();
        if (!fb_sel) {
            Logging::EmitModuleError(kLogModule, "[load] FATAL: ArgSelector required but missing in checkpoint");
            return false;
        }
        const auto& sel = load_req.arg_selector;
        const auto* w_q_src = fb_sel->w_q_data();
        if (!w_q_src) {
            Logging::EmitModuleError(kLogModule, "[load] FATAL: missing required ArgSelector field: W_q");
            return false;
        }
        if (!sel.W_q.ptr) {
            Logging::EmitModuleError(kLogModule, "[load] FATAL: null model destination for ArgSelector field: W_q");
            return false;
        }
        if (static_cast<std::size_t>(w_q_src->size()) != sel.W_q.count) {
            Logging::EmitModuleError(kLogModule,
                Msg("[load] FATAL: ArgSelector field W_q size mismatch: checkpoint=",
                    w_q_src->size(), " model_numel=", sel.W_q.count));
            return false;
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

    // ─── LatentTrajectoryPreset ───
    if (req.requires_latent_trajectory_preset) {
        const auto* fb_ltp = model_fb->latent_trajectory_preset();
        if (!fb_ltp) {
            Logging::EmitModuleError(kLogModule, "[load] FATAL: LatentTrajectoryPreset required but missing in checkpoint");
            return false;
        }
        if (cfg.latent_trajectory_preset_mtp_k <= 0 ||
            cfg.latent_trajectory_preset_fuse_dim <= 0 ||
            cfg.latent_trajectory_preset_dim <= 0 ||
            cfg.latent_trajectory_preset_gate_dim <= 0) {
            Logging::EmitModuleError(kLogModule,
                Msg("[load] FATAL: LatentTrajectoryPreset required but config view has invalid derived dims: mtp_k=",
                    cfg.latent_trajectory_preset_mtp_k,
                    " fuse_dim=", cfg.latent_trajectory_preset_fuse_dim,
                    " preset_dim=", cfg.latent_trajectory_preset_dim,
                    " gate_dim=", cfg.latent_trajectory_preset_gate_dim));
            return false;
        }
        if (static_cast<int>(fb_ltp->d_model()) != cfg.d_model ||
            static_cast<int>(fb_ltp->mtp_k()) != cfg.latent_trajectory_preset_mtp_k ||
            static_cast<int>(fb_ltp->fuse_dim()) != cfg.latent_trajectory_preset_fuse_dim ||
            static_cast<int>(fb_ltp->preset_dim()) != cfg.latent_trajectory_preset_dim ||
            static_cast<int>(fb_ltp->gate_dim()) != cfg.latent_trajectory_preset_gate_dim) {
            Logging::EmitModuleError(kLogModule,
                Msg("[load] FATAL: LatentTrajectoryPreset dimension metadata mismatch: checkpoint=[d_model=",
                    fb_ltp->d_model(), " mtp_k=", fb_ltp->mtp_k(),
                    " fuse_dim=", fb_ltp->fuse_dim(),
                    " preset_dim=", fb_ltp->preset_dim(),
                    " gate_dim=", fb_ltp->gate_dim(),
                    "] model=[d_model=", cfg.d_model,
                    " mtp_k=", cfg.latent_trajectory_preset_mtp_k,
                    " fuse_dim=", cfg.latent_trajectory_preset_fuse_dim,
                    " preset_dim=", cfg.latent_trajectory_preset_dim,
                    " gate_dim=", cfg.latent_trajectory_preset_gate_dim, "]"));
            return false;
        }
        auto ltp_field = [&](const flatbuffers::Vector<float>* src, const DeviceWriteView& dst, const char* name) -> bool {
            if (!src) {
                Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: missing required LatentTrajectoryPreset field: ", name));
                return false;
            }
            if (!dst.ptr) {
                Logging::EmitModuleError(kLogModule, Msg("[load] FATAL: null model destination for LatentTrajectoryPreset field: ", name));
                return false;
            }
            if (static_cast<std::size_t>(src->size()) != dst.count) {
                Logging::EmitModuleError(kLogModule,
                    Msg("[load] FATAL: LatentTrajectoryPreset field ", name,
                        " size mismatch: checkpoint=", src->size(),
                        " model_numel=", dst.count));
                return false;
            }
            return true;
        };
        const auto& ltp = load_req.latent_trajectory_preset;
        ok = true;
        ok = ok && ltp_field(fb_ltp->w_hidden_traj_data(), ltp.W_hidden_traj, "W_hidden_traj");
        ok = ok && ltp_field(fb_ltp->b_hidden_traj_data(), ltp.b_hidden_traj, "b_hidden_traj");
        ok = ok && ltp_field(fb_ltp->w_fuse_data(), ltp.W_fuse, "W_fuse");
        ok = ok && ltp_field(fb_ltp->b_fuse_data(), ltp.b_fuse, "b_fuse");
        ok = ok && ltp_field(fb_ltp->w_down_data(), ltp.W_down, "W_down");
        ok = ok && ltp_field(fb_ltp->b_down_data(), ltp.b_down, "b_down");
        ok = ok && ltp_field(fb_ltp->w_up_data(), ltp.W_up, "W_up");
        ok = ok && ltp_field(fb_ltp->b_up_data(), ltp.b_up, "b_up");
        ok = ok && ltp_field(fb_ltp->w_gate_data(), ltp.W_gate, "W_gate");
        ok = ok && ltp_field(fb_ltp->b_gate_data(), ltp.b_gate, "b_gate");
        ok = ok && ltp_field(fb_ltp->w_target_data(), ltp.W_target, "W_target");
        ok = ok && ltp_field(fb_ltp->b_target_data(), ltp.b_target, "b_target");
        ok = ok && ltp_field(fb_ltp->fuse_norm_gamma_data(), ltp.fuse_norm_gamma, "fuse_norm_gamma");
        ok = ok && ltp_field(fb_ltp->preset_norm_gamma_data(), ltp.preset_norm_gamma, "preset_norm_gamma");
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

    // ─── LM-head residual SwiGLU adapter (presence-driven, lm_head_mlp_enabled) ───
    // When the model allocated adapter destinations the checkpoint must carry the
    // adapter with matching dimensions; a checkpoint-only adapter (model dropped
    // it) is legal and skipped at load time.
    if (load_req.lm_head.expect_mlp) {
        const auto* fb_lm = model_fb->lm_head();
        if (!fb_lm || !fb_lm->mlp_enabled()) {
            Logging::EmitModuleError(kLogModule,
                "[load] FATAL: LM-head SwiGLU adapter required (lm_head_mlp_enabled) "
                "but missing in checkpoint");
            return false;
        }
        const std::size_t mlp_d_ff = static_cast<std::size_t>(fb_lm->mlp_d_ff());
        if (mlp_d_ff == 0) {
            Logging::EmitModuleError(kLogModule,
                "[load] FATAL: LM-head SwiGLU adapter mlp_d_ff=0 in checkpoint");
            return false;
        }
        // Gate/up are [d_model, mlp_d_ff] and down is [mlp_d_ff, d_model]:
        // all three share the same element count.
        const std::size_t mlp_weight_numel = d_model * mlp_d_ff;
        if (!check_fb_vec_size(fb_lm->mlp_w_gate_data(), mlp_weight_numel, "lm_head.mlp_W_gate") ||
            !check_fb_vec_size(fb_lm->mlp_w_up_data(), mlp_weight_numel, "lm_head.mlp_W_up") ||
            !check_fb_vec_size(fb_lm->mlp_w_down_data(), mlp_weight_numel, "lm_head.mlp_W_down")) {
            return false;
        }
        if (!cross_check_view(load_req.lm_head.mlp_w_gate, mlp_weight_numel, "lm_head.mlp_W_gate") ||
            !cross_check_view(load_req.lm_head.mlp_w_up, mlp_weight_numel, "lm_head.mlp_W_up") ||
            !cross_check_view(load_req.lm_head.mlp_w_down, mlp_weight_numel, "lm_head.mlp_W_down")) {
            return false;
        }
    }

    return true;
}

} // namespace GRIM
