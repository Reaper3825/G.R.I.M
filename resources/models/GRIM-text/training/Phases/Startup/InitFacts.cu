//======================================================//
//  Startup/InitFacts.cu
//
//  Implementation of verifyAndDumpInitFacts.
//  See InitFacts.hpp for the contract.
//======================================================//

#include "InitFacts.hpp"

#include "../../../Shared/Telemetry/TelemetryLattice_GPU.hpp"  // MetricStream
#include "../../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../../Shared/LogRecorder/LogRecorder.hpp"

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace GRIMText::Training {

namespace {

constexpr int kInitTieCfg          = static_cast<int>(GRIM::Telemetry::MetricStream::INIT_TIE_CFG);
constexpr int kInitTiePtrsSame     = static_cast<int>(GRIM::Telemetry::MetricStream::INIT_TIE_PTRS_SAME);
constexpr int kInitTieGradsSame    = static_cast<int>(GRIM::Telemetry::MetricStream::INIT_TIE_GRADS_SAME);
constexpr int kInitLmOwnsWeights   = static_cast<int>(GRIM::Telemetry::MetricStream::INIT_LM_OWNS_WEIGHTS);
constexpr int kInitOptGroupsTotal  = static_cast<int>(GRIM::Telemetry::MetricStream::INIT_OPT_GROUPS_TOTAL);
constexpr int kInitOptGroupsEmb    = static_cast<int>(GRIM::Telemetry::MetricStream::INIT_OPT_GROUPS_EMB);
constexpr int kInitOptGroupsLm     = static_cast<int>(GRIM::Telemetry::MetricStream::INIT_OPT_GROUPS_LM);

const char* boolText(bool value) {
    if (value) {
        return "true";
    }
    return "false";
}

// Format a raw pointer for fail-loud invariant messages. Pointer values do
// not fit in float telemetry streams, so they are only emitted on failure.
std::string fmtPtr(const void* p) {
    std::ostringstream os;
    os << p;
    return os.str();
}

std::string fmtInt(int value) {
    return std::to_string(value);
}

std::string fmtUInt64(std::uint64_t value) {
    return std::to_string(value);
}

std::string fmtSize(std::size_t value) {
    return std::to_string(value);
}

std::string fmtFloat(float value) {
    std::ostringstream os;
    os << value;
    return os.str();
}

const char* paramGroupTypeName(GRIM::ParamGroupType type) {
    switch (type) {
        case GRIM::ParamGroupType::EMBEDDING: return "EMBEDDING";
        case GRIM::ParamGroupType::LM_HEAD: return "LM_HEAD";
        case GRIM::ParamGroupType::ATTENTION: return "ATTENTION";
        case GRIM::ParamGroupType::FFN: return "FFN";
        case GRIM::ParamGroupType::RMSNORM: return "RMSNORM";
        case GRIM::ParamGroupType::SCRATCHBLOCK: return "SCRATCHBLOCK";
        case GRIM::ParamGroupType::MTP: return "MTP";
        case GRIM::ParamGroupType::REASONING_HEAD: return "REASONING_HEAD";
        case GRIM::ParamGroupType::EXECUTION_BLOCK: return "EXECUTION_BLOCK";
        case GRIM::ParamGroupType::SLOT_SELECTOR: return "SLOT_SELECTOR";
        case GRIM::ParamGroupType::COUNT: break;
    }
    throw std::runtime_error("paramGroupTypeName: invalid ParamGroupType::COUNT");
}

const char* paramStatsBucketName(GRIM::ParamStatsBucket bucket) {
    switch (bucket) {
        case GRIM::ParamStatsBucket::EMBEDDING: return "EMBEDDING";
        case GRIM::ParamStatsBucket::ENCODER: return "ENCODER";
        case GRIM::ParamStatsBucket::LM_HEAD: return "LM_HEAD";
        case GRIM::ParamStatsBucket::COUNT: break;
    }
    throw std::runtime_error("paramStatsBucketName: invalid ParamStatsBucket::COUNT");
}

void emitInitFactLine(const std::string& line) {
    GRIM::Logging::EmitModuleInfo("Training", line, 0, true);
}

void emitInitFactKeyValue(const std::string& key, const std::string& value) {
    emitInitFactLine("[INIT_FACTS] " + key + " = " + value);
}

} // namespace

void verifyAndDumpInitFacts(TrainingContext& ctx) {
    if (!ctx.model) {
        throw std::runtime_error(
            "verifyAndDumpInitFacts: ctx.model is null (called before initializeModel?)");
    }

    // ── Collect the live structural facts ────────────────────────────
    auto* model = ctx.model.get();
    const float* emb_w_ptr = model->getEmbeddingLayer()->tokenWeights().data;
    const float* lm_w_ptr  = model->getLmHeadLayer()->weights().data;
    const float* emb_g_ptr = model->getEmbeddingLayer()->tokenWeights().grad_data();
    const float* lm_g_ptr  = model->getLmHeadLayer()->weights().grad_data();
    const auto lm_head_hp =
        GRIM::HyperParameters::lmHeadLayerConstructionHP(ctx.config.hyperparameters.architecture);
    const bool cfg_tied  = lm_head_hp.tie_embeddings;
    const bool lm_owns   = model->getLmHeadLayer()->ownsWeights();
    const bool ptrs_same  = (emb_w_ptr == lm_w_ptr);
    const bool grads_same = (emb_g_ptr == lm_g_ptr);

    int emb_groups = 0;
    int lm_groups  = 0;
    for (const auto& g : model->parameterGroups()) {
        if (!g.tensor) {
            throw std::runtime_error("verifyAndDumpInitFacts: parameter group '" + g.name + "' has NULL tensor");
        }
        if (g.tensor->data == emb_w_ptr) ++emb_groups;
        if (g.tensor->data == lm_w_ptr)  ++lm_groups;
    }
    const int total_groups = static_cast<int>(model->parameterGroups().size());

    // ── Assertions: fail loud on tying-contract violations ──────────
    // (Rule 20: structural invariants throw; success path is a single
    //  human log line plus structured telemetry slots.)
    if (cfg_tied && !ptrs_same) {
        throw std::runtime_error(
            "tie_embeddings=true but embedding/lm-head WEIGHT pointers differ "
            "(emb=" + fmtPtr(emb_w_ptr) + " lm=" + fmtPtr(lm_w_ptr) + ")");
    }
    if (!cfg_tied && ptrs_same) {
        throw std::runtime_error(
            "tie_embeddings=false but embedding/lm-head WEIGHT pointers are SAME "
            "(both=" + fmtPtr(emb_w_ptr) + ") — unexpected aliasing");
    }
    if (cfg_tied && !grads_same) {
        throw std::runtime_error(
            "tie_embeddings=true but embedding/lm-head GRAD pointers differ "
            "(emb=" + fmtPtr(emb_g_ptr) + " lm=" + fmtPtr(lm_g_ptr) + ") — "
            "tied weights would receive only one side's gradient");
    }
    if (cfg_tied && emb_groups > 0) {
        // When tied, only the lm-head side should appear in the optimizer's
        // group list; an extra embedding-side group would double-step the
        // shared buffer per AdamW pass.
        throw std::runtime_error(
            "tie_embeddings=true but optimizer has " + std::to_string(emb_groups) +
            " parameter group(s) referencing the embedding buffer in addition to " +
            std::to_string(lm_groups) + " lm-head group(s) — tied buffer would be double-stepped");
    }

    // ── Telemetry stream slots (constant for run) ────────────────────
    ctx.telemetry.last_obs[kInitTieCfg]         = cfg_tied   ? 1.0f : 0.0f;
    ctx.telemetry.last_obs[kInitTiePtrsSame]    = ptrs_same  ? 1.0f : 0.0f;
    ctx.telemetry.last_obs[kInitTieGradsSame]   = grads_same ? 1.0f : 0.0f;
    ctx.telemetry.last_obs[kInitLmOwnsWeights]  = lm_owns    ? 1.0f : 0.0f;
    ctx.telemetry.last_obs[kInitOptGroupsTotal] = static_cast<float>(total_groups);
    ctx.telemetry.last_obs[kInitOptGroupsEmb]   = static_cast<float>(emb_groups);
    ctx.telemetry.last_obs[kInitOptGroupsLm]    = static_cast<float>(lm_groups);

    // Success path: full human-readable dump through the LogRecorder tape.
    // The text sink is training_<session>_tape.log. Each value gets its own
    // line to avoid the fixed LogEntry message buffer truncating the payload.
    emitInitFactLine("[INIT_FACTS] ========================================================================");
    emitInitFactLine("[INIT_FACTS] Init structural facts: effective configuration and live model state");
    emitInitFactLine("[INIT_FACTS] ========================================================================");

    emitInitFactLine("[INIT_FACTS] --- Run identity ---------------------------------------------------------");
    emitInitFactKeyValue("session_id", ctx.logging.session_id);
    emitInitFactKeyValue("config_path", ctx.config.paths.config_path.string());
    emitInitFactKeyValue("log_dir", ctx.config.paths.log_dir);
    emitInitFactKeyValue("training_data", ctx.config.paths.data_path);
    emitInitFactKeyValue("vocab_path", ctx.config.paths.vocab_path);
    emitInitFactKeyValue("checkpoint_dir", ctx.config.paths.checkpoint_dir);
    emitInitFactKeyValue("output_model_path", ctx.config.paths.output_model_path);
    emitInitFactKeyValue("loaded_checkpoint_path", ctx.loaded_checkpoint_path);

    emitInitFactLine("[INIT_FACTS] --- Effective architecture -----------------------------------------------");
    const auto& arch = ctx.config.hyperparameters.architecture;
    emitInitFactKeyValue("architecture.d_model", fmtInt(arch.d_model));
    emitInitFactKeyValue("architecture.num_layers", fmtInt(arch.num_layers));
    emitInitFactKeyValue("architecture.num_heads", fmtInt(arch.num_heads));
    emitInitFactKeyValue("architecture.num_kv_heads", fmtInt(arch.num_kv_heads));
    emitInitFactKeyValue("architecture.head_dim", fmtInt(arch.head_dim));
    emitInitFactKeyValue("architecture.d_ff", fmtInt(arch.d_ff));
    emitInitFactKeyValue("architecture.max_seq_len", fmtInt(arch.max_seq_len));
    emitInitFactKeyValue("architecture.vocab_size", fmtInt(arch.vocab_size));
    emitInitFactKeyValue("actual_vocab_size", fmtUInt64(ctx.config.actual_vocab_size));
    emitInitFactKeyValue("architecture.tie_embeddings", boolText(arch.tie_embeddings));
    emitInitFactKeyValue("architecture.use_bias", boolText(arch.use_bias));
    emitInitFactKeyValue("architecture.use_scratch_block", boolText(arch.use_scratch_block));
    emitInitFactKeyValue("architecture.execution_block_enabled", boolText(arch.execution_block_enabled));
    emitInitFactKeyValue("architecture.mtp_enabled", boolText(arch.mtp_enabled));

    emitInitFactLine("[INIT_FACTS] --- Run capacity ---------------------------------------------------------");
    emitInitFactKeyValue("run_capacity.batch_rows", fmtSize(ctx.run_capacity.batch_rows));
    emitInitFactKeyValue("run_capacity.seq_cap", fmtSize(ctx.run_capacity.seq_cap));
    emitInitFactKeyValue("run_capacity.max_tokens_per_batch", fmtSize(ctx.run_capacity.max_tokens_per_batch));
    emitInitFactKeyValue("startup.max_seq_len", fmtInt(ctx.config.max_seq_len));
    emitInitFactKeyValue("startup.sliding_window_stride", fmtInt(ctx.config.sliding_window_stride));

    emitInitFactLine("[INIT_FACTS] --- Tied embedding / LM-head contract -----------------------------------");
    emitInitFactKeyValue("config.tie_embeddings", boolText(cfg_tied));
    emitInitFactKeyValue("lm_head.owns_weights", boolText(lm_owns));
    emitInitFactKeyValue("embedding.weight_ptr", fmtPtr(emb_w_ptr));
    emitInitFactKeyValue("lm_head.weight_ptr", fmtPtr(lm_w_ptr));
    emitInitFactKeyValue("weight_ptrs_same", boolText(ptrs_same));
    emitInitFactKeyValue("embedding.grad_ptr", fmtPtr(emb_g_ptr));
    emitInitFactKeyValue("lm_head.grad_ptr", fmtPtr(lm_g_ptr));
    emitInitFactKeyValue("grad_ptrs_same", boolText(grads_same));
    emitInitFactKeyValue("optimizer_groups.total", fmtInt(total_groups));
    emitInitFactKeyValue("optimizer_groups.embedding_buffer", fmtInt(emb_groups));
    emitInitFactKeyValue("optimizer_groups.lm_head_buffer", fmtInt(lm_groups));

    emitInitFactLine("[INIT_FACTS] --- Telemetry stream mirror ---------------------------------------------");
    emitInitFactKeyValue("telemetry[48].init_tie_cfg", fmtFloat(ctx.telemetry.last_obs[kInitTieCfg]));
    emitInitFactKeyValue("telemetry[49].init_tie_ptrs_same", fmtFloat(ctx.telemetry.last_obs[kInitTiePtrsSame]));
    emitInitFactKeyValue("telemetry[50].init_tie_grads_same", fmtFloat(ctx.telemetry.last_obs[kInitTieGradsSame]));
    emitInitFactKeyValue("telemetry[51].init_lm_owns_weights", fmtFloat(ctx.telemetry.last_obs[kInitLmOwnsWeights]));
    emitInitFactKeyValue("telemetry[52].init_opt_groups_total", fmtFloat(ctx.telemetry.last_obs[kInitOptGroupsTotal]));
    emitInitFactKeyValue("telemetry[53].init_opt_groups_emb", fmtFloat(ctx.telemetry.last_obs[kInitOptGroupsEmb]));
    emitInitFactKeyValue("telemetry[54].init_opt_groups_lm", fmtFloat(ctx.telemetry.last_obs[kInitOptGroupsLm]));

    emitInitFactLine("[INIT_FACTS] --- Parameter groups -----------------------------------------------------");
    const auto& groups = model->parameterGroups();
    for (std::size_t i = 0; i < groups.size(); ++i) {
        const auto& g = groups[i];
        if (!g.tensor) {
            throw std::runtime_error("verifyAndDumpInitFacts: parameter group index " + fmtSize(i) + " has NULL tensor");
        }
        std::ostringstream line;
        line << "[INIT_FACTS] parameter_groups[" << i << "]"
             << " name=" << g.name
             << " type=" << paramGroupTypeName(g.type)
             << " bucket=" << paramStatsBucketName(g.stats_bucket)
             << " layer=" << g.layer_index
             << " numel=" << g.tensor->numel()
             << " data=" << fmtPtr(g.tensor->data)
             << " grad=" << fmtPtr(g.tensor->grad_data())
             << " m_state=" << fmtPtr(g.m_state())
             << " v_state=" << fmtPtr(g.v_state())
             << " wd_mult=" << g.weight_decay_multiplier
             << " lr_mult=" << g.lr_multiplier
             << " upsilon=" << g.upsilon;
        emitInitFactLine(line.str());
    }

    emitInitFactLine("[INIT_FACTS] ========================================================================");

    if (ctx.logging.logger) {
        ctx.logging.logger->log("✓ Init facts verified and dumped to LogRecorder tape");
    }
}

} // namespace GRIMText::Training
