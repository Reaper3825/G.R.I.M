//======================================================//
//  Startup/InitFacts.cu
//
//  Implementation of verifyAndDumpInitFacts.
//  See InitFacts.hpp for the contract.
//======================================================//

#include "InitFacts.hpp"

#include "../../../Shared/Telemetry/TelemetryLattice_GPU.hpp"  // MetricStream
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
        case GRIM::ParamGroupType::ARG_SELECTOR: return "ARG_SELECTOR";
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
    // ── Collect the live structural facts ────────────────────────────
    auto& embedding_parameters = ctx.parameter_registry.requireEmbeddingParameters("verifyAndDumpInitFacts");
    auto& lm_head_parameters = ctx.parameter_registry.requireLmHeadParameters("verifyAndDumpInitFacts");
    const auto& parameter_groups = ctx.parameter_registry.requireParameterGroups("verifyAndDumpInitFacts");
    const float* emb_w_ptr = embedding_parameters.token_weights.data;
    const float* lm_w_ptr  = lm_head_parameters.weights.data;
    const float* emb_g_ptr = embedding_parameters.token_weights.grad_data();
    const float* lm_g_ptr  = lm_head_parameters.weights.grad_data();
    const bool cfg_tied  = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(ctx.config, "tie_embeddings");
    const bool lm_owns   = lm_head_parameters.owns_weights;
    const bool ptrs_same  = (emb_w_ptr == lm_w_ptr);
    const bool grads_same = (emb_g_ptr == lm_g_ptr);

    int emb_groups = 0;
    int lm_groups  = 0;
    for (const auto& g : parameter_groups) {
        if (!g.tensor) {
            throw std::runtime_error("verifyAndDumpInitFacts: parameter group '" + g.name + "' has NULL tensor");
        }
        const bool refs_emb = (g.tensor->data == emb_w_ptr);
        const bool refs_lm  = (g.tensor->data == lm_w_ptr);
        if (ptrs_same) {
            // Tied: embedding and lm-head alias one buffer. The registrar adds a
            // single shared group on the lm-head side, so attribute it to
            // lm_groups only — counting it on both sides would falsely look like
            // a double-step of the shared buffer.
            if (refs_lm) ++lm_groups;
        } else {
            if (refs_emb) ++emb_groups;
            if (refs_lm)  ++lm_groups;
        }
    }
    const int total_groups = static_cast<int>(parameter_groups.size());

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
    if (cfg_tied && lm_groups != 1) {
        // When tied, exactly one optimizer group must reference the shared
        // embedding/lm-head buffer: zero means the tied weights never update,
        // more than one means AdamW would double-step the shared buffer per pass.
        throw std::runtime_error(
            "tie_embeddings=true expects exactly one optimizer group referencing the shared "
            "embedding/lm-head buffer but found " + std::to_string(lm_groups) +
            " — tied buffer would be " +
            std::string(lm_groups == 0 ? "left unoptimized" : "double-stepped"));
    }

    // ── Telemetry stream slots (constant for run) ────────────────────
    ctx.telemetry.last_obs[kInitTieCfg]         = cfg_tied   ? 1.0f : 0.0f;
    ctx.telemetry.last_obs[kInitTiePtrsSame]    = ptrs_same  ? 1.0f : 0.0f;
    ctx.telemetry.last_obs[kInitTieGradsSame]   = grads_same ? 1.0f : 0.0f;
    ctx.telemetry.last_obs[kInitLmOwnsWeights]  = lm_owns    ? 1.0f : 0.0f;
    ctx.telemetry.last_obs[kInitOptGroupsTotal] = static_cast<float>(total_groups);
    ctx.telemetry.last_obs[kInitOptGroupsEmb]   = static_cast<float>(emb_groups);
    ctx.telemetry.last_obs[kInitOptGroupsLm]    = static_cast<float>(lm_groups);

    // Success path: full human-readable dump through the LogRecorder text sink.
    // The text sink shares training_<session>.log with TrainingLogger. Each value keeps
    // its own line so the init report stays readable and grep-friendly in the shared log.
    emitInitFactLine("[INIT_FACTS] ========================================================================");
    emitInitFactLine("[INIT_FACTS] Init structural facts: effective configuration and live model state");
    emitInitFactLine("[INIT_FACTS] ========================================================================");

    emitInitFactLine("[INIT_FACTS] --- Run identity ---------------------------------------------------------");
    const auto paths_hp = GRIM::HyperParameters::pathsHP(ctx.config);
    emitInitFactKeyValue("session_id", ctx.logging.session_id);
    emitInitFactKeyValue("config_source", "canonical ai_config.json document");
    emitInitFactKeyValue("log_dir", paths_hp.log_dir);
    emitInitFactKeyValue("training_data", paths_hp.data_path);
    emitInitFactKeyValue("vocab_path", paths_hp.vocab_path);
    emitInitFactKeyValue("checkpoint_dir", paths_hp.checkpoint_dir);
    emitInitFactKeyValue("output_model_path", paths_hp.output_model_path);
    const bool checkpoint_loaded = !ctx.loaded_checkpoint_path.empty();
    const char* planned_parameter_source =
        ctx.model_parameter_source_plan == ModelParameterSourcePlan::CHECKPOINT_RESTORE
            ? "checkpoint"
            : (ctx.model_parameter_source_plan == ModelParameterSourcePlan::FRESH_INITIALIZATION
                   ? "random_initialization"
                   : "unresolved");
    emitInitFactKeyValue("planned_model_parameter_source", planned_parameter_source);
    emitInitFactKeyValue("planned_checkpoint_candidates",
                         fmtUInt64(static_cast<std::uint64_t>(ctx.planned_checkpoint_candidates.size())));
    emitInitFactKeyValue("model_parameter_source", checkpoint_loaded ? "checkpoint" : "random_initialization");
    emitInitFactKeyValue("checkpoint_loaded", boolText(checkpoint_loaded));
    emitInitFactKeyValue("loaded_checkpoint_path", checkpoint_loaded ? ctx.loaded_checkpoint_path : "<none>");

    emitInitFactLine("[INIT_FACTS] --- Effective architecture -----------------------------------------------");
    emitInitFactKeyValue("architecture.d_model", fmtInt(GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "d_model")));
    emitInitFactKeyValue("architecture.num_layers", fmtInt(GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "num_layers")));
    emitInitFactKeyValue("architecture.num_heads", fmtInt(GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "num_heads")));
    emitInitFactKeyValue("architecture.num_kv_heads", fmtInt(GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "num_kv_heads")));
    emitInitFactKeyValue("architecture.head_dim", fmtInt(GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "head_dim")));
    emitInitFactKeyValue("architecture.d_ff", fmtInt(GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "d_ff")));
    emitInitFactKeyValue("architecture.max_seq_len", fmtInt(GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "max_seq_len")));
    emitInitFactKeyValue("startup.actual_vocab_size", fmtUInt64(static_cast<std::uint64_t>(ctx.data.vocab_size)));
    emitInitFactKeyValue("architecture.tie_embeddings", boolText(GRIM::HyperParameters::snapshotTrainingConfigField<bool>(ctx.config, "tie_embeddings")));
    emitInitFactKeyValue("architecture.use_bias", boolText(GRIM::HyperParameters::snapshotTrainingConfigField<bool>(ctx.config, "use_bias")));
    emitInitFactKeyValue("architecture.use_atom_data", boolText(GRIM::HyperParameters::snapshotTrainingConfigField<bool>(ctx.config, "use_atom_data")));
    emitInitFactKeyValue("architecture.execution_block_enabled", boolText(GRIM::HyperParameters::snapshotTrainingConfigField<bool>(ctx.config, "execution_block_enabled")));
    const auto fixed_shape = GRIM::HyperParameters::trainingFixedShapeHP(ctx.config);
    emitInitFactLine("[INIT_FACTS] --- Fixed training shape -------------------------------------------------");
    emitInitFactKeyValue("fixed_shape.batch_size", fmtInt(fixed_shape.batch_size));
    emitInitFactKeyValue("fixed_shape.max_seq_len", fmtInt(fixed_shape.max_seq_len));
    emitInitFactKeyValue("fixed_shape.max_tokens_per_batch", fmtInt(fixed_shape.max_tokens_per_batch));
    emitInitFactKeyValue("startup.max_seq_len", fmtInt(GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "max_seq_len")));
    emitInitFactKeyValue("startup.sliding_window_stride", fmtInt(GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "sliding_window_stride")));

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
    const auto& groups = parameter_groups;
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
        ctx.logging.logger->log("✓ Init facts verified and dumped to the shared session log");
    }
}

} // namespace GRIMText::Training
