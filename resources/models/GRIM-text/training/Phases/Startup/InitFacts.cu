//======================================================//
//  Startup/InitFacts.cu
//
//  Implementation of verifyAndDumpInitFacts.
//  See InitFacts.hpp for the contract.
//======================================================//

#include "InitFacts.hpp"

#include "../../../Shared/Telemetry/TelemetryLattice_GPU.hpp"  // MetricStream
#include "../../../Shared/HyperParameters/HyperparameterGroupings.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace GRIMText::Training {

namespace {

constexpr int kInitTieCfg          = static_cast<int>(GRIM::Telemetry::MetricStream::INIT_TIE_CFG);
constexpr int kInitTiePtrsSame     = static_cast<int>(GRIM::Telemetry::MetricStream::INIT_TIE_PTRS_SAME);
constexpr int kInitTieGradsSame    = static_cast<int>(GRIM::Telemetry::MetricStream::INIT_TIE_GRADS_SAME);
constexpr int kInitLmOwnsWeights   = static_cast<int>(GRIM::Telemetry::MetricStream::INIT_LM_OWNS_WEIGHTS);
constexpr int kInitOptGroupsTotal  = static_cast<int>(GRIM::Telemetry::MetricStream::INIT_OPT_GROUPS_TOTAL);
constexpr int kInitOptGroupsEmb    = static_cast<int>(GRIM::Telemetry::MetricStream::INIT_OPT_GROUPS_EMB);
constexpr int kInitOptGroupsLm     = static_cast<int>(GRIM::Telemetry::MetricStream::INIT_OPT_GROUPS_LM);

// Format a raw pointer as 0x-prefixed hex for the CSV. Pointer values
// are not useful as floats, but they're useful when correlating across
// memory dumps or comparing two runs of the same config.
std::string fmtPtr(const void* p) {
    std::ostringstream os;
    os << p;
    return os.str();
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
        if (g.tensor && g.tensor->data == emb_w_ptr) ++emb_groups;
        if (g.tensor && g.tensor->data == lm_w_ptr)  ++lm_groups;
    }
    const int total_groups = static_cast<int>(model->parameterGroups().size());

    // ── Assertions: fail loud on tying-contract violations ──────────
    // (Rule 20: structural invariants throw; success path is silent
    //  in the human log and structured in the two CSV channels.)
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

    // ── Channel 1: telemetry stream slots (constant for run) ─────────
    ctx.telemetry.last_obs[kInitTieCfg]         = cfg_tied   ? 1.0f : 0.0f;
    ctx.telemetry.last_obs[kInitTiePtrsSame]    = ptrs_same  ? 1.0f : 0.0f;
    ctx.telemetry.last_obs[kInitTieGradsSame]   = grads_same ? 1.0f : 0.0f;
    ctx.telemetry.last_obs[kInitLmOwnsWeights]  = lm_owns    ? 1.0f : 0.0f;
    ctx.telemetry.last_obs[kInitOptGroupsTotal] = static_cast<float>(total_groups);
    ctx.telemetry.last_obs[kInitOptGroupsEmb]   = static_cast<float>(emb_groups);
    ctx.telemetry.last_obs[kInitOptGroupsLm]    = static_cast<float>(lm_groups);

    // ── Channel 2: init_facts_<session>.csv (key,value) ──────────────
    // Same naming scheme as training_<session>.log and telemetry_<session>.csv,
    // so all three live side-by-side in <log_dir>.
    const std::string csv_path =
        ctx.config.paths.log_dir + "/init_facts_" + ctx.logging.session_id + ".csv";
    std::ofstream out(csv_path, std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error(
            "verifyAndDumpInitFacts: failed to open " + csv_path + " for writing");
    }
    out << "key,value\n"
        << "config.tie_embeddings,"      << (cfg_tied   ? "true" : "false") << "\n"
        << "lm_owns_weights,"            << (lm_owns    ? "true" : "false") << "\n"
        << "emb_weight_ptr,"             << fmtPtr(emb_w_ptr) << "\n"
        << "lm_weight_ptr,"              << fmtPtr(lm_w_ptr)  << "\n"
        << "weight_ptrs_same,"           << (ptrs_same  ? "1" : "0") << "\n"
        << "emb_grad_ptr,"               << fmtPtr(emb_g_ptr) << "\n"
        << "lm_grad_ptr,"                << fmtPtr(lm_g_ptr)  << "\n"
        << "grad_ptrs_same,"             << (grads_same ? "1" : "0") << "\n"
        << "optimizer_groups.total,"     << total_groups  << "\n"
        << "optimizer_groups.emb_buffer," << emb_groups   << "\n"
        << "optimizer_groups.lm_buffer," << lm_groups     << "\n";
    out.flush();

    // Success path: one human-readable confirmation line. The structured
    // payload lives in the CSV + telemetry slots; readers go there.
    if (ctx.logging.logger) {
        ctx.logging.logger->log("✓ Init facts verified and written: " + csv_path);
    }
}

} // namespace GRIMText::Training
