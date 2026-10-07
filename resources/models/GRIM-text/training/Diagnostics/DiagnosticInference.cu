//======================================================//
//  DiagnosticInference.cu
//  Isolated inference sampling for training diagnostics
//======================================================//
//
//  This file houses ALL training-time inference diagnostic
//  logic.  It is the ONLY entry point for sample generation
//  during training.  No diagnostic code should modify shared
//  training state (weight tensors, requires_grad, optimizer).
//
//  The underlying Phase2 inference path chooses KV decode only for sequence-local
//  geometry. Sequence-coupled centering/projection uses full-context inference.
//  Both modes keep inference state separate from optimizer-owned training state.
//
//  Author: Austin Wadkins
//  Date: April 2026
//======================================================//

#include "DiagnosticInference.hpp"
#include "DiagnosticLens.hpp"
#include "../../Shared/UnigramByte/AtomTable.hpp"
#include "../Phases/Phase2_InferenceLoop.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/DataLoader/DataLoader.hpp"

#include <iostream>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <limits>
#include <optional>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace GRIMText::Training {
namespace {

struct DiagnosticPrompt {
    const char* id;
    const char* prompt;
};

// One probe per diagnostic interval, cycling over every contextual unknown N = DD
// position. IDs are log metadata only; the model must infer the requested role.
constexpr std::array<DiagnosticPrompt, 12> kArithmeticPrompts{{
    {"decrease_removed", "A tank holds 120 liters. After using a few liters, 84 liters remain. How many liters were used?"},
    {"decrease_start", "After 36 liters of water are used from a tank, 84 liters remain. How many liters were in the tank at first?"},
    {"decrease_remaining", "A tank initially holds 120 liters. Then 36 liters are used. How many liters remain?"},
    {"increase_start", "Maya receives 19 marbles and now has 42. How many marbles did she have before receiving them?"},
    {"increase_added", "Maya had 23 marbles before receiving some more. She now has 42. How many marbles did she receive?"},
    {"increase_final", "Maya has 23 marbles and receives 19 more. How many marbles does she have now?"},
    {"comparison_smaller", "The blue rope is 18 feet shorter than the red rope. The red rope is 50 feet long. How long is the blue rope?"},
    {"comparison_difference", "The red rope is 50 feet long and the blue rope is 32 feet long. How many feet shorter is the blue rope?"},
    {"comparison_larger", "The blue rope is 32 feet long, which is 18 feet shorter than the red rope. How long is the red rope?"},
    {"groups_group_count", "There are 72 students arranged into teams with 9 students on each team. How many teams are there?"},
    {"groups_per_group", "There are 72 students divided equally among 8 teams. How many students are on each team?"},
    {"groups_total", "There are 8 teams with 9 students on each team. How many students are there altogether?"},
}};

//------------------------------------------------------
//  Environment helpers (self-contained, no Phase2 deps)
//------------------------------------------------------

int readEnvInt(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    char* end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (end == raw || value < 0) return fallback;
    return static_cast<int>(value);
}

std::string readEnvString(const char* name, const std::string& fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    return std::string(raw);
}

//------------------------------------------------------
//  Text helpers
//------------------------------------------------------

std::string trimSampleText(const std::string& text, std::size_t max_chars) {
    if (text.size() <= max_chars) return text;
    return text.substr(0, max_chars) + "...";
}

}  // anonymous namespace

//======================================================//
//  logDiagnosticSample — public entry point
//======================================================//

void logDiagnosticSample(TrainingContext& ctx,
                         TrainingLoopState& state,
                         bool inference_diagnostic_enabled,
                         int inference_diagnostic_interval) {
    if (!inference_diagnostic_enabled) {
        return;
    }
    if (inference_diagnostic_interval <= 0) {
        throw std::runtime_error("logDiagnosticSample: inference_diagnostic_interval must be > 0 when inference_diagnostic_enabled=true");
    }

    const int optimizer_step = ctx.optimizer.optimizer_step.step;
    if (optimizer_step <= 0 || optimizer_step % inference_diagnostic_interval != 0 || optimizer_step == state.last_sample_step) {
        return;
    }
    state.last_sample_step = optimizer_step;

    if (!ctx.logging.logger) {
        return;
    }
    // Drain deferred CUDA errors from training before launching inference kernels.
    // Without this, async errors from the optimizer/backward pass manifest as
    // "invalid argument" on the first inference kernel launch (RoPE, ScratchBlock).
    {
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            ctx.logging.logger->log("[Sample] WARNING: cudaDeviceSynchronize before generate: " +
                std::string(cudaGetErrorString(sync_err)));
        }
        cudaError_t deferred = cudaGetLastError();
        if (deferred != cudaSuccess) {
            ctx.logging.logger->log("[Sample] WARNING: Cleared deferred CUDA error before generate: " +
                std::string(cudaGetErrorString(deferred)));
        }
    }

    const auto sample_index = static_cast<std::size_t>(
        optimizer_step / inference_diagnostic_interval - 1) % kArithmeticPrompts.size();
    const auto& diagnostic = kArithmeticPrompts[sample_index];
    const std::string prompt = readEnvString("GRIM_SAMPLE_PROMPT", diagnostic.prompt);
    const char* custom_prompt = std::getenv("GRIM_SAMPLE_PROMPT");
    const char* diagnostic_id = custom_prompt && *custom_prompt ? "custom" : diagnostic.id;

    // Match the first-pass arithmetic curriculum exactly: no persisted state,
    // operand bindings, operation hints, or target-role hints in the prefix.
    // This generic goal also remains valid for GRIM_SAMPLE_PROMPT overrides.
    const nlohmann::json input_state{{"goal", {
        {"target_state", "The quantity requested in the question is correctly reported."},
        {"success_criteria", {{{"criterion", "The response answers the question using the stated quantities."},
                               {"evidence", ""}}}},
        {"constraints", {"Use only the quantities stated in the problem.",
                         "Report the quantity in the unit requested by the question.",
                         "Use exactly one single-step arithmetic tool call."}}
    }}};
    const int max_new_tokens = readEnvInt("GRIM_SAMPLE_TOKENS", 256);
    const int max_chars = readEnvInt("GRIM_SAMPLE_MAX_CHARS", 2048);
    if (max_new_tokens <= 0 || max_chars <= 0) {
        return;
    }

    // Start from the finalized root generation view and apply diagnostic overrides locally.
    GRIM::HyperParameters::GenerationHP cfg = GRIM::HyperParameters::generationHP(ctx.config);
    cfg.max_new_tokens = max_new_tokens;
    cfg.min_new_tokens = std::max(1, max_new_tokens / 4);
    cfg.num_return_sequences = 1;
    // Seed from optimizer step for reproducible but varied samples per step
    cfg.seed = static_cast<unsigned int>(optimizer_step);
    // Structured SFT output is copy-heavy (section tags, ${variable} names).
    // SamplingPipeline scans the FULL history (rendered scaffold prompt +
    // generated tokens), so n-gram blocking bans tag trigrams such as
    // ">" "\n\n" "<" and repetition penalty suppresses copied variable
    // pieces. Disable both so the diagnostic reflects the model's argmax.
    cfg.no_repeat_ngram_size = 0;
    cfg.repetition_penalty = 1.0f;

    try {
        const auto lens_options = validateDiagnosticLensOptions({
            cfg.lens_enabled, cfg.lens_validate, cfg.lens_top_k,
            cfg.lens_validation_rounds, cfg.lens_validation_tokens,
            cfg.lens_max_replay_rows, cfg.lens_absolute_tolerance});
        auto tokenizer = LoadInferenceTokenizer(ctx.config, *ctx.logging.logger);
        GRIM::Lenses::LensCaptureRequest lens;
        if (lens_options.enabled) {
            if (!ctx.config.model_config || ctx.logging.session_id.empty())
                throw std::runtime_error("Lens diagnostic requires compiled config and training session identity");
            if (GRIM::HyperParameters::atomInsertionBoundaryProjectionHP(ctx.config).enabled)
                throw std::runtime_error("Lens diagnostic currently requires a token model, not an insertion-gap model");
            std::ostringstream config_hash;
            config_hash << std::hex << std::setfill('0');
            for (auto byte : ctx.config.model_config->integrity.semantic_sha256)
                config_hash << std::setw(2) << static_cast<unsigned int>(byte);
            std::uint64_t vocab_hash = 0;
            const auto vocab_path = GRIM::HyperParameters::pathsHP(ctx.config).vocab_path;
            if (!GRIM::Tokenizer::AtomTable::computeFileHash(vocab_path, vocab_hash))
                throw std::runtime_error("Lens diagnostic could not fingerprint tokenizer artifact");
            lens.identity = {ctx.logging.session_id,
                "live-training:" + ctx.logging.session_id + ":step:" + std::to_string(optimizer_step),
                "sha256:" + config_hash.str(), "vocab-file-hash64:" + std::to_string(vocab_hash),
                static_cast<std::uint64_t>(optimizer_step)};
            lens.top_k = std::min(lens_options.top_k, tokenizer->vocabSize());
            lens.replay_identity_control = true;
            lens.max_replay_rows = lens_options.max_replay_rows;
        }
        const bool atom_insertion =
            GRIM::HyperParameters::atomInsertionBoundaryProjectionHP(ctx.config).enabled;
        const auto encode_start = std::chrono::steady_clock::now();
        std::optional<GRIM::Batching::BatchPayload> prefill;
        if (!atom_insertion)
            prefill = buildPhase2InferencePrefill(ctx, *tokenizer, prompt, input_state);
        const auto prefill_encode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - encode_start).count();
        const auto concept_spans =
            GRIM::HyperParameters::snapshotTrainingConfigField<GRIM::NamedConceptSpanDefinitions>(
                ctx.config, "concept_spans");
        if (lens_options.enabled) {
            const GRIM::NamedConceptSpanDefinition* prompt_definition = nullptr;
            for (const auto& definition : concept_spans) {
                if (definition.source_path != "/prompt") continue;
                if (prompt_definition)
                    throw std::runtime_error("Lens diagnostic: ambiguous configured prompt span");
                prompt_definition = &definition;
            }
            if (!prompt_definition)
                throw std::runtime_error("Lens diagnostic: no configured root span for /prompt");
            lens.prompt_span_name = prompt_definition->name;
        }
        auto run_sample = [&](const GRIM::Lenses::LensCaptureRequest* capture = nullptr) {
            if (atom_insertion)
                return executePhase2TextInference(ctx, *tokenizer, prompt, cfg, capture);
            auto result = executePhase2PayloadInference(ctx, *tokenizer, *prefill, cfg, capture);
            result.encode_ms = prefill_encode_ms;
            return result;
        };
        const auto start = std::chrono::steady_clock::now();
        if (lens_options.validate) {
            cfg.do_sample = false;
            cfg.max_new_tokens = lens_options.validation_tokens;
            cfg.min_new_tokens = 1;
        }
        std::optional<Phase2TextInferenceResult> baseline;
        if (lens_options.validate)
            baseline = run_sample();
        auto sample = run_sample(lens_options.enabled ? &lens : nullptr);
        auto report_lens = [&](const Phase2TextInferenceResult& captured) {
            if (!captured.prefill_lens_snapshot)
                throw std::runtime_error("Lens diagnostic requested capture but inference returned no snapshot");
            const auto report = diagnosticLensReport(*captured.prefill_lens_snapshot,
                lens_options.absolute_tolerance, [&](int id) {
                    return tokenizer->decode(GRIM::Tokenizer::DecodeRequest({id}));
                });
            // Escape partial byte tokens and control characters in a single log record.
            ctx.logging.logger->log("[Lens] " + report.dump(-1, ' ', true,
                nlohmann::json::error_handler_t::replace));
        };
        if (lens_options.enabled) report_lens(sample);
        if (baseline) {
            auto memory_used = []() {
                std::size_t free_bytes = 0, total_bytes = 0;
                const auto sync = cudaDeviceSynchronize();
                if (sync != cudaSuccess) throw std::runtime_error(cudaGetErrorString(sync));
                const auto status = cudaMemGetInfo(&free_bytes, &total_bytes);
                if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
                return total_bytes - free_bytes;
            };
            // Baseline and first captured run warm both paths. Measurements below
            // are device-wide observations, not a leak verdict or peak scratch size.
            std::vector<std::size_t> used_bytes{memory_used()};
            bool token_match = baseline->token_ids == sample.token_ids;
            bool text_match = baseline->text == sample.text;
            bool identity_pass = sample.prefill_lens_snapshot->identity_max_abs_logit_error.value_or(
                std::numeric_limits<double>::infinity()) <= lens_options.absolute_tolerance;
            for (int round = 1; round < lens_options.validation_rounds; ++round) {
                {
                    auto repeated = run_sample(&lens);
                    report_lens(repeated);
                    token_match = token_match && baseline->token_ids == repeated.token_ids;
                    text_match = text_match && baseline->text == repeated.text;
                    identity_pass = identity_pass && repeated.prefill_lens_snapshot->identity_max_abs_logit_error.value_or(
                        std::numeric_limits<double>::infinity()) <= lens_options.absolute_tolerance;
                }
                used_bytes.push_back(memory_used());
            }
            const auto validation = diagnosticLensValidationReport(
                token_match, text_match, identity_pass, lens_options.validation_rounds,
                used_bytes, lens.identity.checkpoint_fingerprint);
            ctx.logging.logger->log("[LensValidation] " + validation.dump());
        }
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();

        const std::string decoded = trimSampleText(
            atom_insertion ? sample.text : sample.continuation_text,
            static_cast<std::size_t>(max_chars));
        ctx.logging.logger->log("[Sample] step=" + std::to_string(optimizer_step) +
                                " ms=" + std::to_string(elapsed_ms) +
                                " encode_ms=" + std::to_string(sample.encode_ms) +
                                " generation_ms=" + std::to_string(sample.generation_ms) +
                                " decode_ms=" + std::to_string(sample.decode_ms) +
                                " prompt_tokens=" + std::to_string(sample.prompt_token_count) +
                                " sequence_tokens=" + std::to_string(sample.sequence_token_count) +
                                " case=" + diagnostic_id +
                                " prompt=\"" + prompt + "\"");
        ctx.logging.logger->log("[Sample] " + decoded);
    } catch (const std::exception& e) {
        ctx.logging.logger->log(std::string("[Sample] generation failed: ") + e.what());
    }

    // Issue #142b: Check for deferred CUDA errors after executePhase2TextInference().
    // Phase2 inference runs 80+ incremental forward passes (prefill + decode).
    // CUDA kernel launches are async — errors may not surface until the NEXT sync.
    // Without this check, deferred errors corrupt batch N+1's forward pass,
    // triggering an SEH exception that bypasses C++ catch blocks → silent exit.
    {
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            std::string err_msg = "[Sample] CUDA ERROR after generate(): " +
                std::string(cudaGetErrorString(sync_err)) +
                " (code=" + std::to_string(static_cast<int>(sync_err)) + ")";
            ctx.logging.logger->log(err_msg);
            fprintf(stderr, "%s\n", err_msg.c_str());
            // Clear the error so training can attempt to continue
            cudaGetLastError();
        }
        // Also check for sticky errors from kernel launches
        cudaError_t peek_err = cudaGetLastError();
        if (peek_err != cudaSuccess) {
            std::string err_msg = "[Sample] CUDA sticky error after generate(): " +
                std::string(cudaGetErrorString(peek_err)) +
                " (code=" + std::to_string(static_cast<int>(peek_err)) + ")";
            ctx.logging.logger->log(err_msg);
            fprintf(stderr, "%s\n", err_msg.c_str());
        }
    }
}

}  // namespace GRIMText::Training
