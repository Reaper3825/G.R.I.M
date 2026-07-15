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
#include "../Phases/Phase2_InferenceLoop.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/DataLoader/DataLoader.hpp"

#include <iostream>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace GRIMText::Training {
namespace {

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

    if (!ctx.model || !ctx.logging.logger) {
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

    const std::string prompt = readEnvString("GRIM_SAMPLE_PROMPT",
        "A tank holds 120 liters. After using some, 84 liters remain. How much was used?");
    const int max_new_tokens = readEnvInt("GRIM_SAMPLE_TOKENS", 100);
    const int max_chars = readEnvInt("GRIM_SAMPLE_MAX_CHARS", 300);
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

    try {
        auto tokenizer = LoadInferenceTokenizer(ctx.config, *ctx.logging.logger);
        const auto start = std::chrono::steady_clock::now();
        auto sample = executePhase2TextInference(ctx, *tokenizer, prompt, cfg);
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();

        std::string decoded = trimSampleText(sample.text, static_cast<std::size_t>(max_chars));
        ctx.logging.logger->log("[Sample] step=" + std::to_string(optimizer_step) +
                                " ms=" + std::to_string(elapsed_ms) +
                                " encode_ms=" + std::to_string(sample.encode_ms) +
                                " generation_ms=" + std::to_string(sample.generation_ms) +
                                " decode_ms=" + std::to_string(sample.decode_ms) +
                                " prompt_tokens=" + std::to_string(sample.prompt_token_count) +
                                " sequence_tokens=" + std::to_string(sample.sequence_token_count) +
                                " exec_gate=" +
                                    (!sample.execution_control.gate_evaluated
                                        ? "not_evaluated"
                                        : (sample.execution_control.gate_predicted_class == 1
                                            ? "execute" : "noop")) +
                                " exec_steps=" + std::to_string(sample.execution_control.steps.size()) +
                                " exec_model_stop=" +
                                    std::to_string(sample.execution_control.stopped_by_model ? 1 : 0) +
                                " exec_max_stop=" +
                                    std::to_string(sample.execution_control.stopped_at_max_steps ? 1 : 0) +
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
