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
//  The underlying model.generate() path chooses KV decode only for sequence-local
//  geometry. Sequence-coupled centering/projection uses full-context inference.
//  Both modes keep inference state separate from optimizer-owned training state.
//
//  Author: Austin Wadkins
//  Date: April 2026
//======================================================//

#include "DiagnosticInference.hpp"
#include "../../Shared/UnigramByte/Unigram.hpp"
#include "../../Shared/UnigramByte/AtomTable.hpp"

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

void appendTokenText(std::string& result,
                     const GRIM::Tokenizer::UniByte& tokenizer,
                     const GRIM::Tokenizer::TokenLayout& layout,
                     int tid) {
    if (layout.isSpecial(tid)) {
        result += GRIM::Tokenizer::specialTokenText(tid);
        return;
    }
    if (layout.isByte(tid)) {
        const uint8_t byte_val = tokenizer.byteEncoder().tokenToByte(tid);
        result.push_back(static_cast<char>(byte_val));
        return;
    }
    if (layout.isAtom(tid)) {
        result += "<";
        result += GRIM::Tokenizer::atomTypeName(GRIM::Tokenizer::tokenIdToAtomType(tid));
        result += ">";
        return;
    }
    if (layout.isUnigram(tid)) {
        const auto* piece = tokenizer.unigramLM().getPiece(tid);
        if (!piece) {
            throw std::runtime_error("DiagnosticInference decode: unigram token_id=" +
                                     std::to_string(tid) + " has no backing piece");
        }
        result += piece->text;
        return;
    }
    throw std::runtime_error("DiagnosticInference decode: token_id=" + std::to_string(tid) +
                             " is outside the GRMT/tokenizer vocab layout");
}

std::string decodeWithAtomSideChannel(
        const GRIM::Tokenizer::UniByte& tokenizer,
        const std::vector<int>& token_ids,
        const std::vector<float>& numeric_values,
        const std::vector<uint8_t>& atom_mask,
        const std::vector<uint32_t>& atom_entry_ids,
        const GRIM::Tokenizer::AtomTable* atom_table) {
    std::string result;
    const size_t n = token_ids.size();
    const GRIM::Tokenizer::TokenLayout layout = tokenizer.tokenLayout();

    for (size_t i = 0; i < n; ++i) {
        const int tid = token_ids[i];

        if (layout.isByte(tid)) {
            appendTokenText(result, tokenizer, layout, tid);
            continue;
        }

        if (layout.isAtom(tid)) {
            if (atom_table && i < atom_entry_ids.size() &&
                atom_entry_ids[i] != GRIM::Tokenizer::kAtomEntryNone) {
                const auto entry = atom_table->getAtom(atom_entry_ids[i]);
                if (entry) {
                    result += atom_table->atomToString(*entry);
                    continue;
                }
            }

            if (i < atom_mask.size() && atom_mask[i] != 0 &&
                i < numeric_values.size()) {
                result += GRIM::Tokenizer::formatNumericValue(numeric_values[i]);
                continue;
            }

            appendTokenText(result, tokenizer, layout, tid);
            continue;
        }

        appendTokenText(result, tokenizer, layout, tid);
    }

    return result;
}

}  // anonymous namespace

//======================================================//
//  logDiagnosticSample — public entry point
//======================================================//

void logDiagnosticSample(TrainingContext& ctx, TrainingLoopState& state) {
    const auto& hp = ctx.config.hyperparameters;
    const int interval = readEnvInt("GRIM_SAMPLE_INTERVAL", hp.log_interval);
    if (interval <= 0) {
        return;
    }

    const int optimizer_step = ctx.optimizer.optimizer_step.step;
    if (optimizer_step <= 0 || optimizer_step % interval != 0 || optimizer_step == state.last_sample_step) {
        return;
    }
    state.last_sample_step = optimizer_step;

    if (!ctx.model || !ctx.tokenizer || !ctx.logging.logger) {
        return;
    }
    const auto& tokenizer = *ctx.tokenizer;

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

    auto prompt_result = tokenizer.tokenizeWithMetadata(prompt);
    std::vector<int> prompt_tokens = std::move(prompt_result.token_ids);
    std::vector<float> prompt_numeric_values = std::move(prompt_result.token_numeric_values);
    std::vector<uint8_t> prompt_atom_mask = std::move(prompt_result.token_atom_mask);
    std::vector<uint32_t> prompt_atom_flags = std::move(prompt_result.token_atom_flags);
    auto prompt_atom_table = prompt_result.atom_table;
    std::vector<uint32_t> prompt_atom_entry_ids = std::move(prompt_result.atom_entry_ids);
    if (prompt_tokens.empty()) {
        ctx.logging.logger->log("[Sample] prompt tokenization returned empty tokens");
        return;
    }

    const int max_seq_len = hp.architecture.max_seq_len;
    if (max_seq_len > 1 && static_cast<int>(prompt_tokens.size()) >= max_seq_len) {
        const size_t keep = static_cast<size_t>(max_seq_len - 1);
        const size_t drop = prompt_tokens.size() - keep;
        prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.begin() + drop);
        if (prompt_numeric_values.size() < drop) {
            throw std::runtime_error("DiagnosticInference: prompt_numeric_values shorter than truncated token span");
        }
        prompt_numeric_values.erase(prompt_numeric_values.begin(),
                                     prompt_numeric_values.begin() + drop);
        if (prompt_atom_mask.size() < drop) {
            throw std::runtime_error("DiagnosticInference: prompt_atom_mask shorter than truncated token span");
        }
        prompt_atom_mask.erase(prompt_atom_mask.begin(),
                                prompt_atom_mask.begin() + drop);
        if (prompt_atom_flags.size() < drop) {
            throw std::runtime_error("DiagnosticInference: prompt_atom_flags shorter than truncated token span");
        }
        prompt_atom_flags.erase(prompt_atom_flags.begin(),
                                prompt_atom_flags.begin() + drop);
        if (prompt_atom_entry_ids.size() < drop) {
            throw std::runtime_error("DiagnosticInference: prompt_atom_entry_ids shorter than truncated token span");
        }
        prompt_atom_entry_ids.erase(prompt_atom_entry_ids.begin(),
                                     prompt_atom_entry_ids.begin() + drop);
    }

    // Use generation config from ai_config.json (configurable strategy, penalties, etc.)
    GRIM::HyperParameters::GenerationConfig cfg = ctx.config.generation;
    cfg.max_new_tokens = max_new_tokens;
    if (cfg.min_new_tokens <= 0) {
        cfg.min_new_tokens = std::max(1, max_new_tokens / 4);
    }
    cfg.num_return_sequences = 1;
    cfg.eos_token_id = GRIM::Tokenizer::EOS_TOKEN_ID;
    cfg.pad_token_id = GRIM::Tokenizer::PAD_TOKEN_ID;
    // Seed from optimizer step for reproducible but varied samples per step
    cfg.seed = static_cast<unsigned int>(optimizer_step);

    try {
        const auto start = std::chrono::steady_clock::now();
        const auto& model_cfg = ctx.model->getConfig();
        const std::vector<int32_t> prompt_token_to_slot_map;
        auto prompt_payload = GRIM::Batching::buildInferenceBatchPayload(
            prompt_tokens,
            prompt_numeric_values,
            prompt_atom_mask,
            prompt_atom_flags,
            prompt_atom_table,
            prompt_atom_entry_ids,
            prompt_token_to_slot_map,
            model_cfg.vocab_size,
            static_cast<size_t>(model_cfg.max_cached_batch),
            static_cast<size_t>(model_cfg.max_cached_seq_len),
            model_cfg.execution_block_num_slots);
        std::vector<GRIM::GeneratedSequence> outputs = ctx.model->generate(
            prompt_payload,
            &cfg);
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();

        if (outputs.empty()) {
            ctx.logging.logger->log("[Sample] step=" + std::to_string(optimizer_step) + " empty output");
            return;
        }

        // DEBUG: Log raw token IDs to diagnose mode collapse
        const auto& gen_tokens = outputs.front().token_ids;
        const size_t prompt_len = prompt_tokens.size();
        std::ostringstream token_debug;
        token_debug << "[Sample] step=" << optimizer_step << " generated_tokens(first20): [";
        for (size_t ti = prompt_len; ti < std::min(prompt_len + 20, gen_tokens.size()); ++ti) {
            token_debug << gen_tokens[ti];
            if (ti < std::min(prompt_len + 19, gen_tokens.size() - 1)) token_debug << ", ";
        }
        token_debug << "] total_generated=" << (gen_tokens.size() - prompt_len);
        ctx.logging.logger->log(token_debug.str());

        // DEBUG: Decode individual token IDs to see what they map to
        if (gen_tokens.size() > prompt_len) {
            int first_gen_token = gen_tokens[prompt_len];
            const GRIM::Tokenizer::TokenLayout layout = tokenizer.tokenLayout();
            std::string first_decoded;
            if (layout.isAtom(first_gen_token)) {
                appendTokenText(first_decoded, tokenizer, layout, first_gen_token);
            } else {
                first_decoded = tokenizer.decode({first_gen_token});
            }
            std::ostringstream tid_decode;
            tid_decode << "[TokenDecode] token_id=" << first_gen_token
                       << " decodes_to=\"" << first_decoded << "\""
                       << " (len=" << first_decoded.size() << " bytes)";
            ctx.logging.logger->log(tid_decode.str());
        }

        std::string decoded = decodeWithAtomSideChannel(tokenizer,
                                                          outputs.front().token_ids,
                                                          outputs.front().token_numeric_values,
                                                          outputs.front().token_atom_mask,
                                                          outputs.front().atom_entry_ids,
                                                          outputs.front().context_atom_table.get());
        decoded = trimSampleText(decoded, static_cast<std::size_t>(max_chars));
        ctx.logging.logger->log("[Sample] step=" + std::to_string(optimizer_step) +
                                " ms=" + std::to_string(elapsed_ms) +
                                " prompt=\"" + prompt + "\"");
        ctx.logging.logger->log("[Sample] " + decoded);
    } catch (const std::exception& e) {
        ctx.logging.logger->log(std::string("[Sample] generation failed: ") + e.what());
    }

    // Issue #142b: Check for deferred CUDA errors after generate().
    // generate() runs 80+ incremental forward passes (forwardInit + forwardStep).
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
