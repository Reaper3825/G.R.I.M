//======================================================//
//  AtomStatsDiagnostic.cu
//  Lifted verbatim from Phase2_TrainingLoop.cu — the
//  AtomStats struct, computeAtomStats helper,
//  shouldLogAtomStats gate, and the inline AtomStats
//  log block (per-batch summary + per-seq breakdown).
//  Behavior: identical. No logic, gating, ordering, or
//  log-string changes vs. the original inline block.
//======================================================//

#include "AtomStatsDiagnostic.hpp"
#include "DiagnosticGates.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace GRIM::Diagnostics {

AtomStats computeAtomStats(const GRIM::Batching::BatchPayload& payload,
                           const GRIM::Tokenizer::TokenLayout& token_layout,
                           std::vector<int>* per_seq_atoms,
                           std::vector<int>* per_seq_lengths) {
    AtomStats stats{};
    if (payload.batch_size == 0) {
        return stats;
    }

    if (payload.max_seq_len <= 0) {
        throw std::runtime_error(
            "computeAtomStats: payload.max_seq_len must be > 0");
    }
    if (static_cast<int>(payload.seq_lengths.size()) != payload.batch_size) {
        throw std::runtime_error(
            "computeAtomStats: payload.seq_lengths.size() != payload.batch_size");
    }
    const int required_tokens = payload.batch_size * payload.max_seq_len;
    if (static_cast<int>(payload.input_ids.size()) < required_tokens) {
        throw std::runtime_error(
            "computeAtomStats: payload.input_ids is smaller than batch_size * max_seq_len");
    }

    stats.min_atoms = std::numeric_limits<int>::max();
    int offset = 0;
    for (int row = 0; row < payload.batch_size; ++row) {
        const int seq_len = payload.seq_lengths[row];
        if (seq_len < 0 || seq_len > payload.max_seq_len) {
            throw std::runtime_error(
                "computeAtomStats: payload.seq_lengths[row] out of valid range");
        }

        int atom_count = 0;
        for (int col = 0; col < seq_len; ++col) {
            const int tid = payload.input_ids[offset + col];
            if (token_layout.isAtom(tid)) {
                ++atom_count;
            }
        }

        if (per_seq_atoms) {
            per_seq_atoms->push_back(atom_count);
        }
        if (per_seq_lengths) {
            per_seq_lengths->push_back(seq_len);
        }

        stats.total_atoms += atom_count;
        stats.total_tokens += seq_len;
        stats.min_atoms = std::min(stats.min_atoms, atom_count);
        stats.max_atoms = std::max(stats.max_atoms, atom_count);

        offset += payload.max_seq_len;
    }

    stats.avg_atoms = static_cast<double>(stats.total_atoms) /
                      static_cast<double>(payload.batch_size);
    return stats;
}

// shouldLogAtomStats is defined in DiagnosticGates.cu

void runAtomStatsDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    int batch_idx)
{
    if (shouldLogAtomStats(ctx, batch_idx)) {
        const auto token_layout = GRIM::Tokenizer::tokenLayoutFromActualVocabOrThrow(
            static_cast<std::uint32_t>(payload.vocab_size),
            "runAtomStatsDiagnostic");
        PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] shouldLogAtomStats=true, creating vectors...\n");
        std::vector<int> per_seq_atoms;
        std::vector<int> per_seq_lengths;
        per_seq_atoms.reserve(payload.batch_size);
        per_seq_lengths.reserve(payload.batch_size);

        PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] About to call computeAtomStats...\n");
        const AtomStats stats = computeAtomStats(payload, token_layout,
                                                 &per_seq_atoms, &per_seq_lengths);
        PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] computeAtomStats returned\n");
        const double atom_ratio = stats.total_tokens > 0
            ? static_cast<double>(stats.total_atoms) / static_cast<double>(stats.total_tokens)
            : 0.0;

        PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] Building atom_msg...\n");
        std::ostringstream atom_msg;
        atom_msg << "[AtomStats] batch=" << (batch_idx + 1)
                 << " seqs=" << payload.batch_size
                 << " atoms=" << stats.total_atoms
                 << " tokens=" << stats.total_tokens
                 << " atom_ratio=" << std::fixed << std::setprecision(4) << atom_ratio
                 << " min=" << stats.min_atoms
                 << " max=" << stats.max_atoms
                 << " avg=" << std::fixed << std::setprecision(2) << stats.avg_atoms;
        PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] About to log atom_msg...\n");
        ctx.logging.logger->log(atom_msg.str());
        PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] atom_msg logged\n");

        const auto runtime_hp =
            GRIM::HyperParameters::trainingRuntimeControlHP(ctx.config);
        const int max_seq_log = std::max(0, runtime_hp.atom_stats_max_seqs);
        if (max_seq_log > 0 && !per_seq_atoms.empty()) {
            const int to_log = std::min<int>(max_seq_log,
                                             static_cast<int>(per_seq_atoms.size()));
            std::ostringstream per_seq_msg;
            per_seq_msg << "[AtomStats] per_seq=";
            for (int i = 0; i < to_log; ++i) {
                const int seq_len = per_seq_lengths[i];
                const int atom_count = per_seq_atoms[i];
                const double ratio = seq_len > 0
                    ? static_cast<double>(atom_count) / static_cast<double>(seq_len)
                    : 0.0;
                per_seq_msg << i << ":" << atom_count << "/" << seq_len
                            << "(" << std::fixed << std::setprecision(3) << ratio << ")";
                if (i + 1 < to_log) {
                    per_seq_msg << " ";
                }
            }
            if (static_cast<int>(per_seq_atoms.size()) > to_log) {
                per_seq_msg << " ...";
            }
            ctx.logging.logger->log(per_seq_msg.str());
        }
    }
}

} // namespace GRIM::Diagnostics
