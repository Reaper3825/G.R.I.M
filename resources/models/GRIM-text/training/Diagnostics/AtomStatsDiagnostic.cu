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

namespace GRIM::Diagnostics {

AtomStats computeAtomStats(const std::vector<std::vector<int>>& batch_inputs,
                           const GRIM::Tokenizer::UniByte& tokenizer,
                           std::vector<int>* per_seq_atoms,
                           std::vector<int>* per_seq_lengths) {
    AtomStats stats{};
    if (batch_inputs.empty()) {
        return stats;
    }

    stats.min_atoms = std::numeric_limits<int>::max();
    for (const auto& seq : batch_inputs) {
        int atom_count = 0;
        for (int tid : seq) {
            if (tokenizer.isAtomToken(tid)) {
                ++atom_count;
            }
        }
        if (per_seq_atoms) {
            per_seq_atoms->push_back(atom_count);
        }
        if (per_seq_lengths) {
            per_seq_lengths->push_back(static_cast<int>(seq.size()));
        }
        stats.total_atoms += atom_count;
        stats.total_tokens += static_cast<int>(seq.size());
        stats.min_atoms = std::min(stats.min_atoms, atom_count);
        stats.max_atoms = std::max(stats.max_atoms, atom_count);
    }

    stats.avg_atoms = static_cast<double>(stats.total_atoms) /
                      static_cast<double>(batch_inputs.size());
    return stats;
}

// shouldLogAtomStats is defined in DiagnosticGates.cu

void runAtomStatsDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    int batch_idx)
{
    if (shouldLogAtomStats(ctx, batch_idx)) {
        if (!ctx.tokenizer) {
            throw std::runtime_error("AtomStatsDiagnostic requires initialized ctx.tokenizer");
        }
        PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] shouldLogAtomStats=true, creating vectors...\n");
        std::vector<int> per_seq_atoms;
        std::vector<int> per_seq_lengths;
        per_seq_atoms.reserve(payload.batch_size);
        per_seq_lengths.reserve(payload.batch_size);

        // Reconstruct per-sequence views from flat payload for atom detection
        std::vector<std::vector<int>> seq_views;
        seq_views.reserve(payload.batch_size);
        int offset = 0;
        for (int i = 0; i < payload.batch_size; ++i) {
            const int len = payload.seq_lengths[i];
            seq_views.emplace_back(payload.input_ids.begin() + offset,
                                   payload.input_ids.begin() + offset + len);
            offset += payload.max_seq_len; // stride is padded length
        }
        PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] About to call computeAtomStats...\n");
        const AtomStats stats = computeAtomStats(seq_views, *ctx.tokenizer,
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

        const int max_seq_log = std::max(0, ctx.config.hyperparameters.atom_stats_max_seqs);
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
