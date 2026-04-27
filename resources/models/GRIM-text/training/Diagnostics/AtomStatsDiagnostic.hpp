//======================================================//
//  AtomStatsDiagnostic.hpp
//  Atom-token stats over a batch (count, ratio, per-seq).
//  Lifted verbatim from Phase2_TrainingLoop.cu —
//  the AtomStats struct, computeAtomStats, the
//  shouldLogAtomStats gate, and the inline log block.
//======================================================//

#pragma once

#include <vector>

#include "../../GRIM/grim_language_model_cuda.hpp"

namespace GRIM { namespace Tokenizer { class UniByte; } }
namespace GRIMText { namespace Training { struct TrainingContext; } }

namespace GRIM::Diagnostics {

struct AtomStats {
    int total_atoms = 0;
    int total_tokens = 0;
    int min_atoms = 0;
    int max_atoms = 0;
    double avg_atoms = 0.0;
};

AtomStats computeAtomStats(const std::vector<std::vector<int>>& batch_inputs,
                           const GRIM::Tokenizer::UniByte& tokenizer,
                           std::vector<int>* per_seq_atoms,
                           std::vector<int>* per_seq_lengths);

// shouldLogAtomStats lives in DiagnosticGates.{hpp,cu}

void runAtomStatsDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    int batch_idx);

} // namespace GRIM::Diagnostics
