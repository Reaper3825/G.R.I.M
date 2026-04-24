//======================================================//
//  BoundaryDiagnostic.cu
//  Lifted verbatim from Phase2_TrainingLoop.cu — the
//  "DIAGNOSTIC: Boundary crossing check" scope.
//  Behavior: identical. No logic, gating, ordering, or
//  log-string changes vs. the original inline block.
//======================================================//

#include "BoundaryDiagnostic.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"

#include <sstream>
#include <algorithm>
#include <climits>
#include <cstddef>

namespace GRIM::Diagnostics {

void runBoundaryDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    int batch_idx)
{
    {
        // Use payload geometry — single source of truth
        const size_t max_seq_len = static_cast<size_t>(payload.max_seq_len);
        const size_t total_tokens = static_cast<size_t>(payload.actual_tokens);

        
        const auto& model_cfg_bd = ctx.model->getConfig();
        static bool logged_max_seq = false;
        const bool is_boundary_max_seq = (max_seq_len >= static_cast<size_t>(model_cfg_bd.max_seq_len) && !logged_max_seq);

        
        if (is_boundary_max_seq) {
            std::ostringstream diag;
            diag << "\n[BOUNDARY_DIAGNOSTIC] ========================================\n";
            diag << "[BOUNDARY_DIAGNOSTIC] Batch " << (batch_idx + 1) << " CROSSING BOUNDARY\n";
            
            // Identify which boundary was crossed
            if (is_boundary_max_seq) diag << "[BOUNDARY_DIAGNOSTIC] *** REACHED model.max_seq_len=" << model_cfg_bd.max_seq_len << " ***\n";

            diag << "[BOUNDARY_DIAGNOSTIC] max_seq_len=" << max_seq_len 
                 << " total_tokens=" << total_tokens 
                 << " batch_size=" << payload.batch_size << "\n";
            
            diag << "[BOUNDARY_DIAGNOSTIC] MODEL CONFIG:\n";
            diag << "  d_model=" << model_cfg_bd.d_model << "\n";
            diag << "  max_seq_len=" << model_cfg_bd.max_seq_len << "\n";
            diag << "  num_heads=" << model_cfg_bd.num_heads << "\n";
            diag << "  num_layers=" << model_cfg_bd.num_layers << "\n";
            diag << "  vocab_size=" << model_cfg_bd.vocab_size << "\n";
            
            // Position embedding checks (this IS a valid concern)
            diag << "[BOUNDARY_DIAGNOSTIC] POSITION EMBEDDING CHECKS:\n";
            diag << "  Current max_seq_len in batch: " << max_seq_len << "\n";
            diag << "  Model max_seq_len: " << model_cfg_bd.max_seq_len << "\n";
            diag << "  Position index range needed: [0, " << (max_seq_len - 1) << "]\n";
            if (max_seq_len > static_cast<size_t>(model_cfg_bd.max_seq_len)) {
                diag << "  *** ERROR: Sequence exceeds model max_seq_len! Position embeddings will OOB! ***\n";
            }
            
            // Per-sequence breakdown using payload geometry
            diag << "[BOUNDARY_DIAGNOSTIC] PER-SEQUENCE BREAKDOWN:\n";
            for (int s = 0; s < payload.batch_size; ++s) {
                const int seq_len = payload.seq_lengths[s];
                diag << "  seq[" << s << "]: len=" << seq_len;
                
                // Check for position IDs that would overflow
                if (seq_len > model_cfg_bd.max_seq_len) {
                    diag << " *** OVERFLOW pos=" << seq_len 
                         << " > max=" << model_cfg_bd.max_seq_len << " ***";
                }
                
                // Sample first and last tokens from flat payload
                if (seq_len > 0) {
                    const int flat_start = s * payload.max_seq_len;
                    diag << " tokens[0]=" << payload.input_ids[flat_start];
                    if (seq_len > 1) {
                        diag << " tokens[" << (seq_len-1) << "]=" << payload.input_ids[flat_start + seq_len - 1];
                    }
                }
                diag << "\n";
            }
            
            // Training state checks - TRAINING cache info (not inference KV cache)
            const auto& ts = ctx.model->getTrainingState();
            diag << "[BOUNDARY_DIAGNOSTIC] TRAINING STATE:\n";
            diag << "  cached_batch_size=" << ts.cached_batch_size << "\n";
            diag << "  cached_seq_len=" << ts.cached_seq_len << "\n";
            diag << "  cached_valid_tokens=" << ts.cached_valid_tokens << "\n";
            
            // Training cache allocation check (the correct fields!)
            diag << "  max_cached_batch=" << ts.max_cached_batch << "\n";
            diag << "  max_cached_seq_len=" << ts.max_cached_seq_len << "\n";
            diag << "  max_cached_tokens=" << ts.max_cached_tokens << "\n";
            
            // Check if sequence fits in TRAINING cache — use payload.total_tokens (already batch*max_seq)
            diag << "  Required tokens for this batch: " << payload.total_tokens << "\n";
            if (static_cast<size_t>(payload.total_tokens) > ts.max_cached_tokens) {
                diag << "  *** WARNING: Batch exceeds training cache capacity! ***\n";
                diag << "  *** Need " << payload.total_tokens << " but have " << ts.max_cached_tokens << " ***\n";
            }
            if (max_seq_len > static_cast<size_t>(ts.max_cached_seq_len)) {
                diag << "  *** WARNING: Sequence exceeds max_cached_seq_len! ***\n";
                diag << "  *** max_seq_len=" << max_seq_len << " > max_cached=" << ts.max_cached_seq_len << " ***\n";
            }
            
            // NOTE: FlashAttention v2 uses O(N) tiled attention, NOT O(N²) buffers.
            // No attention buffer check needed - memory scales linearly with seq_len.
            diag << "[BOUNDARY_DIAGNOSTIC] ATTENTION: Using FlashAttention v2 (O(N) memory)\n";
            
            // Token ID sanity check — scan flat payload
            diag << "[BOUNDARY_DIAGNOSTIC] TOKEN ID SANITY:\n";
            int max_token_id = 0;
            int min_token_id = INT_MAX;
            for (int s = 0; s < payload.batch_size; ++s) {
                const int flat_start = s * payload.max_seq_len;
                const int len = payload.seq_lengths[s];
                for (int t = 0; t < len; ++t) {
                    const int tok = payload.input_ids[flat_start + t];
                    max_token_id = std::max(max_token_id, tok);
                    min_token_id = std::min(min_token_id, tok);
                }
            }
            diag << "  Token ID range: [" << min_token_id << ", " << max_token_id << "]\n";
            diag << "  Vocab size: " << model_cfg_bd.vocab_size << "\n";
            if (max_token_id >= static_cast<int>(model_cfg_bd.vocab_size)) {
                diag << "  *** ERROR: Token ID exceeds vocab size! ***\n";
            }
    
            if (is_boundary_max_seq) logged_max_seq = true;
        }
    }
}

} // namespace GRIM::Diagnostics
