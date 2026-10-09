// CPU-only test of the real BatchPayload validator; no CUDA/runtime targets.
#include "../Shared/Batching/BatchPayload.hpp"
#include <cassert>
#include <iostream>

using GRIM::Batching::BatchPayload;
using GRIM::Batching::BatchPayloadMode;

BatchPayload lmPayload() {
    BatchPayload p;
    p.batch_size = 1;
    p.max_seq_len = p.total_tokens = 4;
    p.actual_tokens = 3;
    p.padding_tokens = 1;
    p.vocab_size = 1000;
    p.valid_tokens = p.lm_valid_tokens = 2;
    p.seq_lengths = {3};
    p.valid_target_counts = {2};
    p.prompt_lengths = {0};
    p.prompt_end_positions = {-1};
    p.named_concept_spans.resize(1);
    p.input_ids = {4, 5, 3, 1};
    p.target_ids = {5, 3, -1, -1};
    p.numeric_values.assign(4, 0);
    p.atom_mask.assign(4, 0);
    p.atom_aux_target_mask.assign(4, 0);
    p.atom_flags.assign(4, 0);
    p.atom_entry_ids.assign(4, GRIM::Tokenizer::kAtomEntryNone);
    p.token_local_atom_indices.assign(4, GRIM::Tokenizer::kLocalAtomIndexNone);
    p.token_to_slot_index_map.assign(4, -1);
    p.fits_in_cache = true;
    return p;
}

void rejects(const BatchPayload& p, const char* diagnostic) {
    try {
        p.validate("batch_payload_lm_validation_host_test");
    } catch (const std::runtime_error& e) {
        assert(std::string(e.what()).find(diagnostic) != std::string::npos);
        return;
    }
    throw std::runtime_error("Expected invalid payload to be rejected");
}

int main() {
    const auto valid = lmPayload();
    valid.validate("valid LM"); // EOS supervised, final/padding targets masked.
    auto p = valid;
    p.target_ids[0] = 0; // Range contract includes zero; masking policy is upstream.
    p.target_ids[1] = p.vocab_size - 1;
    p.validate("range endpoints");
    p = valid;
    p.target_ids[0] = -2;
    rejects(p, "invalid LM target");
    p = valid;
    p.target_ids[0] = p.vocab_size;
    rejects(p, "invalid LM target");
    p = valid;
    p.target_ids.pop_back();
    rejects(p, "target_ids.size()");
    p = valid;
    p.target_ids[0] = -1;
    rejects(p, "realized LM target count");
    p = valid;
    p.lm_valid_tokens = 1;
    rejects(p, "realized LM target count");
    p = valid;
    p.valid_tokens = 1;
    p.valid_target_counts = {1};
    rejects(p, "realized LM target count");

    // Inference has no LM objective and must retain its existing validation.
    p = valid;
    p.mode = BatchPayloadMode::InferencePrefill;
    p.valid_tokens = p.lm_valid_tokens = 0;
    p.valid_target_counts = {0};
    p.target_ids.assign(4, -1);
    p.atom_aux_target_mask.clear();
    p.validate("valid inference");
    p.target_ids[0] = 5;
    rejects(p, "inference BatchPayload.target_ids");
    std::cout << "BatchPayload LM validation host tests passed\n";
}
