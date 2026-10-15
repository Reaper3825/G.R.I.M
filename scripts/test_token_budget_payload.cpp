// Exercise real host payload construction: partial batches retain context width
// and row boundaries, with unchanged token IDs and target masks.
#include "../resources/models/GRIM-text/Shared/Batching/BatchPayload.hpp"
#include "../resources/models/GRIM-text/Shared/Batching/Batching_GPU.hpp"
#include "../resources/models/GRIM-text/Shared/TokenizerArtifacts/GrmtSequence.hpp"
#include <cassert>
#include <iostream>

using namespace GRIM;

int main() try {
    TokenizerArtifacts::GrmtSequence row;
    row.token_ids = {4, 5, 3};
    row.targets = {5, 3, -1};
    row.token_numeric_values.assign(3, 0);
    row.token_atom_mask.assign(3, 0);
    row.token_atom_flags.assign(3, 0);
    row.token_atom_aux_target_mask.assign(3, 0);
    row.atom_entry_ids.assign(3, Tokenizer::kAtomEntryNone);
    row.token_local_atom_indices.assign(3, Tokenizer::kLocalAtomIndexNone);
    row.local_atom_table = std::make_shared<Tokenizer::SequenceLocalAtomTable>();
    const std::vector<TokenizerArtifacts::GrmtSequence*> views{&row, &row};
    const auto layout = Tokenizer::tokenLayoutFromActualVocabOrThrow(1000, "test");
    const auto payload = Batching::buildBatchPayload(
        Batching::BatchAssignment{{0}}, views, 1000, layout, 4, 1024, false);
    assert(payload.batch_size == 1 && payload.max_seq_len == 1024);
    assert(payload.total_tokens == 1024 && payload.actual_tokens == 3);
    assert(payload.padding_tokens == 1021 && payload.lm_valid_tokens == 2);
    assert(payload.input_ids[0] == 4 && payload.input_ids[2] == 3);
    assert(payload.input_ids[3] == Tokenizer::PAD_TOKEN_ID);
    assert(payload.target_ids[0] == 5 && payload.target_ids[2] == -1);
    assert(payload.target_ids.back() == -1);
    const auto two = Batching::buildBatchPayload(
        Batching::BatchAssignment{{0, 1}}, views, 1000, layout, 4, 1024, false);
    assert(two.total_tokens == 2048 && two.batch_size == 2);
    assert(two.input_ids[1024] == 4 && two.input_ids[1026] == 3);
    assert(two.target_ids[1023] == -1 && two.target_ids[1026] == -1);
    bool rejected = false;
    try {
        Batching::buildBatchPayload(
            Batching::BatchAssignment{{0, 1}}, views, 1000, layout, 1, 1024, false);
    } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    std::cout << "Token-budget payload context and boundary tests passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
