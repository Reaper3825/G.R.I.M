// Fitted trie upload and runtime readiness report.

#include "../Unigram.hpp"

#include <iostream>
#include <stdexcept>

namespace GRIM {
namespace Tokenizer {

void UnigramLM::finalizeTrainingRuntime(size_t training_segment_max_length) {
    // Final trie build with converged scores
    buildTrie();

    if (!initGPUForMaxSequenceLength(training_segment_max_length)) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: final tokenizer runtime upload failed after final buildTrie()");
    }

    last_training_runtime_report_.required_viterbi_workspace_length = training_segment_max_length;
    last_training_runtime_report_.finalized_trie_generation = trie_generation_;
    last_training_runtime_report_.final_piece_count = pieceCount();

    const auto runtime_snapshot = runtimeStateSnapshot();
    std::cout << "[UnigramLM] Final tokenizer runtime state: required_viterbi_workspace_length="
              << last_training_runtime_report_.required_viterbi_workspace_length
              << ", uploaded_workspace_max_length=" << runtime_snapshot.workspace_max_length
              << ", trie_generation=" << runtime_snapshot.live_trie_generation
              << std::endl;
}

} // namespace Tokenizer
} // namespace GRIM
