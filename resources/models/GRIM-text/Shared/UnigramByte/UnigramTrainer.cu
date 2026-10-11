// Unigram training entrypoint: validates settings and coordinates named stages.

#include "Unigram.hpp"
#include "Training/UnigramTrainingValidation.hpp"
#include "Training/UnigramCorpusPreparation.hpp"
#include "Training/UnigramCharacterDiagnostics.hpp"
#include "Training/UnigramCandidateAdmission.hpp"
#include "Training/UnigramVocabularyFitting.hpp"
#include "Training/SubwordMining.hpp"
#include "VocabWriteOp.hpp"

#include <iostream>
#include <utility>

namespace GRIM {
namespace Tokenizer {

bool UnigramLM::trainFromCorpus(const std::vector<std::string>& texts,
                                 const std::vector<std::vector<AtomSpan>>& atom_spans,
                                 int target_vocab_size,
                                 float character_coverage,
                                 int min_subword_freq,
                                 bool prune_during_mining,
                                 bool enable_parallel_subword_mining,
                                 int subword_mining_workers,
                                 size_t subword_mining_max_bytes,
                                 const std::vector<UnigramPiece>& manual_pieces) {
    validateTrainFromCorpusParameters(texts, target_vocab_size, character_coverage,
                                      min_subword_freq, subword_mining_workers);
    const int total_piece_target = unigramPieceBudgetWithManualAllowanceOrThrow(
        target_vocab_size, manual_pieces.size());

    // An empty atom_spans vector means "no atom spans for any text". Expand it
    // to a per-text empty list so the rest of the pipeline can index uniformly.
    std::vector<std::vector<AtomSpan>> expanded_empty_atom_spans;
    if (atom_spans.empty()) {
        expanded_empty_atom_spans.resize(texts.size());
    }
    const std::vector<std::vector<AtomSpan>>& effective_atom_spans =
        atom_spans.empty() ? expanded_empty_atom_spans : atom_spans;

    const AtomSpanValidationTotals original_atom_totals =
        validateOriginalAtomSpansBeforeNormalization(texts, effective_atom_spans);

    last_training_runtime_report_ = UnigramTrainingRuntimeReport{};

    std::cout << "[UnigramLM] Training vocabulary from " << texts.size() 
              << " texts (ordinary_piece_target=" << target_vocab_size
              << ", manual_piece_allowance=" << manual_pieces.size()
              << ", total_piece_target=" << total_piece_target << ")" << std::endl;
    std::cout << "[UnigramLM] min_subword_freq=" << min_subword_freq 
              << ", prune_during_mining=" << (prune_during_mining ? "true" : "false")
              << ", parallel_subword_mining=" << (enable_parallel_subword_mining ? "true" : "false")
              << ", subword_mining_workers=" << subword_mining_workers << std::endl;

    if (prune_during_mining) {
        std::cout << "[UnigramLM] prune_during_mining=true requested; exact global subword counting is enforced, so mining-time pruning is disabled"
                  << std::endl;
    }

    if (original_atom_totals.span_count > 0) {
        std::cout << "[UnigramLM] Atom-aware training: " << original_atom_totals.span_count
                  << " atom spans (" << (original_atom_totals.byte_count / 1024) << " KB) will be skipped" << std::endl;
    }
    
    const UnigramPreparedCorpus prepared_corpus = prepareUnigramTrainingCorpus(
        texts, effective_atom_spans, manual_pieces);
    const auto& norm_texts = prepared_corpus.normalized_texts;
    const auto& norm_training_exclusion_spans = prepared_corpus.normalized_exclusion_spans;
    const auto& training_units = norm_texts;
    const size_t training_segment_max_length = prepared_corpus.max_segment_length;
    
    size_t total_corpus_bytes = 0;
    for (const auto& text : texts) {
        total_corpus_bytes += text.size();
    }
    std::cout << "[UnigramLM] Total corpus size: " << (total_corpus_bytes / (1024*1024)) << " MB" << std::endl;
    
    const UnigramSubwordMiningResult mining_result = mineUnigramSubwordsFromTrainingUnits(
        UnigramSubwordMiningRequest{
            training_units,
            norm_training_exclusion_spans,
            enable_parallel_subword_mining,
            subword_mining_workers,
            subword_mining_max_bytes,
            "[UnigramLM]"});
    
    runUnigramCharacterDiagnostics(norm_texts, norm_training_exclusion_spans,
        character_coverage, UnigramVocabWriteTarget{pieces_, piece_to_id_});

    admitUnigramVocabularyCandidates(*this, mining_result, min_subword_freq,
        target_vocab_size, manual_pieces);

    fitUnigramVocabulary(UnigramVocabWriteTarget{pieces_, piece_to_id_},
        norm_texts, norm_training_exclusion_spans, enable_byte_fallback_,
        total_piece_target, subword_mining_workers);

    finalizeTrainingRuntime(training_segment_max_length);

    std::cout << "[UnigramLM] Training complete. Final vocab size: " << pieces_.size() << std::endl;
    return true;
}

} // namespace Tokenizer
} // namespace GRIM
