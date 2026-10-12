// Focused CPU policy tests. No trainer/runtime target or GPU execution.
#include "HyperParameters/HyperparameterGroupings.hpp"
#include "UnigramByte/Training/UnigramTrainingValidation.hpp"
#include "UnigramByte/Training/UnigramEMConvergence.hpp"
#include "UnigramByte/Training/UnigramMaximizationStep.hpp"
// Include mining implementation to exercise internal scheduling/sampling helpers.
#include "UnigramByte/Training/SubwordMining.cu"

#include <cassert>
#include <fstream>
#include <limits>

// Windows CRT assertions can fail without printing in a noninteractive run.
#undef assert
#define assert(condition) do { if (!(condition)) { \
    std::cerr << "Failed at line " << __LINE__ << ": " << #condition << std::endl; \
    std::exit(1); } } while (false)

using namespace GRIM::HyperParameters;
using namespace GRIM::Tokenizer;

namespace {
int expectation_calls = 0;
double likelihood_delta = 0;
template <typename Fn> void requiresFailure(Fn fn) {
    bool failed = false;
    try { fn(); } catch (const std::runtime_error&) { failed = true; }
    assert(failed);
}
}

// Replace only the corpus E-step for the convergence-control unit test.
namespace GRIM::Tokenizer {
UnigramExpectationStepResult runUnigramExpectationStep(const UnigramExpectationStepRequest&) {
    UnigramExpectationStepResult result;
    result.expected_learned_piece_tokens = 1;
    result.learned_token_counts[tokenIdForIndex(0)] = 1;
    result.log_likelihood = -100 + likelihood_delta * expectation_calls++;
    return result;
}
}

int main(int argc, char** argv) try {
    assert(argc == 2);
    GRIM::Config::AiConfigSnapshot snapshot;
    std::ifstream config_file(argv[1]);
    config_file >> snapshot.document;
    // These model-semantic leaves normally enter the finalized document from
    // the compiled model artifact. This test exercises only the training leaves.
    snapshot.document["training"]["config"]["tokenizer_enable_atom_reasoning"] = true;
    snapshot.document["training"]["config"]["tokenizer_enable_byte_fallback"] = true;
    snapshot.document["training"]["config"]["tokenizer_add_bos"] = true;
    snapshot.document["training"]["config"]["tokenizer_add_eos"] = true;
    snapshot.document["training"]["config"]["max_seq_len"] = 1024;
    auto hp = tokenizerHP(snapshot);
    validateTrainFromCorpusParameters({"test"}, hp);

    // Config edits must reach both typed hyperparameters and the tokenizer view.
    auto& config = snapshot.document["training"]["config"];
    config["tokenizer_em_max_iterations"] = 7;
    config["tokenizer_repetition_max_alpha_run"] = 9;
    config["tokenizer_subword_mining_sample_span_bytes"] = 64;
    hp = tokenizerHP(snapshot);
    const auto typed = loadLanguageModelConfig(snapshot.document);
    assert(typed.tokenizer_em_max_iterations == 7 && hp.em_max_iterations == 7);
    assert(typed.tokenizer_repetition_max_alpha_run == 9 && hp.repetition_max_alpha_run == 9);
    assert(typed.tokenizer_subword_mining_sample_span_bytes == 64 && hp.subword_mining_sample_span_bytes == 64);
    config.erase("tokenizer_em_max_iterations");
    bool missing = false;
    try { tokenizerHP(snapshot); } catch (const nlohmann::json::exception&) { missing = true; }
    assert(missing);
    config["tokenizer_em_max_iterations"] = 7;

    auto invalid = hp;
    invalid.prune_shrink_factor = 1;
    requiresFailure([&] { validateTrainFromCorpusParameters({"test"}, invalid); });
    invalid = hp;
    invalid.em_convergence_tolerance = std::numeric_limits<double>::quiet_NaN();
    requiresFailure([&] { validateTrainFromCorpusParameters({"test"}, invalid); });
    invalid = hp;
    invalid.subword_mining_min_chunk_texts = invalid.subword_mining_max_chunk_texts + 1;
    requiresFailure([&] { validateTrainFromCorpusParameters({"test"}, invalid); });
    invalid = hp;
    invalid.repetition_pattern_min_repeats = 1;
    requiresFailure([&] { validateTrainFromCorpusParameters({"test"}, invalid); });
    requiresFailure([&] { validateTrainFromCorpusParameters({"test"}, TokenizerHP{}); });

    hp.repetition_max_alpha_run = 3;
    hp.repetition_pattern_min_bytes = 32;
    assert(isUnigramRepetitionNoise("aaaaZ", hp));
    hp.repetition_max_alpha_run = 4;
    assert(!isUnigramRepetitionNoise("aaaaZ", hp));
    hp.repetition_pattern_min_bytes = 6;
    assert(isUnigramRepetitionNoise("abcabcabc", hp));
    hp.repetition_pattern_min_repeats = 4;
    assert(!isUnigramRepetitionNoise("abcabcabc", hp));
    assert(isUnigramRepetitionNoise("wordword", hp));
    hp.repetition_doubled_min_half_bytes = 5;
    assert(!isUnigramRepetitionNoise("wordword", hp));
    hp.repetition_max_whitespace_run = 2;
    assert(isUnigramRepetitionNoise("a a a", hp));
    hp.repetition_stutter_min_words = 4;
    assert(!isUnigramRepetitionNoise("a a a", hp));
    assert(!isValidUnigramVocabCharacter("\xc2\xa0", hp));
    hp.exclude_nonbreaking_space = false;
    assert(isValidUnigramVocabCharacter("\xc2\xa0", hp));
    assert(!isValidUnigramVocabCharacter("\xee\x80\x80", hp));
    hp.exclude_unicode_private_use = false;
    assert(isValidUnigramVocabCharacter("\xee\x80\x80", hp));
    assert(!isValidUnigramSubword("a\nb", hp));

    hp.enable_parallel_subword_mining = true;
    hp.subword_mining_workers = 2;
    hp.subword_mining_parallel_min_texts = 3;
    assert(resolveSubwordMiningWorkerCount(hp, 2) == 1);
    assert(resolveSubwordMiningWorkerCount(hp, 3) == 2);
    hp.subword_mining_chunks_per_worker = 2;
    hp.subword_mining_min_chunk_texts = 1;
    hp.subword_mining_max_chunk_texts = 100;
    assert(resolveSubwordMiningChunkSize(2, 100, hp) == 25);
    hp.subword_mining_chunks_per_worker = 5;
    assert(resolveSubwordMiningChunkSize(2, 100, hp) == 10);
    hp.subword_mining_max_chunk_texts = 5;
    assert(resolveSubwordMiningChunkSize(2, 100, hp) == 5);
    const std::vector<std::string> large_text{std::string(4096, 'x')};
    hp.subword_mining_sample_span_bytes = 512;
    const auto coarse = buildSubwordMiningPlan(large_text, 4096, 1024, true, hp);
    hp.subword_mining_sample_span_bytes = 64;
    const auto fine = buildSubwordMiningPlan(large_text, 4096, 1024, true, hp);
    assert(fine.sampled_spans > coarse.sampled_spans);

    // Same configured policy must produce identical sequential/parallel counts.
    hp = tokenizerHP(snapshot);
    hp.subword_mining_initial_count_reserve = 0;
    hp.subword_mining_worker_count_reserve = 0;
    hp.subword_mining_parallel_min_texts = 1;
    hp.subword_mining_min_chunk_texts = 1;
    hp.subword_mining_workers = 2;
    std::vector<std::string> texts{"abc abc", "def def", "abc def"};
    std::vector<std::vector<AtomSpan>> spans(texts.size());
    hp.enable_parallel_subword_mining = false;
    const auto sequential = mineUnigramSubwordsFromTrainingUnits({texts, spans, hp, "[Test]"});
    hp.enable_parallel_subword_mining = true;
    const auto parallel = mineUnigramSubwordsFromTrainingUnits({texts, spans, hp, "[Test]"});
    assert(sequential.subword_counts == parallel.subword_counts);
    assert(parallel.worker_count == 2);

    requireUnigramLearnedPosteriorMassNotByteFallbackDominated(0.75, 1, 0.5, "test", "test");
    requiresFailure([&] { requireUnigramLearnedPosteriorMassNotByteFallbackDominated(0.75, 1, 1, "test", "test"); });
    std::vector<UnigramPiece> pieces(2);
    const UnigramExpectationStepRequest expectation{pieces, texts, spans, false};
    hp.em_max_iterations = 3;
    hp.em_convergence_tolerance = 0.001;
    hp.piece_count_smoothing = 0.5;
    hp.em_fail_on_iteration_limit = false;
    likelihood_delta = 1;
    expectation_calls = 0;
    assert(runUnigramEMToConvergence(pieces, expectation, hp, "test").iterations == 3);
    // Count=1 and count=0 with smoothing=.5 => probabilities .75 and .25.
    assert(std::abs(pieces[1].score - std::log(0.25)) < 1e-6);
    hp.em_fail_on_iteration_limit = true;
    expectation_calls = 0;
    requiresFailure([&] { runUnigramEMToConvergence(pieces, expectation, hp, "test"); });
    hp.em_convergence_tolerance = 0.02;
    expectation_calls = 0;
    assert(runUnigramEMToConvergence(pieces, expectation, hp, "test").iterations == 2);
    std::cout << "Tokenizer training policy host tests passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
}
