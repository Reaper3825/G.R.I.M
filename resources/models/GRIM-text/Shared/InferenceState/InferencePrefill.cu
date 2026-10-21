#include "InferencePrefill.hpp"
#include "../ConceptBlock/ConceptSpanDelimiterMap.hpp"

#include "../Forward/InferenceInput.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "../UnigramByte/UniByte.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace GRIM::Inference {

GRIM::Batching::BatchPayload buildInferencePrefill(
    const GRIM::Config::AiConfigSnapshot& config,
    GRIM::Tokenizer::UniByte& tokenizer,
    const std::string& prompt, const nlohmann::json* input_state) {
    const auto model_hp = GRIM::HyperParameters::modelHP(config);
    if (model_hp.atom_insertion_enabled)
        throw std::invalid_argument("Token prefill cannot be used with the atom-insertion model");
    const int vocab_size = model_hp.embedding_vocab_size;
    if (tokenizer.vocabSize() != vocab_size)
        throw std::runtime_error("Inference tokenizer vocabulary does not match compiled model");
    GRIM::SpanText::EncodedText encoded;
    std::shared_ptr<const GRIM::ConceptSpanDelimiterMap> delimiter_map;
    if (input_state) {
        const auto source = GRIM::Forward::inferenceInputSource(prompt, *input_state);
        auto definitions = GRIM::HyperParameters::dataLoadingHP(config).concept_spans;
        GRIM::resolveConceptSpanDelimiters(definitions, [&](const std::string& text) {
            const auto id = tokenizer.unigramLM().getPieceId(text);
            const auto* piece = tokenizer.unigramLM().getPiece(id);
            return piece && piece->text == text ? id : -1;
        });
        delimiter_map = std::make_shared<const GRIM::ConceptSpanDelimiterMap>(definitions);
        const auto rendered = GRIM::SpanText::renderInputWithSpans(source, delimiter_map->definitions());
        encoded = GRIM::SpanText::encodeSpanText(rendered,
            [&](const auto& content, const auto& boundaries, auto* counts) {
                return tokenizer.tokenizeWithMetadata(content, boundaries, counts);
            });
    } else {
        encoded.tokens = tokenizer.tokenizeWithMetadata(prompt);
    }
    auto& t = encoded.tokens;
    // Supplied input is context. Only a tokenizer-added terminal EOS is removed.
    if (!t.token_ids.empty() && t.token_ids.back() == GRIM::Tokenizer::EOS_TOKEN_ID) {
        t.token_ids.pop_back();
        t.token_numeric_values.pop_back();
        t.token_atom_mask.pop_back();
        t.token_atom_flags.pop_back();
        t.atom_entry_ids.pop_back();
        t.token_local_atom_indices.pop_back();
    }
    auto payload = GRIM::Batching::buildInferenceBatchPayload(
        t.token_ids, t.token_numeric_values, t.token_atom_mask, t.token_atom_flags,
        t.atom_table, t.atom_entry_ids, t.local_atom_table, t.token_local_atom_indices,
        vocab_size,
        static_cast<size_t>(model_hp.training_batch_size),
        static_cast<size_t>(model_hp.max_cached_seq_len),
        model_hp.local_atom_retrieval_enabled);
    if (encoded.spans && !encoded.spans->empty()) {
        GRIM::validateNamedConceptSpans(*encoded.spans, t.token_ids.size());
        payload.named_concept_spans = {std::move(encoded.spans)};
    }
    payload.concept_span_delimiters = std::move(delimiter_map);
    payload.validate("buildInferencePrefill");
    return payload;
}

} // namespace GRIM::Inference
