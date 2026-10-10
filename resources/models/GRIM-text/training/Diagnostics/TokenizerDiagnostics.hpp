#pragma once

// Diagnostics use the worker-owned tokenizer; the HTTP bridge owns no artifacts.
#include "../../Shared/UnigramByte/UniByte.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace GRIMText::Diagnostics {

inline nlohmann::json tokenizerVocabulary(const GRIM::Tokenizer::UniByte& tokenizer) {
    using namespace GRIM::Tokenizer;
    const auto layout = tokenLayoutFromActualVocabOrThrow(tokenizer.vocabSize(), "TokenizerDiagnostics");
    return {{"total_vocab_size", layout.total_vocab()}, {"unigram_vocab_size", layout.num_unigram},
            {"byte_vocab_size", layout.num_bytes}, {"atom_vocab_size", layout.num_atoms},
            {"numeric_vocab_size", layout.num_numeric}, {"newline_vocab_size", layout.num_newlines},
            {"special_token_count", layout.num_special}, {"pad_id", PAD_TOKEN_ID},
            {"unk_id", UNK_TOKEN_ID}, {"bos_id", BOS_TOKEN_ID}, {"eos_id", EOS_TOKEN_ID}};
}

inline void validateTokenizerEncoding(const GRIM::Tokenizer::UniByteResult& encoded, int vocab_size) {
    encoded.validate("TokenizerDiagnostics");
    for (int id : encoded.token_ids) {
        if (id < 0 || id >= vocab_size) throw std::runtime_error("Encoded token ID is outside vocabulary");
    }
}

inline nlohmann::json encodeTokenizerText(const GRIM::Tokenizer::UniByte& tokenizer, const std::string& text) {
    using namespace GRIM::Tokenizer;
    if (text.empty() || text.size() > 65536)
        throw std::invalid_argument("Enter between 1 and 65536 bytes of text");
    const auto start = std::chrono::steady_clock::now();
    const auto encoded = tokenizer.tokenizeWithMetadata(text);
    const auto encode_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    validateTokenizerEncoding(encoded, tokenizer.vocabSize());
    auto response = tokenizerVocabulary(tokenizer);
    response["status"] = "success";
    response["input_text"] = text;
    DecodeRequest decode_request(encoded);
    decode_request.lenient_invalid_utf8 = false;
    response["decoded_text"] = tokenizer.decode(decode_request);
    response["token_count"] = encoded.token_ids.size();
    response["encode_time_ms"] = encode_ms;
    response["tokens"] = nlohmann::json::array();
    for (int id : encoded.token_ids) {
        const char* type = isSpecialTokenId(id) ? "special" : isByteTokenId(id) ? "byte" :
            isNumericTokenId(id) ? "numeric" : isAtomTokenId(id) ? "atom" :
            isNewlineTokenId(id) ? "newline" : "unigram";
        response["tokens"].push_back({{"id", id}, {"type", type},
            {"piece", tokenizer.decode(DecodeRequest({id}))}});
    }
    return response;
}

// Smoke checks for the loaded artifact: metadata alignment, ID ranges, and
// decoding across representative text. This is not the offline self-test suite.
inline nlohmann::json validateLoadedTokenizer(const GRIM::Tokenizer::UniByte& tokenizer) {
    const auto start = std::chrono::steady_clock::now();
    auto response = tokenizerVocabulary(tokenizer);
    auto failures = nlohmann::json::array();
    int passed = 0;
    const std::string samples[] = {"", "Hello world!", "  spaces\tand\nlines\r\n",
        "42 -3.14 1e6", "https://example.com user@example.com", "caf\xC3\xA9 \xE4\xB8\x96\xE7\x95\x8C"};
    for (const auto& text : samples) {
        try {
            const auto encoded = tokenizer.tokenizeWithMetadata(text);
            validateTokenizerEncoding(encoded, tokenizer.vocabSize());
            GRIM::Tokenizer::DecodeRequest decode_request(encoded);
            decode_request.lenient_invalid_utf8 = false;
            (void)tokenizer.decode(decode_request);
            ++passed;
        } catch (const std::exception& e) { failures.push_back(e.what()); }
    }
    response["status"] = failures.empty() ? "success" : "error";
    response["phase"] = "loaded_tokenizer_smoke_checks";
    response["validation_tests_passed"] = passed;
    response["validation_tests_total"] = sizeof(samples) / sizeof(samples[0]);
    response["validation_time_ms"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    response["error"] = failures.empty() ? "" : "Loaded tokenizer smoke checks failed";
    response["failures"] = std::move(failures);
    return response;
}
} // namespace GRIMText::Diagnostics
