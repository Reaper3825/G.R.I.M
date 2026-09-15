#include "ManualVocabSource.hpp"

#include "TextUtils.hpp"
#include "VocabWriteOp.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace GRIM::Tokenizer {

std::vector<UnigramPiece> loadManualVocabPieces(const std::string& path) {
    if (path.empty()) return {};
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("[UnigramLM] cannot open manual vocab source: " + path);
    }
    nlohmann::json document = nlohmann::json::parse(input);
    if (!document.is_object() || !document.contains("tokens") ||
        !document.at("tokens").is_object()) {
        throw std::runtime_error("[UnigramLM] manual vocab must contain a tokens object: " + path);
    }
    std::vector<UnigramPiece> pieces;
    const auto& tokens = document.at("tokens");
    pieces.reserve(tokens.size());
    for (auto it = tokens.begin(); it != tokens.end(); ++it) {
        const std::string text = it.key();
        if (!it.value().is_number()) {
            throw std::runtime_error("[UnigramLM] manual vocab score must be numeric for: " + text);
        }
        const float score = it.value().get<float>();
        UnigramPiece piece{text, score, true};
        validateUnigramVocabWritePiece(piece, "loadManualVocabPieces");
        if (score >= 0.0f) {
            throw std::runtime_error("[UnigramLM] manual score must be a negative log probability for: " + text);
        }
        if (text.find('\n') != std::string::npos ||
            normalizeSpaces(text, false) != text) {
            throw std::runtime_error("[UnigramLM] manual piece must use normalized tokenizer text: " + text);
        }
        for (std::size_t byte = 0; byte < text.size();) {
            std::uint32_t codepoint = 0;
            std::size_t length = 0;
            if (!utf8DecodeAt(text, byte, &codepoint, &length)) {
                throw std::runtime_error("[UnigramLM] manual piece is not valid UTF-8: " + text);
            }
            (void)codepoint;
            byte += length;
        }
        pieces.push_back(std::move(piece));
    }
    return pieces;
}

void appendManualVocabPieces(UnigramLM& unigram,
                             const std::vector<UnigramPiece>& pieces,
                             int target_vocab_size) {
    if (target_vocab_size <= 0 ||
        pieces.size() > static_cast<std::size_t>(target_vocab_size)) {
        throw std::runtime_error("[UnigramLM] manual entries exceed target learned vocabulary size");
    }
    for (const auto& piece : pieces) {
        applyUnigramVocabWriteOp(UnigramVocabWriteRequest{
            UnigramVocabWriteTarget{unigram.pieces_, unigram.piece_to_id_},
            piece,
            UnigramLM::tokenIdForIndex(static_cast<int>(unigram.pieces_.size())),
            UnigramVocabWriteMode::AppendOnly,
            "appendManualVocabPieces"});
    }
    std::cout << "[UnigramLM] Manual source supplied " << pieces.size()
              << " protected ordinary pieces" << std::endl;
}

} // namespace GRIM::Tokenizer
