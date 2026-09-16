#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace GRIM::Tokenizer {

struct ExactPieceDefinition {
    std::string text;
    int token_id = -1;
};

struct ExactPieceSpan {
    std::size_t start = 0;
    std::size_t end = 0;
    int token_id = -1;
};

// Deterministic longest-match scanner for authored exact vocabulary pieces.
// Definitions beginning with the SentencePiece boundary marker require a
// lexical boundary after the match so, for example, ▁are cannot capture ▁area.
class ExactPieceMatcher final {
public:
    ExactPieceMatcher();

    void rebuild(const std::vector<ExactPieceDefinition>& definitions);

    std::vector<ExactPieceSpan> findMatches(
        std::string_view normalized_text,
        const std::vector<ExactPieceSpan>& excluded_spans = {}) const;

    bool empty() const { return definition_count_ == 0; }

private:
    struct Node final {
        std::array<int, 256> children{};
        int token_id = -1;
        bool requires_end_boundary = false;

        Node();
    };

    std::vector<Node> trie_;
    std::size_t definition_count_ = 0;
};

} // namespace GRIM::Tokenizer
