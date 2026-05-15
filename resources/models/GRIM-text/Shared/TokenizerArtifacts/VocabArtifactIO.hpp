#pragma once

#include "../UnigramByte/Unigram.hpp"

#include <filesystem>

namespace GRIM::TokenizerArtifacts {

struct TokenizerVocabSaveOptions {
    bool export_text = false;
    float score_multiplier = 1.0f;
};

// Internal vocab-file primitive used by TokenizerArtifactBundle.
// Public tokenizer classes intentionally do not expose vocab-only load/save APIs;
// bundle loading validates vocab.bin against the paired GRMT header before use.
class TokenizerVocabFile {
public:
    explicit TokenizerVocabFile(std::filesystem::path path);

    const std::filesystem::path& path() const { return path_; }

    void readInto(GRIM::Tokenizer::UnigramLM& unigram) const;
    void writeFrom(const GRIM::Tokenizer::UnigramLM& unigram,
                   const TokenizerVocabSaveOptions& options = TokenizerVocabSaveOptions{}) const;

private:
    std::filesystem::path path_;
};

} // namespace GRIM::TokenizerArtifacts