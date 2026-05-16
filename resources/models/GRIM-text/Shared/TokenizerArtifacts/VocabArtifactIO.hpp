#pragma once

#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "../UnigramByte/Unigram.hpp"

#include <filesystem>

namespace GRIM::TokenizerArtifacts {

// Internal vocab-file primitive used by TokenizerArtifactBundle functions.
// Public tokenizer classes intentionally do not expose vocab-only load/save APIs;
// bundle loading validates vocab.bin against the paired GRMT header before use.
class TokenizerVocabFile {
public:
    explicit TokenizerVocabFile(std::filesystem::path path);

    const std::filesystem::path& path() const { return path_; }

    void readInto(GRIM::Tokenizer::UnigramLM& unigram) const;
    void writeFrom(const GRIM::Tokenizer::UnigramLM& unigram,
                   const GRIM::HyperParameters::TokenizerHP& tokenizer_hp) const;

private:
    std::filesystem::path path_;
};

} // namespace GRIM::TokenizerArtifacts