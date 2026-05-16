#pragma once

#include "GrmtCorpusIO.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"

#include <cstdint>
#include <vector>

namespace GRIM::Tokenizer { class UniByte; }

namespace GRIM::TokenizerArtifacts {

struct TokenizerBundleManifest {
    GRIM::GRMT::Header grmt_header{};
    std::uint32_t tokenizer_vocab_size = 0;
};

struct TokenizerBundleSaveReport {
    TokenizerBundleManifest manifest{};
    GrmtSaveReport grmt{};
};

bool tokenizerArtifactBundleExists(const GRIM::HyperParameters::TokenizerHP& hp);

// Loads the tokenizer vocabulary and validates the GRMT header as one artifact pair.
// It intentionally does not read all GRMT rows; use GrmtCorpusReader/loadGrmtCorpus for row data.
TokenizerBundleManifest loadTokenizerArtifactBundle(
    const GRIM::HyperParameters::TokenizerHP& hp,
    GRIM::Tokenizer::UniByte& tokenizer);

// Saves vocabulary + GRMT rows through one boundary so the two artifacts share one vocab authority.
TokenizerBundleSaveReport saveTokenizerArtifactBundle(
    const GRIM::HyperParameters::TokenizerHP& hp,
    const GRIM::Tokenizer::UniByte& tokenizer,
    const std::vector<GrmtSequence>& sequences);

} // namespace GRIM::TokenizerArtifacts
