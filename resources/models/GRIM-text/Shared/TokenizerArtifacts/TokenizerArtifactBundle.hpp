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

// Loads only the shared vocabulary. This is the intentional multi-GRMT path:
// callers may then encode a new corpus without requiring an existing GRMT and
// without granting permission to rewrite vocab.bin.
std::uint32_t loadSharedTokenizerVocabulary(
    const GRIM::HyperParameters::TokenizerHP& hp,
    GRIM::Tokenizer::UniByte& tokenizer);

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

// Writes only hp.data_path using the already-loaded shared vocabulary. The
// vocabulary artifact is opened read-only for validation and is never replaced.
TokenizerBundleSaveReport saveGrmtForSharedTokenizerVocabulary(
    const GRIM::HyperParameters::TokenizerHP& hp,
    const GRIM::Tokenizer::UniByte& tokenizer,
    const std::vector<GrmtSequence>& sequences);

} // namespace GRIM::TokenizerArtifacts
