#pragma once

#include "GrmtCorpusIO.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace GRIM::Tokenizer { class UniByte; }
namespace GRIM::HyperParameters { struct PathConfig; struct StartupConfig; }

namespace GRIM::TokenizerArtifacts {

struct TokenizerArtifactPaths {
    std::filesystem::path grmt_path;
    std::filesystem::path vocab_path;

    void validate() const;
    static TokenizerArtifactPaths fromPathConfig(const GRIM::HyperParameters::PathConfig& paths);
    static TokenizerArtifactPaths fromStartupConfig(const GRIM::HyperParameters::StartupConfig& config);
};

struct TokenizerBundleManifest {
    GRIM::GRMT::Header grmt_header{};
    std::uint32_t tokenizer_vocab_size = 0;
};

struct TokenizerBundleSaveOptions {
    bool save_text_vocab = false;
    float vocab_score_multiplier = 1.0f;
    GrmtSaveOptions grmt{};
};

struct TokenizerBundleSaveReport {
    TokenizerBundleManifest manifest{};
    GrmtSaveReport grmt{};
};

class TokenizerArtifactBundle {
public:
    explicit TokenizerArtifactBundle(TokenizerArtifactPaths paths);
    explicit TokenizerArtifactBundle(const GRIM::HyperParameters::PathConfig& paths);
    explicit TokenizerArtifactBundle(const GRIM::HyperParameters::StartupConfig& config);

    const TokenizerArtifactPaths& paths() const { return paths_; }
    bool exists() const;

    // Loads the tokenizer vocabulary and validates the GRMT header as one artifact pair.
    // It intentionally does not read all GRMT rows; use GrmtCorpusReader/loadGrmtCorpus for row data.
    TokenizerBundleManifest load(GRIM::Tokenizer::UniByte& tokenizer) const;

    // Saves vocabulary + GRMT rows through one boundary so the two artifacts share one vocab authority.
    TokenizerBundleSaveReport save(
        const GRIM::Tokenizer::UniByte& tokenizer,
        const std::vector<GrmtSequence>& sequences,
        const TokenizerBundleSaveOptions& options = TokenizerBundleSaveOptions{}) const;

private:
    TokenizerArtifactPaths paths_;
};

} // namespace GRIM::TokenizerArtifacts
