#include "TokenizerArtifactBundle.hpp"

#include "VocabArtifactIO.hpp"
#include "../UnigramByte/UniByte.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace fs = std::filesystem;

namespace GRIM::TokenizerArtifacts {
namespace {

void ensurePathNotEmpty(const fs::path& path, const char* field) {
    if (path.empty()) {
        throw std::runtime_error(std::string("[TokenizerArtifactBundle] ") + field + " is empty");
    }
}

void ensureParentDirectory(const fs::path& path, const char* field) {
    const fs::path parent = path.parent_path();
    if (parent.empty()) {
        return;
    }
    std::error_code ec;
    fs::create_directories(parent, ec);
    if (ec) {
        throw std::runtime_error(std::string("[TokenizerArtifactBundle] failed to create parent directory for ") +
                                 field + " path " + path.string() + ": " + ec.message());
    }
}

void validateVocabAgreement(const GRIM::GRMT::Header& header,
                            std::uint32_t tokenizer_vocab_size,
                            const fs::path& grmt_path,
                            const fs::path& vocab_path) {
    if (header.vocab_size != tokenizer_vocab_size) {
        throw std::runtime_error(
            "[TokenizerArtifactBundle] vocab mismatch: GRMT header in " + grmt_path.string() +
            " reports " + std::to_string(header.vocab_size) +
            " but tokenizer loaded from " + vocab_path.string() +
            " reports " + std::to_string(tokenizer_vocab_size) +
            ". Delete both artifacts and regenerate them together.");
    }
}

} // namespace

void TokenizerArtifactPaths::validate() const {
    ensurePathNotEmpty(grmt_path, "grmt_path");
    ensurePathNotEmpty(vocab_path, "vocab_path");
}

TokenizerArtifactPaths TokenizerArtifactPaths::fromPathConfig(
    const GRIM::HyperParameters::PathConfig& paths) {
    TokenizerArtifactPaths artifact_paths{paths.data_path, paths.vocab_path};
    artifact_paths.validate();
    return artifact_paths;
}

TokenizerArtifactPaths TokenizerArtifactPaths::fromStartupConfig(
    const GRIM::HyperParameters::StartupConfig& config) {
    return fromPathConfig(config.paths);
}

TokenizerArtifactBundle::TokenizerArtifactBundle(TokenizerArtifactPaths paths)
    : paths_(std::move(paths)) {
    paths_.validate();
}

TokenizerArtifactBundle::TokenizerArtifactBundle(const GRIM::HyperParameters::PathConfig& paths)
    : TokenizerArtifactBundle(TokenizerArtifactPaths::fromPathConfig(paths)) {}

TokenizerArtifactBundle::TokenizerArtifactBundle(const GRIM::HyperParameters::StartupConfig& config)
    : TokenizerArtifactBundle(TokenizerArtifactPaths::fromStartupConfig(config)) {}

bool TokenizerArtifactBundle::exists() const {
    return fs::exists(paths_.grmt_path) && fs::exists(paths_.vocab_path);
}

TokenizerBundleManifest TokenizerArtifactBundle::load(GRIM::Tokenizer::UniByte& tokenizer) const {
    paths_.validate();
    if (!fs::exists(paths_.vocab_path)) {
        throw std::runtime_error("[TokenizerArtifactBundle] vocab file missing: " + paths_.vocab_path.string());
    }
    if (!fs::exists(paths_.grmt_path)) {
        throw std::runtime_error("[TokenizerArtifactBundle] GRMT file missing: " + paths_.grmt_path.string());
    }

    TokenizerVocabFile(paths_.vocab_path).readInto(tokenizer.unigramLM());

    TokenizerBundleManifest manifest{};
    manifest.grmt_header = loadGrmtHeader(paths_.grmt_path);
    manifest.tokenizer_vocab_size = static_cast<std::uint32_t>(tokenizer.vocabSize());
    validateVocabAgreement(manifest.grmt_header,
                           manifest.tokenizer_vocab_size,
                           paths_.grmt_path,
                           paths_.vocab_path);
    return manifest;
}

TokenizerBundleSaveReport TokenizerArtifactBundle::save(
    const GRIM::Tokenizer::UniByte& tokenizer,
    const std::vector<GrmtSequence>& sequences,
    const TokenizerBundleSaveOptions& options) const {
    paths_.validate();
    if (!std::isfinite(options.vocab_score_multiplier)) {
        throw std::runtime_error("[TokenizerArtifactBundle] vocab_score_multiplier is not finite");
    }

    ensureParentDirectory(paths_.vocab_path, "vocab");
    ensureParentDirectory(paths_.grmt_path, "grmt");

    const std::uint32_t tokenizer_vocab_size = static_cast<std::uint32_t>(tokenizer.vocabSize());
    if (tokenizer_vocab_size == 0) {
        throw std::runtime_error("[TokenizerArtifactBundle] tokenizer vocab size is zero before save");
    }

    TokenizerVocabSaveOptions vocab_options{};
    vocab_options.export_text = options.save_text_vocab;
    vocab_options.score_multiplier = options.vocab_score_multiplier;
    TokenizerVocabFile(paths_.vocab_path).writeFrom(tokenizer.unigramLM(), vocab_options);

    TokenizerBundleSaveReport report{};
    report.grmt = saveGrmtCorpus(paths_.grmt_path, sequences, tokenizer_vocab_size, options.grmt);
    report.manifest.grmt_header = loadGrmtHeader(paths_.grmt_path);
    report.manifest.tokenizer_vocab_size = tokenizer_vocab_size;
    validateVocabAgreement(report.manifest.grmt_header,
                           report.manifest.tokenizer_vocab_size,
                           paths_.grmt_path,
                           paths_.vocab_path);
    return report;
}

} // namespace GRIM::TokenizerArtifacts
