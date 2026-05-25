#include "TokenizerArtifactBundle.hpp"

#include "VocabArtifactIO.hpp"
#include "../UnigramByte/UniByte.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace GRIM::TokenizerArtifacts {
namespace {

void ensurePathNotEmpty(const fs::path& path, const char* field) {
    if (path.empty()) {
        throw std::runtime_error(std::string("[TokenizerArtifactBundle] ") + field + " is empty");
    }
}

void requireTokenizerArtifactPaths(const GRIM::HyperParameters::TokenizerHP& hp) {
    ensurePathNotEmpty(hp.data_path, "data_path");
    ensurePathNotEmpty(hp.vocab_path, "vocab_path");
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

bool tokenizerArtifactBundleExists(const GRIM::HyperParameters::TokenizerHP& hp) {
    requireTokenizerArtifactPaths(hp);
    return fs::exists(hp.data_path) && fs::exists(hp.vocab_path);
}

TokenizerBundleManifest loadTokenizerArtifactBundle(
    const GRIM::HyperParameters::TokenizerHP& hp,
    GRIM::Tokenizer::UniByte& tokenizer) {
    requireTokenizerArtifactPaths(hp);
    const fs::path grmt_path(hp.data_path);
    const fs::path vocab_path(hp.vocab_path);
    if (!fs::exists(vocab_path)) {
        throw std::runtime_error("[TokenizerArtifactBundle] vocab file missing: " + vocab_path.string());
    }
    if (!fs::exists(grmt_path)) {
        throw std::runtime_error("[TokenizerArtifactBundle] GRMT file missing: " + grmt_path.string());
    }

    TokenizerVocabFile(vocab_path).readInto(tokenizer.unigramLM());

    TokenizerBundleManifest manifest{};
    manifest.grmt_header = loadGrmtHeader(grmt_path);
    manifest.tokenizer_vocab_size = static_cast<std::uint32_t>(tokenizer.vocabSize());
    validateVocabAgreement(manifest.grmt_header,
                           manifest.tokenizer_vocab_size,
                           grmt_path,
                           vocab_path);
    return manifest;
}

TokenizerBundleSaveReport saveTokenizerArtifactBundle(
    const GRIM::HyperParameters::TokenizerHP& hp,
    const GRIM::Tokenizer::UniByte& tokenizer,
    const std::vector<GrmtSequence>& sequences) {
    requireTokenizerArtifactPaths(hp);
    if (!std::isfinite(hp.vocab_score_multiplier)) {
        throw std::runtime_error("[TokenizerArtifactBundle] vocab_score_multiplier is not finite");
    }

    const fs::path grmt_path(hp.data_path);
    const fs::path vocab_path(hp.vocab_path);

    ensureParentDirectory(vocab_path, "vocab");
    ensureParentDirectory(grmt_path, "grmt");

    const std::uint32_t tokenizer_vocab_size = static_cast<std::uint32_t>(tokenizer.vocabSize());
    if (tokenizer_vocab_size == 0) {
        throw std::runtime_error("[TokenizerArtifactBundle] tokenizer vocab size is zero before save");
    }

    TokenizerVocabFile(vocab_path).writeFrom(tokenizer.unigramLM(), hp);

    TokenizerBundleSaveReport report{};
    report.grmt = saveGrmtCorpus(grmt_path, sequences, tokenizer_vocab_size);
    report.manifest.grmt_header = loadGrmtHeader(grmt_path);
    report.manifest.tokenizer_vocab_size = tokenizer_vocab_size;
    validateVocabAgreement(report.manifest.grmt_header,
                           report.manifest.tokenizer_vocab_size,
                           grmt_path,
                           vocab_path);
    return report;
}

} // namespace GRIM::TokenizerArtifacts
