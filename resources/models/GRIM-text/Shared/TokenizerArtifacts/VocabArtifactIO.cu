#include "VocabArtifactIO.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace GRIM::TokenizerArtifacts {
namespace {

fs::path requireBinaryVocabPath(fs::path path) {
    if (path.empty()) {
        throw std::runtime_error("[TokenizerVocabFile] vocab path is empty");
    }
    if (path.extension() != ".bin") {
        throw std::runtime_error("[TokenizerVocabFile] vocab path must be a .bin KTMG artifact, got: " +
                                 path.string());
    }
    return path;
}

void ensureParentDirectory(const fs::path& path) {
    const fs::path parent = path.parent_path();
    if (parent.empty()) {
        return;
    }
    std::error_code ec;
    fs::create_directories(parent, ec);
    if (ec) {
        throw std::runtime_error("[TokenizerVocabFile] failed to create parent directory for " +
                                 path.string() + ": " + ec.message());
    }
}

void writeExact(std::ostream& output, const void* data, std::size_t bytes, const std::string& sink) {
    output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!output) {
        throw std::runtime_error("[TokenizerVocabFile] failed write to " + sink + " (bytes=" +
                                 std::to_string(bytes) + ")");
    }
}

template <typename T>
void writeScalar(std::ostream& output, const T& value, const std::string& sink) {
    writeExact(output, &value, sizeof(T), sink);
}

void readExact(std::istream& input, void* data, std::size_t bytes, const std::string& source) {
    input.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!input) {
        throw std::runtime_error("[TokenizerVocabFile] failed read from " + source + " (bytes=" +
                                 std::to_string(bytes) + ")");
    }
}

template <typename T>
T readScalar(std::istream& input, const std::string& source) {
    T value{};
    readExact(input, &value, sizeof(T), source);
    return value;
}

std::string readTokenText(std::istream& input, std::uint32_t len, const std::string& source) {
    if (len > GRIM::Tokenizer::MAX_PIECE_LENGTH) {
        throw std::runtime_error("[TokenizerVocabFile] invalid piece length " + std::to_string(len) +
                                 " in " + source + " (max=" +
                                 std::to_string(GRIM::Tokenizer::MAX_PIECE_LENGTH) + ")");
    }
    std::string text(len, '\0');
    if (len > 0) {
        readExact(input, text.data(), len, source);
    }
    return text;
}

} // namespace

TokenizerVocabFile::TokenizerVocabFile(fs::path path)
    : path_(requireBinaryVocabPath(std::move(path))) {}

void TokenizerVocabFile::readInto(GRIM::Tokenizer::UnigramLM& unigram) const {
    const fs::path bin_path = requireBinaryVocabPath(path_);
    std::ifstream bin_file(bin_path, std::ios::binary);
    if (!bin_file.is_open()) {
        throw std::runtime_error("[TokenizerVocabFile] failed to open binary vocab file: " + bin_path.string());
    }

    const std::string source = bin_path.string();
    char magic[4]{};
    readExact(bin_file, magic, sizeof(magic), source);
    if (magic[0] != 'K' || magic[1] != 'T' || magic[2] != 'M' || magic[3] != 'G') {
        throw std::runtime_error("[TokenizerVocabFile] invalid KTMG magic in vocab file: " + source);
    }

    const std::uint16_t version = readScalar<std::uint16_t>(bin_file, source);
    if (version != 4) {
        throw std::runtime_error("[TokenizerVocabFile] vocab file version " + std::to_string(version) +
                                 " is unsupported; required version 4. Retrain tokenizer: " + source);
    }

    (void)readScalar<std::uint32_t>(bin_file, source); // checksum placeholder
    const std::uint32_t serialized_record_count = readScalar<std::uint32_t>(bin_file, source);
    (void)readScalar<std::uint32_t>(bin_file, source); // max_length

    char flags[3]{};
    readExact(bin_file, flags, sizeof(flags), source);

    const std::uint32_t saved_token_space_size = readScalar<std::uint32_t>(bin_file, source);

    GRIM::Tokenizer::UnigramLM loaded;
    loaded.setByteFallbackEnabled(unigram.byteFallbackEnabled());

    std::uint32_t special_records_seen = 0;
    for (std::uint32_t i = 0; i < serialized_record_count; ++i) {
        const std::uint32_t len = readScalar<std::uint32_t>(bin_file, source);
        const std::string text = readTokenText(bin_file, len, source);
        const float score = readScalar<float>(bin_file, source);
        const int token_id = readScalar<int>(bin_file, source);

        if (GRIM::Tokenizer::isSpecialTokenId(token_id)) {
            if (text != GRIM::Tokenizer::specialTokenText(token_id)) {
                throw std::runtime_error(
                    "[TokenizerVocabFile] special vocab record mismatch at record " + std::to_string(i) +
                    ": stored token_id=" + std::to_string(token_id) +
                    " text='" + text + "' expected='" + GRIM::Tokenizer::specialTokenText(token_id) + "'");
            }
            ++special_records_seen;
            continue;
        }

        const int expected_id = GRIM::Tokenizer::UnigramLM::tokenIdForIndex(loaded.pieceCount());
        if (token_id != expected_id) {
            throw std::runtime_error(
                "[TokenizerVocabFile] vocab.bin token_id mismatch at record " + std::to_string(i) +
                " ('" + text.substr(0, 30) + "'): stored=" + std::to_string(token_id) +
                " expected=" + std::to_string(expected_id) +
                ". Retrain tokenizer; do not patch the vocab header.");
        }

        loaded.addPiece(text, score, false);
    }

    if (special_records_seen != GRIM::Tokenizer::NUM_SPECIAL_TOKENS) {
        throw std::runtime_error("[TokenizerVocabFile] vocab file " + source +
                                 " has special metadata records=" + std::to_string(special_records_seen) +
                                 " expected=" + std::to_string(GRIM::Tokenizer::NUM_SPECIAL_TOKENS));
    }

    loaded.buildTrie();

    const std::uint32_t computed_token_space_size =
        static_cast<std::uint32_t>(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET + loaded.pieceCount());
    if (saved_token_space_size != computed_token_space_size) {
        throw std::runtime_error(
            "[TokenizerVocabFile] vocab.bin token-space size mismatch in " + source +
            ": header=" + std::to_string(saved_token_space_size) +
            " computed=" + std::to_string(computed_token_space_size) +
            ". Retrain tokenizer; do not patch the header.");
    }

    unigram = std::move(loaded);

    std::cout << "[TokenizerVocabFile] Loaded " << unigram.pieceCount()
              << " learned pieces from " << source << std::endl;
    std::cout << "[TokenizerVocabFile] Token-space size: " << saved_token_space_size
              << " (" << GRIM::Tokenizer::NUM_SPECIAL_TOKENS << " special + "
              << GRIM::Tokenizer::BYTE_VOCAB_SIZE << " bytes + "
              << GRIM::Tokenizer::ATOM_VOCAB_SIZE << " atom type placeholders + "
              << unigram.pieceCount() << " unigram pieces)" << std::endl;
}

void TokenizerVocabFile::writeFrom(const GRIM::Tokenizer::UnigramLM& unigram,
                                   const TokenizerVocabSaveOptions& options) const {
    const fs::path bin_path = requireBinaryVocabPath(path_);
    if (!std::isfinite(options.score_multiplier)) {
        throw std::runtime_error("[TokenizerVocabFile] score_multiplier is not finite for " + bin_path.string());
    }

    ensureParentDirectory(bin_path);

    std::ofstream bin_file(bin_path, std::ios::binary | std::ios::trunc);
    if (!bin_file.is_open()) {
        throw std::runtime_error("[TokenizerVocabFile] failed to create binary vocab file: " + bin_path.string());
    }

    const std::string sink = bin_path.string();
    const char magic[4] = {'K', 'T', 'M', 'G'};
    writeExact(bin_file, magic, sizeof(magic), sink);

    const std::uint16_t version = 4;
    writeScalar(bin_file, version, sink);

    const std::uint32_t checksum = 0;
    writeScalar(bin_file, checksum, sink);

    const std::uint32_t piece_count = static_cast<std::uint32_t>(unigram.pieceCount());
    const std::uint32_t serialized_record_count =
        static_cast<std::uint32_t>(GRIM::Tokenizer::NUM_SPECIAL_TOKENS) + piece_count;
    writeScalar(bin_file, serialized_record_count, sink);

    const std::uint32_t max_length = GRIM::Tokenizer::MAX_PIECE_LENGTH;
    writeScalar(bin_file, max_length, sink);

    const char flags[3] = {0, 0, 0};
    writeExact(bin_file, flags, sizeof(flags), sink);

    const std::uint32_t token_space_size =
        static_cast<std::uint32_t>(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET + unigram.pieceCount());
    writeScalar(bin_file, token_space_size, sink);

    if (options.score_multiplier != 1.0f) {
        std::cout << "[TokenizerVocabFile] Applying vocab_score_multiplier="
                  << options.score_multiplier << " while writing " << sink << std::endl;
    }

    auto write_record = [&](const std::string& text, float score, int token_id) {
        if (text.size() > GRIM::Tokenizer::MAX_PIECE_LENGTH) {
            throw std::runtime_error("[TokenizerVocabFile] token text too long while writing " + sink +
                                     " (bytes=" + std::to_string(text.size()) + ")");
        }
        const std::uint32_t len = static_cast<std::uint32_t>(text.size());
        writeScalar(bin_file, len, sink);
        if (len > 0) {
            writeExact(bin_file, text.data(), len, sink);
        }
        writeScalar(bin_file, score, sink);
        writeScalar(bin_file, token_id, sink);
    };

    for (const auto& def : GRIM::Tokenizer::SPECIAL_TOKEN_DEFINITIONS) {
        write_record(def.text, 0.0f, def.id);
    }

    for (std::uint32_t i = 0; i < piece_count; ++i) {
        const int token_id = GRIM::Tokenizer::UnigramLM::tokenIdForIndex(static_cast<int>(i));
        const auto* piece = unigram.getPiece(token_id);
        if (!piece) {
            throw std::runtime_error("[TokenizerVocabFile] missing piece for token_id=" +
                                     std::to_string(token_id) + " while writing " + sink);
        }
        write_record(piece->text, piece->score * options.score_multiplier, token_id);
    }

    bin_file.flush();
    bin_file.close();
    if (!bin_file.good()) {
        throw std::runtime_error("[TokenizerVocabFile] stream failed on flush/close: " + sink);
    }

    std::cout << "[TokenizerVocabFile] Wrote binary vocab (token-space size=" << token_space_size
              << " = " << GRIM::Tokenizer::NUM_SPECIAL_TOKENS << " special + "
              << GRIM::Tokenizer::BYTE_VOCAB_SIZE << " bytes + "
              << GRIM::Tokenizer::ATOM_VOCAB_SIZE << " atom type placeholders + "
              << piece_count << " unigram pieces) to " << sink << std::endl;

    if (options.export_text) {
        const fs::path text_path = bin_path.parent_path() / (bin_path.stem().string() + ".txt");
        std::ofstream text_file(text_path, std::ios::trunc);
        if (!text_file.is_open()) {
            throw std::runtime_error("[TokenizerVocabFile] failed to create text vocab export: " +
                                     text_path.string());
        }
        for (const auto& def : GRIM::Tokenizer::SPECIAL_TOKEN_DEFINITIONS) {
            text_file << def.text << "\t0\n";
        }
        for (std::uint32_t i = 0; i < piece_count; ++i) {
            const int token_id = GRIM::Tokenizer::UnigramLM::tokenIdForIndex(static_cast<int>(i));
            const auto* piece = unigram.getPiece(token_id);
            if (!piece) {
                throw std::runtime_error("[TokenizerVocabFile] missing piece for text export token_id=" +
                                         std::to_string(token_id));
            }
            text_file << piece->text << "\t" << (piece->score * options.score_multiplier) << "\n";
        }
        text_file.flush();
        text_file.close();
        if (!text_file.good()) {
            throw std::runtime_error("[TokenizerVocabFile] text export stream failed on close: " +
                                     text_path.string());
        }
        std::cout << "[TokenizerVocabFile] Wrote human-readable vocab export to "
                  << text_path.string() << std::endl;
    }
}

} // namespace GRIM::TokenizerArtifacts