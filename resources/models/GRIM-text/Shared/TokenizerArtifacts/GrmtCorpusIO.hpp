#pragma once

#include "GrmtSequence.hpp"
#include "../GRMT/GrmtFormat.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace GRIM::TokenizerArtifacts {

struct GrmtCorpus {
    GRIM::GRMT::Header header{};
    std::vector<GrmtSequence> sequences;
};

struct GrmtSaveReport {
    std::filesystem::path path;
    std::uint32_t written_sequences = 0;
    std::uint32_t vocab_size = 0;
    std::size_t dropped_empty_sequences = 0;
    std::size_t dropped_targetless_sequences = 0;
};

class GrmtCorpusWriter {
public:
    GrmtCorpusWriter(
        const std::filesystem::path& final_path,
        std::uint32_t num_sequences,
        std::uint32_t vocab_size);
    ~GrmtCorpusWriter() noexcept;

    GrmtCorpusWriter(const GrmtCorpusWriter&) = delete;
    GrmtCorpusWriter& operator=(const GrmtCorpusWriter&) = delete;
    GrmtCorpusWriter(GrmtCorpusWriter&&) = delete;
    GrmtCorpusWriter& operator=(GrmtCorpusWriter&&) = delete;

    void writeSequence(const GrmtSequence& sequence);
    GrmtSaveReport commit(std::size_t dropped_empty_sequences,
                          std::size_t dropped_targetless_sequences);

private:
    std::filesystem::path final_path_;
    std::filesystem::path temp_path_;
    std::ofstream file_;
    std::uint32_t expected_sequences_ = 0;
    std::uint32_t written_sequences_ = 0;
    std::uint32_t vocab_size_ = 0;
    bool committed_ = false;
};

class GrmtCorpusReader {
public:
    explicit GrmtCorpusReader(const std::filesystem::path& path);
    ~GrmtCorpusReader();

    GrmtCorpusReader(const GrmtCorpusReader&) = delete;
    GrmtCorpusReader& operator=(const GrmtCorpusReader&) = delete;
    GrmtCorpusReader(GrmtCorpusReader&&) = delete;
    GrmtCorpusReader& operator=(GrmtCorpusReader&&) = delete;

    const GRIM::GRMT::Header& header() const { return header_; }
    std::uint32_t sequencesRead() const { return sequences_read_; }

    bool readNext(GrmtSequence& out_sequence);
    GrmtCorpus readAll();

private:
    std::filesystem::path path_;
    std::ifstream file_;
    GRIM::GRMT::Header header_{};
    std::uint32_t sequences_read_ = 0;
};

GRIM::GRMT::Header loadGrmtHeader(const std::filesystem::path& path);

GrmtSaveReport saveGrmtCorpus(
    const std::filesystem::path& path,
    const std::vector<GrmtSequence>& sequences,
    std::uint32_t vocab_size);

GrmtCorpus loadGrmtCorpus(const std::filesystem::path& path);

} // namespace GRIM::TokenizerArtifacts
