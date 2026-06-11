#include "GrmtCorpusIO.hpp"

#include "../UnigramByte/TokenLayout.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace GRIM::TokenizerArtifacts {
namespace {

std::string pathString(const fs::path& path) {
    const std::string s = path.string();
    if (s.empty()) {
        throw std::runtime_error("[GRMT] path is empty");
    }
    return s;
}

void ensureParentDirectory(const fs::path& path) {
    const fs::path parent = path.parent_path();
    if (parent.empty()) {
        return;
    }
    std::error_code ec;
    fs::create_directories(parent, ec);
    if (ec) {
        throw std::runtime_error("[GRMT] failed to create parent directory for " +
                                 path.string() + ": " + ec.message());
    }
}

void writeExact(std::ostream& output, const void* data, std::size_t bytes, const std::string& sink) {
    output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!output) {
        throw std::runtime_error("[GRMT] failed write to " + sink + " (bytes=" +
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
        throw std::runtime_error("[GRMT] failed read from " + source + " (bytes=" +
                                 std::to_string(bytes) + ")");
    }
}

template <typename T>
T readScalar(std::istream& input, const std::string& source) {
    T value{};
    readExact(input, &value, sizeof(T), source);
    return value;
}

void publishTempFileOrThrow(const fs::path& temp_path, const fs::path& final_path) {
#ifdef _WIN32
    const std::wstring temp_w = temp_path.wstring();
    const std::wstring final_w = final_path.wstring();
    if (!MoveFileExW(temp_w.c_str(), final_w.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD error = GetLastError();
        throw std::runtime_error("[GRMT] failed to publish temp file " + temp_path.string() +
                                 " -> " + final_path.string() + " (GetLastError=" +
                                 std::to_string(static_cast<unsigned long>(error)) + ")");
    }
#else
    std::error_code ec;
    fs::rename(temp_path, final_path, ec);
    if (ec) {
        throw std::runtime_error("[GRMT] failed to publish temp file " + temp_path.string() +
                                 " -> " + final_path.string() + ": " + ec.message());
    }
#endif
}

bool shouldWriteSequence(const GrmtSequence& sequence) {
    return !sequence.token_ids.empty() && sequence.hasAnyValidTarget();
}

void writeAtomTextForToken(std::ostream& output,
                           const GrmtSequence& sequence,
                           std::uint32_t token_index,
                           const std::string& sink) {
    std::string atom_text;
    const std::uint32_t entry_id = sequence.atom_entry_ids[token_index];
    if (entry_id != GRIM::Tokenizer::kAtomEntryNone) {
        if (!sequence.atom_table) {
            throw std::runtime_error("[GRMT] atom_entry_id present without AtomTable while writing " + sink);
        }
        const auto entry = sequence.atom_table->getAtom(entry_id);
        if (!entry) {
            throw std::runtime_error("[GRMT] atom_entry_id=" + std::to_string(entry_id) +
                                     " has no AtomEntry while writing " + sink);
        }
        atom_text = sequence.atom_table->atomToString(*entry);
    }

    if (atom_text.size() > std::numeric_limits<std::uint16_t>::max()) {
        throw std::runtime_error("[GRMT] atom text too long for uint16 length prefix while writing " + sink +
                                 " (bytes=" + std::to_string(atom_text.size()) + ")");
    }
    const std::uint16_t len = static_cast<std::uint16_t>(atom_text.size());
    writeScalar(output, len, sink);
    if (len > 0) {
        writeExact(output, atom_text.data(), len, sink);
    }
}

} // namespace

bool GrmtSequence::hasAnyValidTarget() const {
    return std::any_of(targets.begin(), targets.end(), [](int target) { return target >= 0; });
}

void GrmtSequence::validateForWrite(const std::string& source) const {
    const std::size_t n = token_ids.size();
    if (targets.size() != n) {
        throw std::runtime_error("[GRMT] " + source + ": targets.size()=" +
                                 std::to_string(targets.size()) + " != token_ids.size()=" +
                                 std::to_string(n));
    }
    if (token_numeric_values.size() != n) {
        throw std::runtime_error("[GRMT] " + source + ": token_numeric_values.size()=" +
                                 std::to_string(token_numeric_values.size()) +
                                 " != token_ids.size()=" + std::to_string(n));
    }
    if (token_atom_mask.size() != n) {
        throw std::runtime_error("[GRMT] " + source + ": token_atom_mask.size()=" +
                                 std::to_string(token_atom_mask.size()) +
                                 " != token_ids.size()=" + std::to_string(n));
    }
    if (token_atom_flags.size() != n) {
        throw std::runtime_error("[GRMT] " + source + ": token_atom_flags.size()=" +
                                 std::to_string(token_atom_flags.size()) +
                                 " != token_ids.size()=" + std::to_string(n));
    }
    if (atom_entry_ids.size() != n) {
        throw std::runtime_error("[GRMT] " + source + ": atom_entry_ids.size()=" +
                                 std::to_string(atom_entry_ids.size()) +
                                 " != token_ids.size()=" + std::to_string(n));
    }
    if (token_exec_slots.size() != n) {
        throw std::runtime_error("[GRMT] " + source + ": token_exec_slots.size()=" +
                                 std::to_string(token_exec_slots.size()) +
                                 " != token_ids.size()=" + std::to_string(n));
    }

    for (std::size_t i = 0; i < n; ++i) {
        const bool token_is_atom = token_ids[i] >= GRIM::Tokenizer::ATOM_TOKEN_OFFSET &&
                                   token_ids[i] < GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET;
        if (token_atom_mask[i] != 0 && !token_is_atom) {
            throw std::runtime_error("[GRMT] " + source + ": token_atom_mask is set at non-atom token index=" +
                                     std::to_string(i) + " token_id=" + std::to_string(token_ids[i]));
        }
        if (atom_entry_ids[i] != GRIM::Tokenizer::kAtomEntryNone && !token_is_atom) {
            throw std::runtime_error("[GRMT] " + source + ": atom_entry_id is set at non-atom token index=" +
                                     std::to_string(i) + " token_id=" + std::to_string(token_ids[i]));
        }
        if (token_is_atom && token_atom_mask[i] == 0) {
            throw std::runtime_error("[GRMT] " + source + ": atom token has token_atom_mask=0 at index=" +
                                     std::to_string(i) + " token_id=" + std::to_string(token_ids[i]));
        }
    }

    if (execution_active && teacher_steps.empty()) {
        throw std::runtime_error("[GRMT] " + source +
                                 ": execution_active sequence has no teacher_steps");
    }
}

GrmtCorpusWriter::GrmtCorpusWriter(
    const fs::path& final_path,
    std::uint32_t num_sequences,
    std::uint32_t vocab_size)
    : final_path_(final_path)
    , temp_path_(final_path)
    , expected_sequences_(num_sequences)
    , vocab_size_(vocab_size)
{
    pathString(final_path_);
    if (num_sequences == 0) {
        throw std::runtime_error("[GRMT] writer requires num_sequences > 0 for " + final_path_.string());
    }
    if (vocab_size == 0) {
        throw std::runtime_error("[GRMT] writer requires vocab_size > 0 for " + final_path_.string());
    }

    ensureParentDirectory(final_path_);
    temp_path_ += ".tmp";
    std::error_code ec;
    fs::remove(temp_path_, ec);
    if (ec) {
        throw std::runtime_error("[GRMT] failed to remove stale temp file " + temp_path_.string() +
                                 ": " + ec.message());
    }

    file_.open(temp_path_, std::ios::binary | std::ios::trunc);
    if (!file_.is_open()) {
        throw std::runtime_error("[GRMT] cannot open temp file for write: " + temp_path_.string());
    }

    const GRIM::GRMT::Header header = GRIM::GRMT::makeCurrentHeader(num_sequences, vocab_size);
    GRIM::GRMT::writeHeaderOrThrow(file_, header, temp_path_.string());
}

GrmtCorpusWriter::~GrmtCorpusWriter() noexcept {
    if (file_.is_open()) {
        file_.close();
    }
    if (!committed_) {
        std::error_code ec;
        fs::remove(temp_path_, ec);
    }
}

void GrmtCorpusWriter::writeSequence(const GrmtSequence& sequence) {
    if (!file_.is_open()) {
        throw std::runtime_error("[GRMT] writeSequence called after stream close: " + temp_path_.string());
    }
    if (written_sequences_ >= expected_sequences_) {
        throw std::runtime_error("[GRMT] writeSequence would exceed header num_sequences=" +
                                 std::to_string(expected_sequences_) + " for " + temp_path_.string());
    }

    const std::string sink = temp_path_.string() + "#seq" + std::to_string(written_sequences_);
    sequence.validateForWrite(sink);

    const std::uint32_t len = static_cast<std::uint32_t>(sequence.token_ids.size());
    writeScalar(file_, len, sink);
    writeExact(file_, sequence.token_ids.data(), static_cast<std::size_t>(len) * sizeof(int), sink);
    writeExact(file_, sequence.targets.data(), static_cast<std::size_t>(len) * sizeof(int), sink);
    writeExact(file_, sequence.token_numeric_values.data(), static_cast<std::size_t>(len) * sizeof(float), sink);
    writeExact(file_, sequence.token_atom_mask.data(), static_cast<std::size_t>(len) * sizeof(std::uint8_t), sink);
    writeExact(file_, sequence.token_atom_flags.data(), static_cast<std::size_t>(len) * sizeof(std::uint32_t), sink);

    for (std::uint32_t j = 0; j < len; ++j) {
        writeAtomTextForToken(file_, sequence, j, sink);
    }

    const std::uint8_t exec_active = sequence.execution_active ? 1 : 0;
    writeScalar(file_, exec_active, sink);
    writeExact(file_, sequence.token_exec_slots.data(), static_cast<std::size_t>(len) * sizeof(std::int32_t), sink);

    static_assert(sizeof(GRIM::Execution::CompiledBootstrapBinding) == 12,
        "CompiledBootstrapBinding must be 12 bytes for bulk GRMT serialization");
    const std::uint32_t cbb_count = static_cast<std::uint32_t>(sequence.compiled_bootstrap_bindings.size());
    writeScalar(file_, cbb_count, sink);
    if (cbb_count > 0) {
        writeExact(file_, sequence.compiled_bootstrap_bindings.data(),
                   static_cast<std::size_t>(cbb_count) * sizeof(GRIM::Execution::CompiledBootstrapBinding), sink);
    }

    static_assert(sizeof(GRIM::Execution::TeacherStep) == 20,
        "TeacherStep must be 20 bytes for bulk GRMT serialization");
    const std::uint32_t ts_count = static_cast<std::uint32_t>(sequence.teacher_steps.size());
    writeScalar(file_, ts_count, sink);
    if (ts_count > 0) {
        writeExact(file_, sequence.teacher_steps.data(),
                   static_cast<std::size_t>(ts_count) * sizeof(GRIM::Execution::TeacherStep), sink);
    }

    ++written_sequences_;
}

GrmtSaveReport GrmtCorpusWriter::commit(std::size_t dropped_empty_sequences,
                                        std::size_t dropped_targetless_sequences) {
    if (written_sequences_ != expected_sequences_) {
        throw std::runtime_error("[GRMT] commit refused for " + temp_path_.string() +
                                 ": written_sequences=" + std::to_string(written_sequences_) +
                                 " != expected_sequences=" + std::to_string(expected_sequences_));
    }
    if (!file_.is_open()) {
        throw std::runtime_error("[GRMT] commit called after stream close: " + temp_path_.string());
    }

    file_.flush();
    file_.close();
    if (!file_.good()) {
        throw std::runtime_error("[GRMT] stream failed on flush/close: " + temp_path_.string());
    }

    publishTempFileOrThrow(temp_path_, final_path_);
    committed_ = true;

    GrmtSaveReport report{};
    report.path = final_path_;
    report.written_sequences = written_sequences_;
    report.vocab_size = vocab_size_;
    report.dropped_empty_sequences = dropped_empty_sequences;
    report.dropped_targetless_sequences = dropped_targetless_sequences;
    return report;
}

GrmtCorpusReader::GrmtCorpusReader(const fs::path& path,
                                   int max_mantissa_digit_slots)
    : path_(path)
    , max_mantissa_digit_slots_(max_mantissa_digit_slots)
{
    pathString(path_);
    file_.open(path_, std::ios::binary);
    if (!file_.is_open()) {
        throw std::runtime_error("[GRMT] cannot open file for read: " + path_.string());
    }
    header_ = GRIM::GRMT::readHeaderOrThrow(file_, path_.string());
}

GrmtCorpusReader::~GrmtCorpusReader() {
    if (file_.is_open()) {
        file_.close();
    }
}

bool GrmtCorpusReader::readNext(GrmtSequence& out_sequence) {
    if (sequences_read_ >= header_.num_sequences) {
        return false;
    }

    const std::string source = path_.string() + "#seq" + std::to_string(sequences_read_);
    const std::uint32_t seq_len = readScalar<std::uint32_t>(file_, source);
    if (seq_len == 0) {
        throw std::runtime_error("[GRMT] sequence length is zero in " + source);
    }

    GrmtSequence seq;
    seq.token_ids.resize(seq_len);
    seq.targets.resize(seq_len);
    seq.token_numeric_values.resize(seq_len);
    seq.token_atom_mask.resize(seq_len);
    seq.token_atom_flags.resize(seq_len);

    readExact(file_, seq.token_ids.data(), static_cast<std::size_t>(seq_len) * sizeof(int), source);
    readExact(file_, seq.targets.data(), static_cast<std::size_t>(seq_len) * sizeof(int), source);
    readExact(file_, seq.token_numeric_values.data(), static_cast<std::size_t>(seq_len) * sizeof(float), source);
    readExact(file_, seq.token_atom_mask.data(), static_cast<std::size_t>(seq_len) * sizeof(std::uint8_t), source);
    readExact(file_, seq.token_atom_flags.data(), static_cast<std::size_t>(seq_len) * sizeof(std::uint32_t), source);

    std::vector<std::string> atom_text(seq_len);
    for (std::uint32_t j = 0; j < seq_len; ++j) {
        const std::uint16_t len = readScalar<std::uint16_t>(file_, source);
        if (len > 0) {
            atom_text[j].resize(len);
            readExact(file_, atom_text[j].data(), len, source);
        }
    }

    seq.atom_entry_ids.assign(seq_len, GRIM::Tokenizer::kAtomEntryNone);

    std::string reconstructed_atom_source;
    std::vector<GRIM::Tokenizer::Detector::RawTextDetection> atom_detections;
    std::vector<std::uint32_t> detection_token_indices;
    atom_detections.reserve(seq_len);
    detection_token_indices.reserve(seq_len);

    for (std::uint32_t j = 0; j < seq_len; ++j) {
        if (atom_text[j].empty()) {
            continue;
        }
        const int token_id = seq.token_ids[j];
        if (token_id < GRIM::Tokenizer::ATOM_TOKEN_OFFSET ||
            token_id >= GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET) {
            throw std::runtime_error("[GRMT] atom text exists for non-atom token in " + source +
                                     " index=" + std::to_string(j) +
                                     " token_id=" + std::to_string(token_id));
        }
        const auto type = GRIM::Tokenizer::tokenIdToAtomType(token_id);
        const size_t start = reconstructed_atom_source.size();
        reconstructed_atom_source += atom_text[j];
        const size_t end = reconstructed_atom_source.size();
        atom_detections.emplace_back(start, end, type, "GRMT");
        detection_token_indices.push_back(j);
    }

    seq.atom_table = GRIM::Tokenizer::createAtomTableFromRawTextDetectionsForTokenSideChannels(
        std::string_view(reconstructed_atom_source.data(), reconstructed_atom_source.size()),
        atom_detections,
        detection_token_indices,
        seq.token_ids,
        seq.token_numeric_values,
        seq.token_atom_mask,
        seq.token_atom_flags,
        seq.atom_entry_ids,
        max_mantissa_digit_slots_,
        source.c_str());

    const std::uint8_t exec_active = readScalar<std::uint8_t>(file_, source);
    seq.execution_active = (exec_active != 0);

    seq.token_exec_slots.resize(seq_len);
    readExact(file_, seq.token_exec_slots.data(), static_cast<std::size_t>(seq_len) * sizeof(std::int32_t), source);

    const std::uint32_t cbb_count = readScalar<std::uint32_t>(file_, source);
    static_assert(sizeof(GRIM::Execution::CompiledBootstrapBinding) == 12,
        "CompiledBootstrapBinding must be 12 bytes for bulk GRMT deserialization");
    if (cbb_count > 0) {
        seq.compiled_bootstrap_bindings.resize(cbb_count);
        readExact(file_, seq.compiled_bootstrap_bindings.data(),
                  static_cast<std::size_t>(cbb_count) * sizeof(GRIM::Execution::CompiledBootstrapBinding), source);
    }

    const std::uint32_t ts_count = readScalar<std::uint32_t>(file_, source);
    static_assert(sizeof(GRIM::Execution::TeacherStep) == 20,
        "TeacherStep must be 20 bytes for bulk GRMT deserialization");
    if (ts_count > 0) {
        seq.teacher_steps.resize(ts_count);
        readExact(file_, seq.teacher_steps.data(),
                  static_cast<std::size_t>(ts_count) * sizeof(GRIM::Execution::TeacherStep), source);
    }

    out_sequence = std::move(seq);
    ++sequences_read_;
    return true;
}

GrmtCorpus GrmtCorpusReader::readAll() {
    GrmtCorpus corpus;
    corpus.header = header_;
    corpus.sequences.reserve(header_.num_sequences);

    GrmtSequence sequence;
    while (readNext(sequence)) {
        corpus.sequences.push_back(std::move(sequence));
        sequence = GrmtSequence{};
    }

    if (corpus.sequences.size() != header_.num_sequences) {
        throw std::runtime_error("[GRMT] loaded sequence count mismatch in " + path_.string() +
                                 ": loaded=" + std::to_string(corpus.sequences.size()) +
                                 " header=" + std::to_string(header_.num_sequences));
    }
    return corpus;
}

GRIM::GRMT::Header loadGrmtHeader(const fs::path& path) {
    return GRIM::GRMT::readHeaderOrThrow(pathString(path));
}

GrmtSaveReport saveGrmtCorpus(
    const fs::path& path,
    const std::vector<GrmtSequence>& sequences,
    std::uint32_t vocab_size) {
    pathString(path);
    if (vocab_size == 0) {
        throw std::runtime_error("[GRMT] saveGrmtCorpus requires vocab_size > 0 for " + path.string());
    }

    std::uint32_t valid_sequences = 0;
    std::size_t dropped_empty = 0;
    std::size_t dropped_targetless = 0;
    for (const auto& sequence : sequences) {
        if (sequence.token_ids.empty()) {
            ++dropped_empty;
            continue;
        }
        if (!sequence.hasAnyValidTarget()) {
            ++dropped_targetless;
            continue;
        }
        ++valid_sequences;
    }

    if (valid_sequences == 0) {
        throw std::runtime_error("[GRMT] refusing to write zero valid sequences to " + path.string() +
                                 " (input=" + std::to_string(sequences.size()) +
                                 ", dropped_empty=" + std::to_string(dropped_empty) +
                                 ", dropped_targetless=" + std::to_string(dropped_targetless) + ")");
    }
    if (dropped_empty > 0 || dropped_targetless > 0) {
        throw std::runtime_error("[GRMT] refused writer-side drops for " + path.string() +
                                 " (dropped_empty=" + std::to_string(dropped_empty) +
                                 ", dropped_targetless=" + std::to_string(dropped_targetless) + ")");
    }

    GrmtCorpusWriter writer(path, valid_sequences, vocab_size);
    for (const auto& sequence : sequences) {
        if (!shouldWriteSequence(sequence)) {
            continue;
        }
        writer.writeSequence(sequence);
    }
    return writer.commit(dropped_empty, dropped_targetless);
}

GrmtCorpus loadGrmtCorpus(const fs::path& path,
                          int max_mantissa_digit_slots) {
    GrmtCorpusReader reader(path, max_mantissa_digit_slots);
    return reader.readAll();
}

} // namespace GRIM::TokenizerArtifacts
