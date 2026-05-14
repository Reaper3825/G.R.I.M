//======================================================//
//  grmt_vocab_metrics_test.cu
//  Streaming corpus + vocabulary diagnostics for .grmt
//
//  Single scan → full metrics dump. No test framework.
//  Prints numbers you can actually use.
//
//  Build:
//    cmake --build build --config Release \
//          --target grmt_vocab_metrics_test
//  Run:
//    ./grmt_vocab_metrics_test \
//        --vocab  <path-to-vocab.bin> \
//        --grmt   <path-to-training_data.grmt>
//======================================================//

#include "grmt_vocab_metrics_test.hpp"
#include "../Shared/UnigramByte/Byte.hpp"
#include "../Shared/UnigramByte/Unigram.hpp"
#include "../Shared/UnigramByte/UniByte.hpp"
#include "../Shared/GRMT/GrmtFormat.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

using namespace GRIM::Tokenizer;
using namespace GRIM::Test;

namespace fs = std::filesystem;

// ════════════════════════════════════════════════════════
//  Vocab loader — builds id-to-text map from vocab.bin
// ════════════════════════════════════════════════════════

std::unordered_map<int, std::string> GRIM::Test::loadVocabMap(const std::string& path) {
    std::unordered_map<int, std::string> id_to_text;

    // Special tokens
    id_to_text[0] = "<unk>";
    id_to_text[1] = "<pad>";
    id_to_text[2] = "<s>";
    id_to_text[3] = "</s>";

    // Byte fallback tokens [4..259]
    for (int b = 0; b < BYTE_VOCAB_SIZE; ++b) {
        char c = static_cast<char>(b);
        id_to_text[BYTE_TOKEN_OFFSET + b] = std::string(1, c);
    }

    // Atom placeholder tokens
    for (int a = 0; a < ATOM_VOCAB_SIZE; ++a) {
        id_to_text[ATOM_TOKEN_OFFSET + a] = "<ATOM" + std::to_string(a) + ">";
    }

    // Read unigram pieces from KTMG binary
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("loadVocabMap: cannot open " + path);
    }

    char magic[4];
    f.read(magic, 4);
    if (std::strncmp(magic, "KTMG", 4) != 0) {
        throw std::runtime_error("loadVocabMap: bad magic in " + path);
    }

    uint16_t version;
    f.read(reinterpret_cast<char*>(&version), 2);
    if (version != 3 && version != 4) {
        throw std::runtime_error("loadVocabMap: unsupported vocab version " + std::to_string(version));
    }

    uint32_t checksum, serialized_record_count, max_length;
    f.read(reinterpret_cast<char*>(&checksum), 4);
    f.read(reinterpret_cast<char*>(&serialized_record_count), 4);
    f.read(reinterpret_cast<char*>(&max_length), 4);

    char flags[3];
    f.read(flags, 3);

    uint32_t token_space_size;
    f.read(reinterpret_cast<char*>(&token_space_size), 4);
    (void)token_space_size;

    std::vector<char> buf;
    buf.reserve(256);

    for (uint32_t i = 0; i < serialized_record_count; ++i) {
        uint32_t len;
        f.read(reinterpret_cast<char*>(&len), 4);
        buf.resize(len);
        f.read(buf.data(), len);
        std::string text(buf.data(), len);

        float score;
        f.read(reinterpret_cast<char*>(&score), 4);
        int token_id;
        f.read(reinterpret_cast<char*>(&token_id), 4);

        // Skip specials that were duplicated in the binary
        if (text == "<unk>" || text == "<pad>" || text == "<s>" || text == "</s>")
            continue;

        id_to_text[token_id] = text;
    }

    if (!f) {
        throw std::runtime_error("loadVocabMap: read error in " + path);
    }

    return id_to_text;
}

// ════════════════════════════════════════════════════════
//  Streaming GRMT scanner — one-pass metrics
// ════════════════════════════════════════════════════════

GRMTCorpusMetrics GRIM::Test::scanGRMT(
    const std::string& grmt_path,
    const std::unordered_map<int, std::string>& vocab)
{
    GRMTCorpusMetrics m;

    std::ifstream f(grmt_path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("scanGRMT: cannot open " + grmt_path);
    }

    // ── Header (16 bytes) ──
    const GRIM::GRMT::Header header = GRIM::GRMT::readHeaderOrThrow(f, grmt_path);
    m.grmt_version = header.version;
    m.num_sequences = header.num_sequences;
    m.vocab_size = header.vocab_size;

    // Token ID histogram (sparse — only observed IDs)
    m.token_hist.reserve(m.vocab_size);

    uint64_t total_utf8_bytes = 0;
    uint64_t total_words      = 0;
    m.seq_len_min = UINT32_MAX;
    m.seq_len_max = 0;
    uint64_t seq_len_sum = 0;
    double   seq_len_sum_sq = 0.0;   // for stddev

    // Reusable per-sequence buffers
    std::vector<int32_t> token_ids;
    std::string decoded;

    uint32_t s = 0;
    for (; s < m.num_sequences; ++s) {
        uint32_t seq_len;
        f.read(reinterpret_cast<char*>(&seq_len), 4);
        if (!f || seq_len == 0 || seq_len > 1'000'000) break;

        // ── Sequence length stats ──
        m.seq_len_min = std::min(m.seq_len_min, seq_len);
        m.seq_len_max = std::max(m.seq_len_max, seq_len);
        seq_len_sum += seq_len;
        seq_len_sum_sq += static_cast<double>(seq_len) * seq_len;

        // ── Read token_ids ──
        token_ids.resize(seq_len);
        f.read(reinterpret_cast<char*>(token_ids.data()), seq_len * sizeof(int32_t));

        // ── Histogram + token class breakdown ──
        for (uint32_t t = 0; t < seq_len; ++t) {
            uint32_t tid = static_cast<uint32_t>(token_ids[t]);
            ++m.token_hist[tid];

            // Classify token
            if (tid == UNK_TOKEN_ID) {
                ++m.unk_count;
            } else if (tid == PAD_TOKEN_ID || tid == BOS_TOKEN_ID || tid == EOS_TOKEN_ID) {
                ++m.special_count;
            } else if (tid >= BYTE_TOKEN_OFFSET && tid < BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE) {
                ++m.byte_fallback_count;
            } else if (tid >= ATOM_TOKEN_OFFSET && tid < static_cast<uint32_t>(ATOM_TOKEN_OFFSET + ATOM_VOCAB_SIZE)) {
                ++m.atom_count;
            } else {
                ++m.unigram_count;
            }
        }
        m.total_tokens += seq_len;

        // ── Decode to text for bytes-per-token & fertility ──
        decoded.clear();
        decoded.reserve(seq_len * 4);
        for (uint32_t t = 0; t < seq_len; ++t) {
            int tid = token_ids[t];
            auto it = vocab.find(tid);
            if (it != vocab.end()) {
                decoded += it->second;
            } else if (tid >= BYTE_TOKEN_OFFSET && tid < ATOM_TOKEN_OFFSET) {
                decoded += static_cast<char>(tid - BYTE_TOKEN_OFFSET);
            }
            // atom / unknown → skip (no text contribution)
        }
        // Replace unigram space marker ▁ with actual space
        {
            const std::string marker = "\xE2\x96\x81"; // UTF-8 for ▁ (U+2581)
            std::string::size_type pos = 0;
            while ((pos = decoded.find(marker, pos)) != std::string::npos) {
                decoded.replace(pos, marker.size(), " ");
                pos += 1;
            }
        }

        total_utf8_bytes += decoded.size();

        // Count whitespace-delimited words
        bool in_word = false;
        for (char c : decoded) {
            bool ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
            if (!ws && !in_word) ++total_words;
            in_word = !ws;
        }

        // ── Skip remaining per-sequence fields (GRMT v11 layout) ──
        // targets (int32 × seq_len)
        f.seekg(seq_len * sizeof(int32_t), std::ios::cur);
        // numeric_values (float × seq_len)
        f.seekg(seq_len * sizeof(float), std::ios::cur);
        // atom_mask (uint8 × seq_len)
        f.seekg(seq_len * sizeof(uint8_t), std::ios::cur);
        // atom_flags (uint32 × seq_len)
        f.seekg(seq_len * sizeof(uint32_t), std::ios::cur);
        // Per-token atom text strings: each is uint16 length + text
        for (uint32_t j = 0; j < seq_len; ++j) {
            uint16_t slen = 0;
            f.read(reinterpret_cast<char*>(&slen), sizeof(uint16_t));
            if (slen > 0) f.seekg(slen, std::ios::cur);
        }
        // exec_active (uint8)
        f.seekg(sizeof(uint8_t), std::ios::cur);
        // token_exec_slots (int32 × seq_len)
        f.seekg(seq_len * sizeof(int32_t), std::ios::cur);
        // compiled_bootstrap_bindings: count(uint32) + entries(12 each)
        {
            uint32_t cbb_count = 0;
            f.read(reinterpret_cast<char*>(&cbb_count), sizeof(uint32_t));
            if (cbb_count > 0) f.seekg(cbb_count * 12, std::ios::cur);
        }
        // teacher_steps: count(uint32) + entries(20 each)
        {
            uint32_t ts_count = 0;
            f.read(reinterpret_cast<char*>(&ts_count), sizeof(uint32_t));
            if (ts_count > 0) f.seekg(ts_count * 20, std::ios::cur);
        }
        // slot_selection_targets: count(uint32) + entries(5 each: uint8 + int32)
        {
            uint32_t sst_count = 0;
            f.read(reinterpret_cast<char*>(&sst_count), sizeof(uint32_t));
            if (sst_count > 0) f.seekg(sst_count * 5, std::ios::cur);
        }

        if (!f) {
            std::cerr << "[grmt-metrics] Read error at sequence " << s << "\n";
            break;
        }

        if ((s + 1) % 5000 == 0) {
            std::cout << "[grmt-metrics] scanned " << (s + 1) << "/" << m.num_sequences
                      << " sequences (" << m.total_tokens << " tokens)\n";
            std::cout.flush();
        }
    }

    // ── Derived metrics ──
    m.sequences_scanned = s;
    m.scan_ok = f.good() || f.eof();
    m.distinct_ids = static_cast<uint32_t>(m.token_hist.size());
    m.seq_len_mean = m.num_sequences > 0
        ? static_cast<double>(seq_len_sum) / m.num_sequences
        : 0.0;
    if (m.num_sequences > 1) {
        double var = (seq_len_sum_sq / m.num_sequences) - (m.seq_len_mean * m.seq_len_mean);
        m.seq_len_stddev = var > 0.0 ? std::sqrt(var) : 0.0;
    }

    // Shannon entropy H(token) in bits
    if (m.total_tokens > 0) {
        double h = 0.0;
        double N = static_cast<double>(m.total_tokens);
        for (auto& [id, count] : m.token_hist) {
            double p = static_cast<double>(count) / N;
            if (p > 0.0) h -= p * std::log2(p);
        }
        m.shannon_entropy = h;
    }

    // Bytes per token
    m.bytes_per_token = m.total_tokens > 0
        ? static_cast<double>(total_utf8_bytes) / static_cast<double>(m.total_tokens)
        : 0.0;

    // Fertility (tokens per word)
    m.fertility = total_words > 0
        ? static_cast<double>(m.total_tokens) / static_cast<double>(total_words)
        : 0.0;

    // Vocab utilization
    m.vocab_utilization = m.vocab_size > 0
        ? 100.0 * m.distinct_ids / m.vocab_size
        : 0.0;
    m.dead_vocab_ids = (m.vocab_size > m.distinct_ids)
        ? m.vocab_size - m.distinct_ids : 0;

    if (m.seq_len_min == UINT32_MAX) m.seq_len_min = 0;

    return m;
}

// ════════════════════════════════════════════════════════
//  Metrics report — actionable numbers
// ════════════════════════════════════════════════════════

void GRIM::Test::printMetricsReport(
    const GRMTCorpusMetrics& m,
    const std::unordered_map<int, std::string>& vocab,
    const std::string& vocab_path,
    const std::string& grmt_path)
{
    auto pct = [&](uint64_t n) -> double {
        return m.total_tokens > 0 ? 100.0 * n / m.total_tokens : 0.0;
    };

    std::cout << std::fixed;

    std::cout << "\n"
        << "╔══════════════════════════════════════════════════════════════╗\n"
        << "║              GRMT Corpus + Vocabulary Metrics              ║\n"
        << "╠══════════════════════════════════════════════════════════════╣\n"
        << "║  FILES                                                     ║\n"
        << "║    vocab.bin:        " << vocab_path << "\n"
        << "║    training_data:    " << grmt_path << "\n"
        << "║    GRMT version:     " << m.grmt_version << "\n"
        << "╠══════════════════════════════════════════════════════════════╣\n"
        << "║  CORPUS SIZE                                               ║\n"
        << "║    sequences:        " << m.num_sequences << "\n"
        << "║    total tokens:     " << m.total_tokens << "\n"
        << "║    scanned OK:       " << m.sequences_scanned << "/" << m.num_sequences
        << (m.scan_ok ? "" : "  *** READ ERROR ***") << "\n"
        << "╠══════════════════════════════════════════════════════════════╣\n"
        << "║  SEQUENCE LENGTHS                                          ║\n"
        << "║    min:              " << m.seq_len_min << "\n"
        << "║    max:              " << m.seq_len_max << "\n"
        << "║    mean:             " << std::setprecision(1) << m.seq_len_mean << "\n"
        << "║    stddev:           " << std::setprecision(1) << m.seq_len_stddev << "\n"
        << "╠══════════════════════════════════════════════════════════════╣\n"
        << "║  VOCABULARY UTILIZATION                                    ║\n"
        << "║    vocab_size (hdr): " << m.vocab_size << "\n"
        << "║    distinct IDs:     " << m.distinct_ids << "\n"
        << "║    utilization:      " << std::setprecision(2) << m.vocab_utilization << "%\n"
        << "║    dead IDs:         " << m.dead_vocab_ids
        << " (never appear in corpus)\n"
        << "╠══════════════════════════════════════════════════════════════╣\n"
        << "║  TOKEN CLASS BREAKDOWN      count          % of total      ║\n"
        << "║    unigram pieces:   " << std::setw(12) << m.unigram_count
        << "    " << std::setprecision(2) << std::setw(7) << pct(m.unigram_count) << "%\n"
        << "║    byte fallback:    " << std::setw(12) << m.byte_fallback_count
        << "    " << std::setprecision(2) << std::setw(7) << pct(m.byte_fallback_count) << "%\n"
        << "║    atom tokens:      " << std::setw(12) << m.atom_count
        << "    " << std::setprecision(2) << std::setw(7) << pct(m.atom_count) << "%\n"
        << "║    <unk>:            " << std::setw(12) << m.unk_count
        << "    " << std::setprecision(2) << std::setw(7) << pct(m.unk_count) << "%\n"
        << "║    special (s/pad):  " << std::setw(12) << m.special_count
        << "    " << std::setprecision(2) << std::setw(7) << pct(m.special_count) << "%\n"
        << "╠══════════════════════════════════════════════════════════════╣\n"
        << "║  QUALITY METRICS                                           ║\n"
        << "║    Shannon entropy:  " << std::setprecision(4) << m.shannon_entropy << " bits"
        << "  (theoretical max: " << std::setprecision(2) << std::log2(std::max(1u, m.distinct_ids)) << ")\n"
        << "║    Bytes per token:  " << std::setprecision(4) << m.bytes_per_token << "\n"
        << "║    Fertility:        " << std::setprecision(4) << m.fertility << " tok/word\n"
        << "╠══════════════════════════════════════════════════════════════╣\n";

    // ── Top 30 most frequent tokens ──
    std::cout << "║  TOP 30 TOKENS (by frequency)                              ║\n";
    {
        std::vector<std::pair<uint32_t, uint64_t>> sorted_tokens(
            m.token_hist.begin(), m.token_hist.end());
        std::sort(sorted_tokens.begin(), sorted_tokens.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        int shown = 0;
        for (const auto& [id, count] : sorted_tokens) {
            if (shown >= 30) break;

            // Get display text
            std::string display;
            auto it = vocab.find(static_cast<int>(id));
            if (it != vocab.end()) {
                display = it->second;
            } else {
                display = "ID:" + std::to_string(id);
            }
            // Escape control chars for display
            for (char& c : display) {
                if (c == '\n') c = 'N';  // visual indicator
                else if (c == '\t') c = 'T';
                else if (c == '\r') c = 'R';
                else if (static_cast<unsigned char>(c) < 32) c = '.';
            }
            if (display.size() > 20) display = display.substr(0, 17) + "...";

            double p = 100.0 * count / m.total_tokens;
            std::cout << "║    " << std::setw(5) << (shown + 1) << ". "
                      << std::left << std::setw(22) << display << std::right
                      << " id=" << std::setw(6) << id
                      << "  n=" << std::setw(10) << count
                      << "  " << std::setprecision(3) << std::setw(7) << p << "%\n";
            ++shown;
        }
    }

    // ── Bottom 20 least frequent tokens (likely dead/waste) ──
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  BOTTOM 20 TOKENS (rarest observed)                        ║\n";
    {
        std::vector<std::pair<uint32_t, uint64_t>> sorted_tokens(
            m.token_hist.begin(), m.token_hist.end());
        std::sort(sorted_tokens.begin(), sorted_tokens.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        int shown = 0;
        for (const auto& [id, count] : sorted_tokens) {
            if (shown >= 20) break;

            std::string display;
            auto it = vocab.find(static_cast<int>(id));
            if (it != vocab.end()) {
                display = it->second;
            } else {
                display = "ID:" + std::to_string(id);
            }
            for (char& c : display) {
                if (static_cast<unsigned char>(c) < 32) c = '.';
            }
            if (display.size() > 20) display = display.substr(0, 17) + "...";

            std::cout << "║    " << std::setw(5) << (shown + 1) << ". "
                      << std::left << std::setw(22) << display << std::right
                      << " id=" << std::setw(6) << id
                      << "  n=" << std::setw(10) << count << "\n";
            ++shown;
        }
    }

    std::cout
        << "╚══════════════════════════════════════════════════════════════╝\n\n";
}

// ════════════════════════════════════════════════════════
//  Sanity checks — returns warning count
// ════════════════════════════════════════════════════════

int GRIM::Test::checkSanity(const GRMTCorpusMetrics& m) {
    int warnings = 0;
    auto warn = [&](const std::string& msg) {
        std::cout << "[WARNING] " << msg << "\n";
        ++warnings;
    };

    if (m.total_tokens == 0) {
        warn("No tokens scanned from GRMT file");
        return warnings;
    }
    if (!m.scan_ok) {
        warn("Read error during scan — only " + std::to_string(m.sequences_scanned) +
             "/" + std::to_string(m.num_sequences) + " sequences read");
    }
    if (m.shannon_entropy < 3.0)
        warn("Shannon entropy very low (" + std::to_string(m.shannon_entropy) +
             " bits) — corpus may lack diversity");
    if (m.shannon_entropy > 18.0)
        warn("Shannon entropy very high (" + std::to_string(m.shannon_entropy) +
             " bits) — possible near-uniform distribution");
    if (m.bytes_per_token < 1.0)
        warn("Bytes/token < 1.0 (" + std::to_string(m.bytes_per_token) +
             ") — impossible, data may be corrupt");
    if (m.bytes_per_token > 12.0)
        warn("Bytes/token > 12 (" + std::to_string(m.bytes_per_token) +
             ") — very aggressive subword merges");
    if (m.fertility > 10.0)
        warn("Fertility > 10 (" + std::to_string(m.fertility) +
             " tok/word) — tokenizer may be mostly byte-level");
    if (m.fertility < 0.5)
        warn("Fertility < 0.5 (" + std::to_string(m.fertility) +
             ") — word counting may be broken");

    double unk_pct = m.total_tokens > 0 ? 100.0 * m.unk_count / m.total_tokens : 0.0;
    if (unk_pct > 1.0)
        warn("<unk> rate is " + std::to_string(unk_pct) + "% — too many unknown tokens");

    double byte_pct = m.total_tokens > 0 ? 100.0 * m.byte_fallback_count / m.total_tokens : 0.0;
    if (byte_pct > 30.0)
        warn("Byte fallback rate is " + std::to_string(byte_pct) +
             "% — tokenizer may need a larger vocab or better training data");

    if (m.vocab_utilization < 50.0)
        warn("Only " + std::to_string(m.vocab_utilization) +
             "% of vocab IDs appear in corpus — " +
             std::to_string(m.dead_vocab_ids) + " dead entries");

    if (m.seq_len_max > 100000)
        warn("Max sequence length is " + std::to_string(m.seq_len_max) +
             " — suspiciously long");

    return warnings;
}

// ════════════════════════════════════════════════════════
//  main()
// ════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    GRIM::Tokenizer::configureTokenLayout(GRIM::Tokenizer::kAtomTypeCount);

    std::string vocab_path;
    std::string grmt_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--vocab" || arg == "-v") && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if ((arg == "--grmt" || arg == "-g") && i + 1 < argc) {
            grmt_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: grmt_vocab_metrics_test --vocab <vocab.bin> --grmt <training_data.grmt>\n";
            return 0;
        }
    }

    // Default paths relative to expected training data directory
    if (vocab_path.empty() || grmt_path.empty()) {
        std::vector<std::string> search_roots;
        const char* repo_root = std::getenv("GRIM_REPO_ROOT");
        if (repo_root) search_roots.push_back(std::string(repo_root));
        search_roots.push_back("../../../data");
        search_roots.push_back("../../../../resources/models/GRIM-text/training/data");
        search_roots.push_back("resources/models/GRIM-text/training/data");

        for (const auto& root : search_roots) {
            if (vocab_path.empty()) {
                std::string candidate = root + "/vocab.bin";
                if (fs::exists(candidate)) vocab_path = candidate;
            }
            if (grmt_path.empty()) {
                std::string candidate = root + "/training_data.grmt";
                if (fs::exists(candidate)) grmt_path = candidate;
            }
        }
    }

    if (vocab_path.empty()) {
        std::cerr << "ERROR: --vocab <path> required (or set GRIM_REPO_ROOT)\n";
        return 1;
    }
    if (grmt_path.empty()) {
        std::cerr << "ERROR: --grmt <path> required (or set GRIM_REPO_ROOT)\n";
        return 1;
    }
    if (!fs::exists(vocab_path)) {
        std::cerr << "ERROR: vocab.bin not found: " << vocab_path << "\n";
        return 1;
    }
    if (!fs::exists(grmt_path)) {
        std::cerr << "ERROR: training_data.grmt not found: " << grmt_path << "\n";
        return 1;
    }

    // ── Single scan ──
    std::cout << "Loading vocab: " << vocab_path << "\n";
    auto vocab = GRIM::Test::loadVocabMap(vocab_path);
    std::cout << "Loaded " << vocab.size() << " vocab entries\n\n";

    std::cout << "Scanning GRMT: " << grmt_path << "\n";
    auto metrics = GRIM::Test::scanGRMT(grmt_path, vocab);

    // ── Print everything ──
    GRIM::Test::printMetricsReport(metrics, vocab, vocab_path, grmt_path);

    // ── Sanity checks ──
    int warnings = GRIM::Test::checkSanity(metrics);
    if (warnings > 0) {
        std::cout << "\n" << warnings << " warning(s) detected.\n";
        return 1;
    }

    std::cout << "All sanity checks passed.\n";
    return 0;
}
