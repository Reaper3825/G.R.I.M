//======================================================//
//  grmt_vocab_metrics_test.cu
//  Streaming corpus + vocabulary diagnostics for .grmt
//
//  Computes Shannon entropy, bytes-per-token, fertility,
//  and sequence-length statistics in a single pass without
//  loading the entire corpus into memory.
//
//  Build:
//    cmake --build build --config Release \
//          --target grmt_vocab_metrics_test
//  Run:
//    ./build/Release/grmt_vocab_metrics_test \
//        --vocab  <path-to-vocab.bin> \
//        --grmt   <path-to-training_data.grmt>
//======================================================//

#include "grmt_vocab_metrics_test.hpp"
#include "../Shared/UnigramByte/Byte.hpp"
#include "../Shared/UnigramByte/Unigram.hpp"
#include "../Shared/UnigramByte/UniByte.hpp"
#include "../Common/grim_model_serialization_version.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
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

static std::unordered_map<int, std::string> loadVocabMap(const std::string& path) {
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
    if (version != 3) {
        throw std::runtime_error("loadVocabMap: unsupported vocab version " + std::to_string(version));
    }

    uint32_t checksum, config_vocab_size, max_length;
    f.read(reinterpret_cast<char*>(&checksum), 4);
    f.read(reinterpret_cast<char*>(&config_vocab_size), 4);
    f.read(reinterpret_cast<char*>(&max_length), 4);

    char flags[3];
    f.read(flags, 3);

    uint32_t total_vocab_size;
    f.read(reinterpret_cast<char*>(&total_vocab_size), 4);

    std::vector<char> buf;
    buf.reserve(256);

    for (uint32_t i = 0; i < config_vocab_size; ++i) {
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

static GRMTCorpusMetrics scanGRMT(
    const std::string& grmt_path,
    const std::unordered_map<int, std::string>& vocab)
{
    GRMTCorpusMetrics m;

    std::ifstream f(grmt_path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("scanGRMT: cannot open " + grmt_path);
    }

    // ── Header (16 bytes) ──
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x474D5254) {
        throw std::runtime_error("scanGRMT: invalid GRMT magic 0x" +
            ([&]{ char b[16]; snprintf(b, sizeof(b), "%08X", magic); return std::string(b); })());
    }

    f.read(reinterpret_cast<char*>(&m.grmt_version), 4);
    f.read(reinterpret_cast<char*>(&m.num_sequences), 4);
    f.read(reinterpret_cast<char*>(&m.vocab_size), 4);

    std::cout << "[grmt-metrics] GRMT v" << m.grmt_version
              << ", " << m.num_sequences << " sequences"
              << ", vocab_size=" << m.vocab_size << "\n";

    // Token ID histogram (sparse — only observed IDs)
    std::unordered_map<uint32_t, uint64_t> token_hist;
    token_hist.reserve(m.vocab_size);

    uint64_t total_utf8_bytes = 0;
    uint64_t total_words      = 0;
    m.seq_len_min = UINT32_MAX;
    m.seq_len_max = 0;
    uint64_t seq_len_sum = 0;

    // Reusable per-sequence buffers
    std::vector<int32_t> token_ids;
    std::string decoded;

    for (uint32_t s = 0; s < m.num_sequences; ++s) {
        uint32_t seq_len;
        f.read(reinterpret_cast<char*>(&seq_len), 4);
        if (!f || seq_len == 0 || seq_len > 1'000'000) break;

        // ── Sequence length stats ──
        m.seq_len_min = std::min(m.seq_len_min, seq_len);
        m.seq_len_max = std::max(m.seq_len_max, seq_len);
        seq_len_sum += seq_len;

        // ── Read token_ids ──
        token_ids.resize(seq_len);
        f.read(reinterpret_cast<char*>(token_ids.data()), seq_len * sizeof(int32_t));

        // ── Histogram ──
        for (uint32_t t = 0; t < seq_len; ++t) {
            ++token_hist[static_cast<uint32_t>(token_ids[t])];
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
        // text_features (uint16 × seq_len × kTextFeatureDim)
        f.seekg(seq_len * kTextFeatureDim * sizeof(uint16_t), std::ios::cur);
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
    m.distinct_ids = static_cast<uint32_t>(token_hist.size());
    m.seq_len_mean = m.num_sequences > 0
        ? static_cast<double>(seq_len_sum) / m.num_sequences
        : 0.0;

    // Shannon entropy H(token) in bits
    if (m.total_tokens > 0) {
        double h = 0.0;
        double N = static_cast<double>(m.total_tokens);
        for (auto& [id, count] : token_hist) {
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

    return m;
}

// ════════════════════════════════════════════════════════
//  Test registration
// ════════════════════════════════════════════════════════

void GRIM::Test::registerGRMTVocabMetricsTests(
    UnigramByteTestSuite& suite,
    const std::string& vocab_path,
    const std::string& grmt_path)
{
    // ── 1. Vocab file loads correctly ──
    suite.addTest("grmt-vocab: vocab.bin loads", [vocab_path](std::string& message) -> bool {
        auto vocab = loadVocabMap(vocab_path);
        ASSERT_TRUE(vocab.size() > 260,
            "Vocab too small (" + std::to_string(vocab.size()) + " entries). "
            "Expected at least 260 (specials + byte fallback).");
        message = std::to_string(vocab.size()) + " entries";
        return true;
    });

    // ── 2. GRMT header is valid ──
    suite.addTest("grmt-vocab: GRMT header valid", [grmt_path](std::string& message) -> bool {
        std::ifstream f(grmt_path, std::ios::binary);
        ASSERT_TRUE(f.is_open(), "Cannot open GRMT file: " + grmt_path);

        uint32_t magic, version, num_seq, vocab_size;
        f.read(reinterpret_cast<char*>(&magic), 4);
        f.read(reinterpret_cast<char*>(&version), 4);
        f.read(reinterpret_cast<char*>(&num_seq), 4);
        f.read(reinterpret_cast<char*>(&vocab_size), 4);

        ASSERT_EQ(magic, 0x474D5254u, "Bad GRMT magic");
        ASSERT_TRUE(version >= 4, "GRMT version too old: " + std::to_string(version));
        ASSERT_TRUE(num_seq > 0, "Zero sequences in GRMT");
        ASSERT_TRUE(vocab_size > 0, "Zero vocab_size in GRMT header");

        message = "v" + std::to_string(version) + ", "
                + std::to_string(num_seq) + " seqs, vocab=" + std::to_string(vocab_size);
        return true;
    });

    // ── 3. Vocab/GRMT vocab_size match ──
    suite.addTest("grmt-vocab: vocab sizes match", [vocab_path, grmt_path](std::string& message) -> bool {
        auto vocab = loadVocabMap(vocab_path);
        uint32_t vocab_total = static_cast<uint32_t>(vocab.size());

        std::ifstream f(grmt_path, std::ios::binary);
        ASSERT_TRUE(f.is_open(), "Cannot open GRMT file");

        uint32_t magic, version, num_seq, grmt_vocab_size;
        f.read(reinterpret_cast<char*>(&magic), 4);
        f.read(reinterpret_cast<char*>(&version), 4);
        f.read(reinterpret_cast<char*>(&num_seq), 4);
        f.read(reinterpret_cast<char*>(&grmt_vocab_size), 4);

        ASSERT_EQ(vocab_total, grmt_vocab_size,
            "vocab.bin total (" + std::to_string(vocab_total) + ") != "
            "GRMT header vocab_size (" + std::to_string(grmt_vocab_size) + ")");

        message = "both report " + std::to_string(grmt_vocab_size);
        return true;
    });

    // ── 4. Full corpus metrics scan ──
    suite.addTest("grmt-vocab: corpus metrics", [vocab_path, grmt_path](std::string& message) -> bool {
        auto vocab = loadVocabMap(vocab_path);
        auto m = scanGRMT(grmt_path, vocab);

        ASSERT_TRUE(m.total_tokens > 0, "No tokens scanned from GRMT");
        ASSERT_TRUE(m.distinct_ids > 1, "Only " + std::to_string(m.distinct_ids) + " distinct token IDs");

        // Print the metrics report
        std::cout << "\n"
            << "╔══════════════════════════════════════════════════╗\n"
            << "║        GRMT Corpus + Vocabulary Metrics         ║\n"
            << "╠══════════════════════════════════════════════════╣\n"
            << "║  vocab.bin:       " << vocab_path << "\n"
            << "║  training_data:   " << grmt_path << "\n"
            << "║  GRMT version:    " << m.grmt_version << "\n"
            << "╠══════════════════════════════════════════════════╣\n"
            << "║  sequences:       " << m.num_sequences << "\n"
            << "║  total_tokens:    " << m.total_tokens << "\n"
            << "║  |V| observed:    " << m.distinct_ids << " distinct IDs\n"
            << "║  vocab_size (hdr): " << m.vocab_size << "\n"
            << "╠══════════════════════════════════════════════════╣\n"
            << "║  seq_len min:     " << m.seq_len_min << "\n"
            << "║  seq_len max:     " << m.seq_len_max << "\n"
            << "║  seq_len mean:    " << std::fixed << std::setprecision(1) << m.seq_len_mean << "\n"
            << "╠══════════════════════════════════════════════════╣\n"
            << "║  Shannon entropy: " << std::fixed << std::setprecision(6) << m.shannon_entropy << " bits\n"
            << "║  Bytes per token: " << std::fixed << std::setprecision(6) << m.bytes_per_token << "\n"
            << "║  Fertility:       " << std::fixed << std::setprecision(6) << m.fertility << " tok/word\n"
            << "╚══════════════════════════════════════════════════╝\n";

        message = "H=" + std::to_string(m.shannon_entropy).substr(0, 8) + " bits"
                + ", B/T=" + std::to_string(m.bytes_per_token).substr(0, 6)
                + ", fert=" + std::to_string(m.fertility).substr(0, 6);
        return true;
    });

    // ── 5. Shannon entropy sanity check ──
    suite.addTest("grmt-vocab: entropy range", [vocab_path, grmt_path](std::string& message) -> bool {
        auto vocab = loadVocabMap(vocab_path);
        auto m = scanGRMT(grmt_path, vocab);

        // For a reasonable subword tokenizer on natural language:
        //   H should be between ~5 and ~16 bits.
        //   <5 implies near-uniform tokens or pathologically small vocab.
        //   >16 implies >65k distinct buckets with near-uniform usage.
        ASSERT_TRUE(m.shannon_entropy > 3.0,
            "Shannon entropy suspiciously low (" + std::to_string(m.shannon_entropy) + " bits)");
        ASSERT_TRUE(m.shannon_entropy < 20.0,
            "Shannon entropy suspiciously high (" + std::to_string(m.shannon_entropy) + " bits)");

        message = std::to_string(m.shannon_entropy) + " bits (expected 5-16)";
        return true;
    });

    // ── 6. Bytes per token sanity ──
    suite.addTest("grmt-vocab: bytes/token range", [vocab_path, grmt_path](std::string& message) -> bool {
        auto vocab = loadVocabMap(vocab_path);
        auto m = scanGRMT(grmt_path, vocab);

        // Reasonable range: 1.0 (pure byte) to ~12 (very aggressive merges).
        ASSERT_TRUE(m.bytes_per_token >= 1.0,
            "bytes/token < 1.0 (" + std::to_string(m.bytes_per_token) + ") — impossible");
        ASSERT_TRUE(m.bytes_per_token < 20.0,
            "bytes/token too high (" + std::to_string(m.bytes_per_token) + ") — vocab may be corrupt");

        message = std::to_string(m.bytes_per_token);
        return true;
    });

    // ── 7. Fertility sanity ──
    suite.addTest("grmt-vocab: fertility range", [vocab_path, grmt_path](std::string& message) -> bool {
        auto vocab = loadVocabMap(vocab_path);
        auto m = scanGRMT(grmt_path, vocab);

        // Typical unigram tokenizers: 1.1 – 2.5 tokens per word.
        // Byte-level fallback can push higher, but >10 signals a problem.
        ASSERT_TRUE(m.fertility > 0.5,
            "Fertility too low (" + std::to_string(m.fertility) + ") — possibly no words decoded");
        ASSERT_TRUE(m.fertility < 15.0,
            "Fertility too high (" + std::to_string(m.fertility) + ") — tokenizer may be mostly byte-level");

        message = std::to_string(m.fertility) + " tok/word";
        return true;
    });

    // ── 8. No unknown tokens dominate ──
    suite.addTest("grmt-vocab: <unk> usage", [vocab_path, grmt_path](std::string& message) -> bool {
        auto vocab = loadVocabMap(vocab_path);

        // Quick scan just for UNK token count
        std::ifstream f(grmt_path, std::ios::binary);
        ASSERT_TRUE(f.is_open(), "Cannot open GRMT file");

        uint32_t magic, version, num_seq, vocab_size;
        f.read(reinterpret_cast<char*>(&magic), 4);
        f.read(reinterpret_cast<char*>(&version), 4);
        f.read(reinterpret_cast<char*>(&num_seq), 4);
        f.read(reinterpret_cast<char*>(&vocab_size), 4);

        uint64_t unk_count = 0;
        uint64_t total = 0;
        std::vector<int32_t> tokens;

        for (uint32_t s = 0; s < num_seq; ++s) {
            uint32_t seq_len;
            f.read(reinterpret_cast<char*>(&seq_len), 4);
            if (!f || seq_len == 0 || seq_len > 1'000'000) break;

            tokens.resize(seq_len);
            f.read(reinterpret_cast<char*>(tokens.data()), seq_len * sizeof(int32_t));
            total += seq_len;
            for (int32_t tid : tokens) {
                if (tid == UNK_TOKEN_ID) ++unk_count;
            }

            // Skip remaining fields (same layout as scanGRMT)
            f.seekg(seq_len * sizeof(int32_t), std::ios::cur);  // targets
            f.seekg(seq_len * sizeof(float), std::ios::cur);    // numeric_values
            f.seekg(seq_len * sizeof(uint8_t), std::ios::cur);  // atom_mask
            f.seekg(seq_len * kTextFeatureDim * sizeof(uint16_t), std::ios::cur); // text_features
            f.seekg(seq_len * sizeof(uint32_t), std::ios::cur); // atom_flags
            for (uint32_t j = 0; j < seq_len; ++j) {           // atom text strings
                uint16_t slen = 0;
                f.read(reinterpret_cast<char*>(&slen), sizeof(uint16_t));
                if (slen > 0) f.seekg(slen, std::ios::cur);
            }
            f.seekg(sizeof(uint8_t), std::ios::cur);            // exec_active
            f.seekg(seq_len * sizeof(int32_t), std::ios::cur);  // token_exec_slots
            { uint32_t n = 0; f.read(reinterpret_cast<char*>(&n), 4); if (n) f.seekg(n * 12, std::ios::cur); }
            { uint32_t n = 0; f.read(reinterpret_cast<char*>(&n), 4); if (n) f.seekg(n * 20, std::ios::cur); }
            { uint32_t n = 0; f.read(reinterpret_cast<char*>(&n), 4); if (n) f.seekg(n * 5, std::ios::cur); }

            if (!f) break;
        }

        double unk_pct = total > 0 ? 100.0 * unk_count / total : 0.0;
        ASSERT_TRUE(unk_pct < 1.0,
            "<unk> tokens are " + std::to_string(unk_pct) + "% of corpus — too high");

        message = std::to_string(unk_count) + " <unk> out of "
                + std::to_string(total) + " (" + std::to_string(unk_pct).substr(0, 6) + "%)";
        return true;
    });
}

// ════════════════════════════════════════════════════════
//  main()
// ════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    // Configure token layout (must happen before using ATOM_VOCAB_SIZE etc.)
    GRIM::Tokenizer::configureTokenLayout(GRIM::Tokenizer::kAtomTypeCount);

    // Parse --vocab and --grmt from command line
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
        // Try to locate from known relative paths
        std::vector<std::string> search_roots;

        // Check GRIM_REPO_ROOT environment variable
        const char* repo_root = std::getenv("GRIM_REPO_ROOT");
        if (repo_root) search_roots.push_back(std::string(repo_root));

        // Common relative paths from build directory
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

    std::cout << "vocab:  " << vocab_path << "\n";
    std::cout << "grmt:   " << grmt_path << "\n\n";

    UnigramByteTestSuite suite;
    registerGRMTVocabMetricsTests(suite, vocab_path, grmt_path);

    auto results = suite.runAll();

    int failed = 0;
    for (const auto& r : results) {
        if (!r.passed) ++failed;
    }

    return failed > 0 ? 1 : 0;
}
