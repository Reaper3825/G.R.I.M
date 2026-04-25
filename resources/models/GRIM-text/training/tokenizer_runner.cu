//======================================================//
//  Tokenizer Runner — Standalone Tokenizer Executable
//
//  Runs the full tokenizer validation pipeline independently:
//    1. Load config from ai_config.json
//    2. Validate paths (vocab, training data)
//    3. Load tokenizer and training data
//    4. Run self-test validation checks
//    5. Output JSON payload to stdout on success
//    6. Output JSON error to stdout on failure
//
//  The training executable (train_gpu) launches this as a
//  subprocess before starting the training loop. The JSON
//  payload communicates tokenizer metadata back to the
//  training process.
//
//  Can also be run standalone from the UI for vocab testing.
//
//  Exit codes:
//    0 = success (payload on stdout)
//    1 = fatal error (error JSON on stdout)
//    2 = validation failure (error JSON on stdout)
//
//  Author: Austin Wadkins
//  Date: April 2026
//======================================================//

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <cstring>

#include <nlohmann/json.hpp>

#include "../Shared/UnigramByte/UniByte.hpp"
#include "../Common/grim_model_serialization_version.hpp"
#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"  // single entry point; pulls in control/ai_config_paths.hpp transitively

namespace fs = std::filesystem;
using json = nlohmann::json;
using GrimTokenizer = GRIM::Tokenizer::UniByte;

//======================================================//
//  CLI Options
//======================================================//
struct RunnerOptions {
    std::string vocab_path;
    std::string data_path;
    std::string config_path = "ai_config.json";
    std::string encode_text;  // non-empty = encode mode
    bool verbose = false;
    bool standalone = false;  // true = standalone mode (human-readable output)
};

//======================================================//
//  Validation Result
//======================================================//
struct ValidationResult {
    bool passed = false;
    std::string name;
    std::string details;
};

//======================================================//
//  Tokenizer Payload — JSON output on success
//======================================================//
struct TokenizerPayload {
    int total_vocab_size = 0;
    int unigram_vocab_size = 0;
    int byte_vocab_size = 0;
    int atom_vocab_size = 0;
    int special_token_count = 0;
    int pad_id = 0;
    int unk_id = 0;
    int bos_id = 0;
    int eos_id = 0;
    std::string vocab_path;
    std::string data_path;
    int validation_tests_passed = 0;
    int validation_tests_total = 0;
    double load_time_ms = 0.0;
    double validation_time_ms = 0.0;

    json toJson() const {
        json j;
        j["status"] = "success";
        j["total_vocab_size"] = total_vocab_size;
        j["unigram_vocab_size"] = unigram_vocab_size;
        j["byte_vocab_size"] = byte_vocab_size;
        j["atom_vocab_size"] = atom_vocab_size;
        j["special_token_count"] = special_token_count;
        j["pad_id"] = pad_id;
        j["unk_id"] = unk_id;
        j["bos_id"] = bos_id;
        j["eos_id"] = eos_id;
        j["vocab_path"] = vocab_path;
        j["data_path"] = data_path;
        j["validation_tests_passed"] = validation_tests_passed;
        j["validation_tests_total"] = validation_tests_total;
        j["load_time_ms"] = load_time_ms;
        j["validation_time_ms"] = validation_time_ms;
        return j;
    }
};

//======================================================//
//  Error output — JSON on failure
//======================================================//
static json makeErrorJson(const std::string& error, const std::string& phase, int tests_passed = 0, int tests_total = 0) {
    json j;
    j["status"] = "error";
    j["error"] = error;
    j["phase"] = phase;
    j["validation_tests_passed"] = tests_passed;
    j["validation_tests_total"] = tests_total;
    return j;
}

//======================================================//
//  GRMT Corpus Sampler — read token_id sequences from
//  the actual training data file for validation
//======================================================//
struct GRMTSample {
    uint32_t grmt_vocab_size = 0;
    uint32_t num_sequences = 0;
    std::vector<std::vector<int>> sampled_sequences;  // token_id arrays
};

static GRMTSample sampleGRMTSequences(const std::string& data_path, int max_samples, bool verbose) {
    GRMTSample result;

    std::ifstream file(data_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open GRMT file: " + data_path);
    }

    // Read header
    uint32_t magic = 0, version = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&version), 4);
    file.read(reinterpret_cast<char*>(&result.num_sequences), 4);
    file.read(reinterpret_cast<char*>(&result.grmt_vocab_size), 4);

    if (magic != 0x474D5254) {
        throw std::runtime_error("Invalid GRMT magic: 0x" +
            ([&]{ char buf[16]; snprintf(buf, sizeof(buf), "%08X", magic); return std::string(buf); })());
    }
    if (version != GRIM::GRMT_FORMAT_VERSION) {
        throw std::runtime_error("GRMT version mismatch: file=" + std::to_string(version) +
            " expected=" + std::to_string(GRIM::GRMT_FORMAT_VERSION));
    }
    if (result.num_sequences == 0) {
        throw std::runtime_error("GRMT file has 0 sequences");
    }

    // Compute which sequence indices to sample (evenly spaced)
    int step = std::max(1u, result.num_sequences / static_cast<uint32_t>(max_samples));
    std::vector<uint32_t> sample_indices;
    for (uint32_t i = 0; i < result.num_sequences && static_cast<int>(sample_indices.size()) < max_samples; i += step) {
        sample_indices.push_back(i);
    }

    if (verbose) {
        fprintf(stderr, "[tokenizer_runner] Sampling %zu/%u sequences from GRMT\n",
                sample_indices.size(), result.num_sequences);
    }

    size_t next_sample = 0;
    for (uint32_t i = 0; i < result.num_sequences; ++i) {
        uint32_t seq_len = 0;
        file.read(reinterpret_cast<char*>(&seq_len), 4);
        if (!file) throw std::runtime_error("GRMT read error at sequence " + std::to_string(i));

        bool keep = (next_sample < sample_indices.size() && sample_indices[next_sample] == i);

        // Read token_ids
        std::vector<int> token_ids(seq_len);
        file.read(reinterpret_cast<char*>(token_ids.data()), seq_len * sizeof(int));

        if (keep) {
            result.sampled_sequences.push_back(std::move(token_ids));
            next_sample++;
        }

        // Skip remaining per-sequence fields to maintain file position:
        // targets (int[seq_len]) + numeric_values (float[seq_len]) + atom_mask (uint8[seq_len])
        // + text_features (uint16[seq_len * 16]) + atom_flags (uint32[seq_len])
        size_t skip_bytes = seq_len * sizeof(int)           // targets
                          + seq_len * sizeof(float)         // numeric_values
                          + seq_len * sizeof(uint8_t)       // atom_mask
                          + seq_len * GRIM::Tokenizer::kTextFeatureDim * sizeof(uint16_t)  // text_features
                          + seq_len * sizeof(uint32_t);     // atom_flags
        file.seekg(static_cast<std::streamoff>(skip_bytes), std::ios::cur);

        // Skip variable-length atom text strings
        for (uint32_t j = 0; j < seq_len; ++j) {
            uint16_t slen = 0;
            file.read(reinterpret_cast<char*>(&slen), sizeof(uint16_t));
            if (slen > 0) file.seekg(slen, std::ios::cur);
        }

        // Skip v11 execution payload
        {
            uint8_t exec_active = 0;
            file.read(reinterpret_cast<char*>(&exec_active), sizeof(uint8_t));
            // token_exec_slots
            file.seekg(static_cast<std::streamoff>(seq_len * sizeof(int32_t)), std::ios::cur);
            // compiled_bootstrap_bindings
            uint32_t cbb_count = 0;
            file.read(reinterpret_cast<char*>(&cbb_count), sizeof(uint32_t));
            if (cbb_count > 0) file.seekg(static_cast<std::streamoff>(cbb_count * 12), std::ios::cur);
            // teacher_steps
            uint32_t ts_count = 0;
            file.read(reinterpret_cast<char*>(&ts_count), sizeof(uint32_t));
            if (ts_count > 0) file.seekg(static_cast<std::streamoff>(ts_count * 20), std::ios::cur);
            // slot_selection_targets (5 bytes each: uint8 kind + int32 slot_id)
            uint32_t sst_count = 0;
            file.read(reinterpret_cast<char*>(&sst_count), sizeof(uint32_t));
            if (sst_count > 0) file.seekg(static_cast<std::streamoff>(sst_count * 5), std::ios::cur);
        }

        if (!file) throw std::runtime_error("GRMT read/seek error at sequence " + std::to_string(i));

        // Early exit once we have all samples
        if (next_sample >= sample_indices.size()) break;
    }

    return result;
}

//======================================================//
//  Raw Text Sampler — read raw text from
//  concept_blocks.jsonl (sibling of .grmt) so we can
//  test the tokenizer against pre-tokenization text,
//  not against its own baked output.
//======================================================//
static std::vector<std::string> sampleRawCorpusText(const std::string& data_path, int max_samples, bool verbose) {
    // concept_blocks.jsonl sits in the same directory as the .grmt file
    fs::path grmt_path(data_path);
    fs::path corpus_path = grmt_path.parent_path() / "concept_blocks.jsonl";

    if (!fs::exists(corpus_path)) {
        if (verbose) {
            fprintf(stderr, "[tokenizer_runner] No concept_blocks.jsonl at %s — raw text tests will be skipped\n",
                    corpus_path.string().c_str());
        }
        return {};
    }

    std::ifstream in(corpus_path);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open concept_blocks.jsonl: " + corpus_path.string());
    }

    // Count lines first (cheap) to compute even sampling
    std::vector<std::streampos> line_offsets;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            line_offsets.push_back(in.tellg() - static_cast<std::streamoff>(line.size() + 1));
        }
    }

    if (line_offsets.empty()) {
        throw std::runtime_error("concept_blocks.jsonl is empty");
    }

    int total_lines = static_cast<int>(line_offsets.size());
    int step = std::max(1, total_lines / max_samples);

    if (verbose) {
        fprintf(stderr, "[tokenizer_runner] Sampling %d/%d entries from concept_blocks.jsonl\n",
                std::min(max_samples, total_lines), total_lines);
    }

    std::vector<std::string> texts;
    texts.reserve(std::min(max_samples, total_lines));

    in.clear();
    in.seekg(0);
    int line_idx = 0;
    int next_target = 0;
    while (std::getline(in, line) && static_cast<int>(texts.size()) < max_samples) {
        if (line.empty()) continue;
        if (line_idx == next_target) {
            try {
                auto j = json::parse(line);
                // Extract raw text the same way DataLoader::renderPlainText does:
                // question + explanation/intermediates + answer
                std::string text;
                if (j.contains("question") && j["question"].is_string()) {
                    text += j["question"].get<std::string>();
                    text += "\n";
                }
                const json* expl = nullptr;
                if (j.contains("explanation") && j["explanation"].is_array())
                    expl = &j["explanation"];
                else if (j.contains("intermediates") && j["intermediates"].is_array())
                    expl = &j["intermediates"];
                if (expl) {
                    for (const auto& s : *expl) {
                        if (s.is_string()) {
                            text += s.get<std::string>();
                            text += "\n";
                        }
                    }
                }
                if (j.contains("answer") && j["answer"].is_string()) {
                    text += j["answer"].get<std::string>();
                    text += "\n";
                }

                if (!text.empty()) {
                    texts.push_back(std::move(text));
                }
            } catch (const std::exception&) {
                // Skip malformed JSON lines
            }
            next_target += step;
        }
        line_idx++;
    }

    return texts;
}

//======================================================//
//  Validation Tests
//
//  Two levels of validation:
//  A) GRMT consistency — do the baked token IDs agree with
//     this tokenizer's vocab and decode correctly?
//  B) Raw text quality gate — encode raw pre-tokenization
//     text and verify the tokenizer reproduces it exactly.
//     This catches bad vocabs that the GRMT tests cannot.
//======================================================//
static constexpr int CORPUS_SAMPLE_COUNT = 32;

static std::vector<ValidationResult> runValidationChecks(
        GrimTokenizer& tokenizer,
        const std::string& data_path,
        bool verbose) {
    std::vector<ValidationResult> results;

    // ── Load corpus samples ──────────────────────────────
    GRMTSample corpus = sampleGRMTSequences(data_path, CORPUS_SAMPLE_COUNT, verbose);
    std::vector<std::string> raw_texts = sampleRawCorpusText(data_path, CORPUS_SAMPLE_COUNT, verbose);

    //==========================================================
    // GRMT Consistency Tests (A)
    //==========================================================

    // Test 1: GRMT-tokenizer vocab size consistency
    {
        ValidationResult r;
        r.name = "GRMT/tokenizer vocab consistency";
        int tok_vocab = tokenizer.totalVocabSize();
        int grmt_vocab = static_cast<int>(corpus.grmt_vocab_size);
        r.passed = (tok_vocab == grmt_vocab);
        if (!r.passed) {
            r.details = "MISMATCH: tokenizer=" + std::to_string(tok_vocab) +
                        " GRMT=" + std::to_string(grmt_vocab) +
                        " — GRMT data was encoded with a different vocab. Delete .grmt and regenerate.";
        } else {
            r.details = "Both report " + std::to_string(tok_vocab) + " tokens";
        }
        results.push_back(r);
    }

    // Test 2: Special token IDs are in valid range
    {
        ValidationResult r;
        r.name = "Special token IDs in range";
        int total = tokenizer.totalVocabSize();
        int pad = tokenizer.padId(), unk = tokenizer.unkId();
        int bos = tokenizer.bosId(), eos = tokenizer.eosId();
        bool all_valid = (pad >= 0 && pad < total) &&
                         (unk >= 0 && unk < total) &&
                         (bos >= 0 && bos < total) &&
                         (eos >= 0 && eos < total);
        bool all_unique = (pad != unk && pad != bos && pad != eos &&
                           unk != bos && unk != eos && bos != eos);
        r.passed = all_valid && all_unique;
        if (!r.passed) {
            r.details = "PAD=" + std::to_string(pad) + " UNK=" + std::to_string(unk) +
                        " BOS=" + std::to_string(bos) + " EOS=" + std::to_string(eos) +
                        " (total=" + std::to_string(total) + ")";
        } else {
            r.details = "PAD=" + std::to_string(pad) + " UNK=" + std::to_string(unk) +
                        " BOS=" + std::to_string(bos) + " EOS=" + std::to_string(eos);
        }
        results.push_back(r);
    }

    // Test 3: Corpus token ID range — every ID in sampled sequences must be valid
    {
        ValidationResult r;
        r.name = "Corpus token IDs in range";
        int total = tokenizer.totalVocabSize();
        size_t total_tokens = 0;
        size_t oob_count = 0;
        int worst_id = 0;
        for (const auto& seq : corpus.sampled_sequences) {
            for (int id : seq) {
                total_tokens++;
                if (id < 0 || id >= total) {
                    oob_count++;
                    worst_id = id;
                }
            }
        }
        r.passed = (oob_count == 0);
        if (!r.passed) {
            r.details = std::to_string(oob_count) + "/" + std::to_string(total_tokens) +
                        " tokens out of range [0," + std::to_string(total) +
                        ") — worst ID=" + std::to_string(worst_id);
        } else {
            r.details = std::to_string(total_tokens) + " tokens all in [0," +
                        std::to_string(total) + ")";
        }
        results.push_back(r);
    }

    // Test 4: GRMT decode round-trip — decode GRMT sequences (filtering special
    //         AND atom tokens), re-encode, re-decode, verify exact match.
    //         Atom tokens decode to placeholder strings that won't re-encode
    //         identically, so they must be excluded from the content pipeline.
    {
        ValidationResult r;
        r.name = "GRMT decode round-trip (exact)";
        int tested = 0, passed_count = 0;
        std::string worst_orig, worst_rt;
        for (const auto& seq : corpus.sampled_sequences) {
            if (seq.empty()) continue;
            // Filter out special tokens AND atom placeholders
            std::vector<int> content_ids;
            for (int id : seq) {
                if (tokenizer.isSpecialToken(id)) continue;
                if (tokenizer.isAtomToken(id)) continue;
                content_ids.push_back(id);
            }
            if (content_ids.empty()) continue;

            std::string decoded = tokenizer.decode(content_ids);
            if (decoded.empty()) continue;

            auto re_encoded = tokenizer.encode(decoded);
            std::string re_decoded = tokenizer.decode(re_encoded);

            // Exact comparison — no lowercasing, no whitespace normalization.
            // The tokenizer must be lossless on its own non-atom output.
            tested++;
            if (decoded == re_decoded) {
                passed_count++;
            } else if (worst_orig.empty()) {
                worst_orig = decoded.substr(0, 100);
                worst_rt = re_decoded.substr(0, 100);
            }
        }
        // 100% required — any failure means the tokenizer is lossy
        r.passed = (tested > 0 && passed_count == tested);
        if (!r.passed) {
            r.details = std::to_string(passed_count) + "/" + std::to_string(tested) +
                        " exact round-trips";
            if (!worst_orig.empty()) {
                r.details += " — first diff: '" + worst_orig + "' vs '" + worst_rt + "'";
            }
        } else {
            r.details = "All " + std::to_string(tested) +
                        " sequences exact round-tripped (special+atom tokens excluded)";
        }
        results.push_back(r);
    }

    // Test 5: Corpus decode coverage — sampled sequences decode to non-empty text
    {
        ValidationResult r;
        r.name = "Corpus decode coverage";
        int tested = 0, non_empty = 0;
        for (const auto& seq : corpus.sampled_sequences) {
            if (seq.empty()) continue;
            tested++;
            std::string decoded = tokenizer.decode(seq);
            if (!decoded.empty()) non_empty++;
        }
        r.passed = (tested > 0 && non_empty == tested);
        if (!r.passed) {
            r.details = std::to_string(non_empty) + "/" + std::to_string(tested) +
                        " decoded to non-empty text";
        } else {
            r.details = "All " + std::to_string(tested) + " sequences decoded to text";
        }
        results.push_back(r);
    }

    // Test 6: Token type distribution — corpus uses unigram, byte, and atom tokens
    {
        ValidationResult r;
        r.name = "Corpus token type distribution";
        size_t byte_count = 0, atom_count = 0, unigram_count = 0, special_count = 0;
        size_t total_tokens = 0;
        for (const auto& seq : corpus.sampled_sequences) {
            for (int id : seq) {
                total_tokens++;
                if (tokenizer.isSpecialToken(id))       special_count++;
                else if (tokenizer.isByteToken(id))     byte_count++;
                else if (tokenizer.isAtomToken(id))     atom_count++;
                else if (tokenizer.isUnigramToken(id))  unigram_count++;
            }
        }
        // Unigram tokens must appear — if 0, the vocab is broken or not loaded
        r.passed = (unigram_count > 0 && total_tokens > 0);
        if (!r.passed) {
            r.details = "No unigram tokens in corpus — vocab may be corrupt or empty. "
                        "unigram=0, byte=" + std::to_string(byte_count) +
                        ", atom=" + std::to_string(atom_count) +
                        ", special=" + std::to_string(special_count);
        } else {
            auto pct = [&](size_t n) -> std::string {
                if (total_tokens == 0) return "0.0";
                char buf[16];
                snprintf(buf, sizeof(buf), "%.1f", 100.0 * n / total_tokens);
                return buf;
            };
            r.details = "unigram=" + pct(unigram_count) + "% byte=" + pct(byte_count) +
                        "% atom=" + pct(atom_count) + "% special=" + pct(special_count) +
                        "% (N=" + std::to_string(total_tokens) + ")";
        }
        results.push_back(r);
    }

    // Test 7: Corpus BOS/EOS framing — sequences should start with BOS and end with EOS
    {
        ValidationResult r;
        r.name = "Corpus BOS/EOS framing";
        int tested = 0, correct = 0;
        int bos = tokenizer.bosId(), eos = tokenizer.eosId();
        for (const auto& seq : corpus.sampled_sequences) {
            if (seq.size() < 2) continue;
            tested++;
            if (seq.front() == bos && seq.back() == eos) correct++;
        }
        r.passed = (tested > 0 && correct == tested);
        if (!r.passed) {
            r.details = std::to_string(correct) + "/" + std::to_string(tested) +
                        " sequences properly framed with BOS/EOS";
        } else {
            r.details = "All " + std::to_string(tested) + " sequences have BOS/EOS framing";
        }
        results.push_back(r);
    }

    //==========================================================
    // Raw Text Quality Gate (B)
    // These test the tokenizer against raw pre-tokenization text
    // from concept_blocks.jsonl. This catches a bad vocab that
    // GRMT consistency tests cannot — if the tokenizer was broken
    // when GRMT was built, tests 1-7 can still pass because
    // they grade the tokenizer against its own baked output.
    //==========================================================

    if (!raw_texts.empty()) {

        // Test 8: Raw text exact round-trip — encode raw corpus text, decode,
        //         verify we get the original back byte-for-byte.
        {
            ValidationResult r;
            r.name = "Raw text exact round-trip";
            int tested = 0, passed_count = 0;
            std::string worst_input, worst_output;
            for (const auto& text : raw_texts) {
                if (text.size() < 10) continue;  // Skip trivially short entries

                auto ids = tokenizer.encode(text);
                std::string decoded = tokenizer.decode(ids);

                tested++;
                if (decoded == text) {
                    passed_count++;
                } else if (worst_input.empty()) {
                    worst_input = text.substr(0, 100);
                    worst_output = decoded.substr(0, 100);
                }
            }
            // 100% required — the tokenizer must be lossless on all corpus text
            r.passed = (tested > 0 && passed_count == tested);
            if (!r.passed) {
                r.details = std::to_string(passed_count) + "/" + std::to_string(tested) +
                            " raw texts survived encode→decode";
                if (!worst_input.empty()) {
                    r.details += " — first diff: '" + worst_input + "' vs '" + worst_output + "'";
                }
            } else {
                r.details = "All " + std::to_string(tested) +
                            " raw corpus texts exact round-tripped";
            }
            results.push_back(r);
        }

        // Test 9: Raw text token efficiency — chars/token on raw text (independent
        //         of GRMT, so a bad vocab trained on garbage will show low ratio)
        {
            ValidationResult r;
            r.name = "Raw text token efficiency";
            size_t total_chars = 0, total_tokens = 0;
            for (const auto& text : raw_texts) {
                if (text.size() < 10) continue;
                auto ids = tokenizer.encode(text);
                total_chars += text.size();
                total_tokens += ids.size();
            }
            double ratio = (total_tokens > 0) ? static_cast<double>(total_chars) / total_tokens : 0.0;
            r.passed = (ratio > 1.5);  // Stricter than GRMT test — raw text should compress well
            char buf[64];
            snprintf(buf, sizeof(buf), "%.2f chars/token", ratio);
            if (!r.passed) {
                r.details = std::string(buf) +
                            " — tokenizer compresses raw text poorly (expected >1.5)";
            } else {
                r.details = std::string(buf) + " on " + std::to_string(raw_texts.size()) +
                            " raw corpus entries (" + std::to_string(total_chars) + " chars)";
            }
            results.push_back(r);
        }

        // Test 10: Raw text case and whitespace fidelity — verify the tokenizer
        //          preserves mixed case, leading/trailing whitespace, and multiple
        //          consecutive spaces exactly as they appear in the raw corpus.
        {
            ValidationResult r;
            r.name = "Raw text case/whitespace fidelity";
            int tested = 0;
            int case_failures = 0, ws_failures = 0;
            for (const auto& text : raw_texts) {
                if (text.size() < 20) continue;
                tested++;

                auto ids = tokenizer.encode(text);
                std::string decoded = tokenizer.decode(ids);

                // Check case preservation
                bool case_ok = true;
                size_t min_len = std::min(text.size(), decoded.size());
                for (size_t ci = 0; ci < min_len; ++ci) {
                    if (text[ci] != decoded[ci] &&
                        std::tolower(static_cast<unsigned char>(text[ci])) ==
                        std::tolower(static_cast<unsigned char>(decoded[ci]))) {
                        case_ok = false;
                        break;
                    }
                }
                if (!case_ok) case_failures++;

                // Check whitespace preservation
                auto extract_ws = [](const std::string& s) {
                    std::string ws;
                    for (char c : s) {
                        if (std::isspace(static_cast<unsigned char>(c))) ws += c;
                    }
                    return ws;
                };
                if (extract_ws(text) != extract_ws(decoded)) ws_failures++;
            }
            r.passed = (tested > 0 && case_failures == 0 && ws_failures == 0);
            if (!r.passed) {
                r.details = std::to_string(case_failures) + " case failures, " +
                            std::to_string(ws_failures) + " whitespace failures in " +
                            std::to_string(tested) + " texts";
            } else {
                r.details = "All " + std::to_string(tested) +
                            " texts preserve case and whitespace exactly";
            }
            results.push_back(r);
        }
    } else {
        // No concept_blocks.jsonl found — log warning but don't fail
        ValidationResult r;
        r.name = "Raw text quality gate";
        r.passed = true;
        r.details = "SKIPPED — concept_blocks.jsonl not found alongside GRMT. "
                     "Raw text validation unavailable.";
        results.push_back(r);
    }

    // Test 11: Empty input encode — encode("") must not crash
    {
        ValidationResult r;
        r.name = "Empty input encode safety";
        auto ids = tokenizer.encode("");
        r.passed = true;  // If we got here, it didn't crash
        r.details = "encode(\"\") returned " + std::to_string(ids.size()) + " tokens";
        results.push_back(r);
    }

    // Test 12: Invalid token ID decode — must not crash on garbage IDs
    {
        ValidationResult r;
        r.name = "Invalid token ID decode safety";
        std::vector<int> bad_ids = {-1, 999999, tokenizer.totalVocabSize() + 100};
        std::string decoded = tokenizer.decode(bad_ids);
        r.passed = true;  // If we got here, it didn't crash
        r.details = "decode([-1, 999999, OOB]) handled gracefully";
        results.push_back(r);
    }

    // Print results if verbose
    if (verbose) {
        for (const auto& r : results) {
            if (r.passed) {
                fprintf(stderr, "  [PASS] %s - %s\n", r.name.c_str(), r.details.c_str());
            } else {
                fprintf(stderr, "  [FAIL] %s - %s\n", r.name.c_str(), r.details.c_str());
            }
        }
    }

    return results;
}

//======================================================//
//  CLI Parsing
//======================================================//
static RunnerOptions parseOptions(int argc, char** argv) {
    RunnerOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--vocab" && i + 1 < argc) {
            opts.vocab_path = argv[++i];
        } else if (arg == "--data" && i + 1 < argc) {
            opts.data_path = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            opts.config_path = argv[++i];
        } else if (arg == "--encode" && i + 1 < argc) {
            opts.encode_text = argv[++i];
        } else if (arg == "--verbose" || arg == "-v") {
            opts.verbose = true;
        } else if (arg == "--standalone") {
            opts.standalone = true;
        } else if (arg == "--help" || arg == "-h") {
            // Print to stderr so stdout stays clean for JSON payload
            fprintf(stderr, "GRIM Tokenizer Runner\n\n");
            fprintf(stderr, "Usage: tokenizer_runner [options]\n\n");
            fprintf(stderr, "Options:\n");
            fprintf(stderr, "  --vocab PATH      Path to vocab.bin (default: from ai_config.json)\n");
            fprintf(stderr, "  --data PATH       Path to training_data.grmt (default: from ai_config.json)\n");
            fprintf(stderr, "  --config PATH     Path to ai_config.json (default: ai_config.json)\n");
            fprintf(stderr, "  --encode TEXT     Encode text and output token IDs (skips validation)\n");
            fprintf(stderr, "  --verbose, -v     Show detailed validation output on stderr\n");
            fprintf(stderr, "  --standalone      Human-readable output (for direct invocation)\n");
            fprintf(stderr, "  --help, -h        Show this help\n");
            fprintf(stderr, "\nExit codes:\n");
            fprintf(stderr, "  0 = success (JSON payload on stdout)\n");
            fprintf(stderr, "  1 = fatal error\n");
            fprintf(stderr, "  2 = validation failure\n");
            std::exit(0);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            std::exit(1);
        }
    }
    return opts;
}

//======================================================//
//  Main Entry Point
//======================================================//
int main(int argc, char** argv) {
    try {
        auto opts = parseOptions(argc, argv);

        if (opts.standalone) {
            fprintf(stderr, "╔══════════════════════════════════════════════════╗\n");
            fprintf(stderr, "║        GRIM Tokenizer Runner (Standalone)        ║\n");
            fprintf(stderr, "╚══════════════════════════════════════════════════╝\n");
        }

        //==============================================================
        // Phase: Load Configuration
        //==============================================================
        if (opts.verbose) {
            fprintf(stderr, "[tokenizer_runner] Loading configuration...\n");
        }

        // Load paths from ai_config.json if not specified on command line
        GRIM::Config::GrimTextPaths paths;
        if (!GRIM::Config::loadGrimTextPaths(paths, opts.config_path)) {
            std::string err = "Could not load GRIM-text paths from " + opts.config_path;
            std::cout << makeErrorJson(err, "config").dump() << std::endl;
            return 1;
        }

        if (opts.vocab_path.empty()) {
            if (paths.vocab.empty()) {
                std::cout << makeErrorJson("No vocab path configured in ai_config.json", "config").dump() << std::endl;
                return 1;
            }
            opts.vocab_path = paths.vocab;
        }

        if (opts.data_path.empty()) {
            if (paths.training_data.empty()) {
                std::cout << makeErrorJson("No training_data path configured in ai_config.json", "config").dump() << std::endl;
                return 1;
            }
            opts.data_path = paths.training_data;
        }

        if (opts.verbose) {
            fprintf(stderr, "[tokenizer_runner] Vocab: %s\n", opts.vocab_path.c_str());
            fprintf(stderr, "[tokenizer_runner] Data:  %s\n", opts.data_path.c_str());
        }

        //==============================================================
        // Phase: Path Validation
        //==============================================================
        if (!fs::exists(opts.vocab_path)) {
            std::string err = "Vocabulary file does not exist: " + opts.vocab_path;
            std::cout << makeErrorJson(err, "path_validation").dump() << std::endl;
            return 1;
        }

        if (!fs::exists(opts.data_path)) {
            std::string err = "Training data file does not exist: " + opts.data_path;
            std::cout << makeErrorJson(err, "path_validation").dump() << std::endl;
            return 1;
        }

        //==============================================================
        // Phase: Load Tokenizer
        //==============================================================
        if (opts.verbose) {
            fprintf(stderr, "[tokenizer_runner] Loading tokenizer...\n");
        }

        auto load_start = std::chrono::steady_clock::now();

        // Load tokenizer config
        GRIM::Config::TokenizerConfig tok_config;
        GRIM::Config::loadTokenizerConfig(tok_config, opts.config_path);

        GRIM::Tokenizer::UniByteConfig cfg;
        if (tok_config.vocab_size > 0) {
            cfg.target_vocab_size = tok_config.vocab_size;
        }
        cfg.enable_byte_fallback = tok_config.enable_byte_fallback;

        GrimTokenizer tokenizer(cfg);
        if (!tokenizer.load(opts.vocab_path)) {
            std::string err = "Failed to load vocabulary: " + opts.vocab_path;
            std::cout << makeErrorJson(err, "tokenizer_load").dump() << std::endl;
            return 1;
        }

        auto load_end = std::chrono::steady_clock::now();
        double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

        if (opts.verbose) {
            fprintf(stderr, "[tokenizer_runner] Loaded %d tokens in %.1f ms\n",
                    tokenizer.totalVocabSize(), load_ms);
        }

        //==============================================================
        // Mode: Encode Text (early return — skips validation)
        //==============================================================
        if (!opts.encode_text.empty()) {
            auto encode_start = std::chrono::steady_clock::now();
            auto ids = tokenizer.encode(opts.encode_text);
            auto encode_end = std::chrono::steady_clock::now();
            double encode_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

            // Build per-token pieces
            json token_array = json::array();
            for (int id : ids) {
                json tok;
                tok["id"] = id;
                // Decode single token to get its surface form
                std::vector<int> single = {id};
                tok["piece"] = tokenizer.decode(single);
                // Classify token type
                if (id < 4) {
                    tok["type"] = "special";
                } else if (id >= 4 && id < 260) {
                    tok["type"] = "byte";
                } else if (id >= 260 && id < 263) {
                    tok["type"] = "atom";
                } else {
                    tok["type"] = "unigram";
                }
                token_array.push_back(tok);
            }

            // Decode full sequence for round-trip check
            std::string decoded = tokenizer.decode(ids);

            json result;
            result["status"] = "success";
            result["mode"] = "encode";
            result["input_text"] = opts.encode_text;
            result["decoded_text"] = decoded;
            result["token_count"] = static_cast<int>(ids.size());
            result["tokens"] = token_array;
            result["encode_time_ms"] = encode_ms;
            result["load_time_ms"] = load_ms;
            result["total_vocab_size"] = tokenizer.totalVocabSize();

            if (opts.standalone) {
                fprintf(stderr, "\n[ENCODE] \"%s\" → %zu tokens (%.1f ms)\n",
                        opts.encode_text.c_str(), ids.size(), encode_ms);
                for (size_t i = 0; i < ids.size(); ++i) {
                    fprintf(stderr, "  [%3zu] id=%5d  piece=\"%s\"\n",
                            i, ids[i], token_array[i]["piece"].get<std::string>().c_str());
                }
                fprintf(stderr, "  Decoded: \"%s\"\n", decoded.c_str());
            }

            std::cout << result.dump() << std::endl;
            return 0;
        }

        //==============================================================
        // Phase: Validation Checks
        //==============================================================
        if (opts.verbose) {
            fprintf(stderr, "[tokenizer_runner] Running validation checks...\n");
        }

        auto val_start = std::chrono::steady_clock::now();
        auto validation_results = runValidationChecks(tokenizer, opts.data_path, opts.verbose);
        auto val_end = std::chrono::steady_clock::now();
        double val_ms = std::chrono::duration<double, std::milli>(val_end - val_start).count();

        int passed = 0;
        int total = static_cast<int>(validation_results.size());
        std::vector<std::string> failures;
        for (const auto& r : validation_results) {
            if (r.passed) {
                passed++;
            } else {
                failures.push_back(r.name + ": " + r.details);
            }
        }

        if (opts.verbose) {
            fprintf(stderr, "[tokenizer_runner] Validation: %d/%d passed (%.1f ms)\n",
                    passed, total, val_ms);
        }

        // Check for critical failures
        if (!failures.empty()) {
            json err_json;
            err_json["status"] = "error";
            err_json["error"] = "Tokenizer validation failed";
            err_json["phase"] = "validation";
            err_json["validation_tests_passed"] = passed;
            err_json["validation_tests_total"] = total;
            json failure_list = json::array();
            for (const auto& f : failures) {
                failure_list.push_back(f);
            }
            err_json["failures"] = failure_list;
            err_json["vocab_path"] = opts.vocab_path;
            err_json["data_path"] = opts.data_path;

            if (opts.standalone) {
                fprintf(stderr, "\n[FAILED] Tokenizer validation failed:\n");
                for (const auto& f : failures) {
                    fprintf(stderr, "  - %s\n", f.c_str());
                }
            }

            std::cout << err_json.dump() << std::endl;
            return 2;
        }

        //==============================================================
        // Phase: Build Payload
        //==============================================================
        TokenizerPayload payload;
        payload.total_vocab_size = tokenizer.totalVocabSize();
        payload.unigram_vocab_size = tokenizer.vocabSize();
        payload.byte_vocab_size = 256;
        payload.atom_vocab_size = GRIM::Tokenizer::kAtomTypeCount;
        payload.special_token_count = 4;  // UNK, PAD, BOS, EOS
        payload.pad_id = tokenizer.padId();
        payload.unk_id = tokenizer.unkId();
        payload.bos_id = tokenizer.bosId();
        payload.eos_id = tokenizer.eosId();
        payload.vocab_path = opts.vocab_path;
        payload.data_path = opts.data_path;
        payload.validation_tests_passed = passed;
        payload.validation_tests_total = total;
        payload.load_time_ms = load_ms;
        payload.validation_time_ms = val_ms;

        if (opts.standalone) {
            fprintf(stderr, "\n[SUCCESS] Tokenizer validation passed (%d/%d tests)\n", passed, total);
            fprintf(stderr, "  Total vocab size: %d\n", payload.total_vocab_size);
            fprintf(stderr, "  Unigram pieces:   %d\n", payload.unigram_vocab_size);
            fprintf(stderr, "  Load time:        %.1f ms\n", payload.load_time_ms);
            fprintf(stderr, "  Validation time:  %.1f ms\n", payload.validation_time_ms);
        }

        // Output JSON payload to stdout
        std::cout << payload.toJson().dump() << std::endl;
        return 0;

    } catch (const std::exception& e) {
        json err = makeErrorJson(std::string("Fatal exception: ") + e.what(), "fatal");
        std::cout << err.dump() << std::endl;
        return 1;
    } catch (...) {
        json err = makeErrorJson("Unknown fatal exception", "fatal");
        std::cout << err.dump() << std::endl;
        return 1;
    }
}
