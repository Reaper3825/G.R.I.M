#include "DataLoader.hpp"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <optional>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <exception>

#include <nlohmann/json.hpp>
#include "../../Shared/UnigramByte/UniByte.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../../../control/ai_config_paths.hpp"

namespace fs = std::filesystem;

namespace GRIM {

using GRIM::Config::GrimTextPaths;

// Minimum cleaned text length to include in training data.
// Shorter texts lack sufficient context for meaningful next-token prediction.
constexpr size_t kMinCleanedTextLength = 20;

// HTML cleaning utilities — state-machine implementations (no std::regex)
namespace {
	// Strip HTML tags from text using a simple scanner.
	// Matches <TAG_CONTENT> where TAG_CONTENT is 1+ chars (same semantics as
	// the old regex "<[^>]+>"). Lone '<' without closing '>' is preserved,
	// preventing content loss on mathematical expressions like "a < b".
	std::string stripHtmlTags(const std::string& text) {
		std::string result;
		result.reserve(text.size());
		size_t i = 0;
		while (i < text.size()) {
			if (text[i] == '<') {
				size_t close = text.find('>', i + 1);
				if (close != std::string::npos && close > i + 1) {
					// Valid tag: replace with space so words don't merge
					result += ' ';
					i = close + 1;
				} else {
					// No closing '>' or empty <> — preserve the '<'
					result += text[i];
					++i;
				}
			} else {
				result += text[i];
				++i;
			}
		}
		return result;
	}

	// Decode HTML entities in a single left-to-right pass.
	//
	// This inherently prevents double-decoding: once "&amp;" is decoded to "&",
	// the cursor advances past the semicolon, so the resulting "&" is never
	// re-interpreted as the start of another entity like "&lt;".
	//
	// Supports:
	//   - Named entities: &lt; &gt; &amp; &quot; &apos; &nbsp;
	//   - Decimal numeric:  &#NNN;   (e.g. &#8217; → right single quote)
	//   - Hex numeric:      &#xHHHH; (e.g. &#x2019; → right single quote)
	//   - Full UTF-8 encoding for codepoints up to U+10FFFF
	std::string decodeHtmlEntities(const std::string& text) {
		std::string result;
		result.reserve(text.size());
		size_t i = 0;
		while (i < text.size()) {
			if (text[i] == '&') {
				// Look for closing semicolon within a reasonable distance
				size_t semi = text.find(';', i + 1);
				if (semi != std::string::npos && semi - i < 12) {
					const size_t elen = semi - i + 1;

					// Named entities
					if (text.compare(i, 4, "&lt;") == 0 && elen == 4)
						{ result += '<'; i = semi + 1; continue; }
					if (text.compare(i, 4, "&gt;") == 0 && elen == 4)
						{ result += '>'; i = semi + 1; continue; }
					if (text.compare(i, 6, "&quot;") == 0 && elen == 6)
						{ result += '"'; i = semi + 1; continue; }
					if (text.compare(i, 6, "&apos;") == 0 && elen == 6)
						{ result += '\''; i = semi + 1; continue; }
					if (text.compare(i, 6, "&nbsp;") == 0 && elen == 6)
						{ result += ' '; i = semi + 1; continue; }
					if (text.compare(i, 5, "&amp;") == 0 && elen == 5)
						{ result += '&'; i = semi + 1; continue; }

					// Numeric entities: &#NNN; or &#xHHHH;
					if (elen >= 4 && text[i + 1] == '#') {
						unsigned long codepoint = 0;
						bool valid = false;
						try {
							if (text[i + 2] == 'x' || text[i + 2] == 'X') {
								codepoint = std::stoul(text.substr(i + 3, semi - i - 3), nullptr, 16);
							} else {
								codepoint = std::stoul(text.substr(i + 2, semi - i - 2), nullptr, 10);
							}
							valid = (codepoint > 0 && codepoint < 0x110000);
						} catch (...) {}

						if (valid) {
							// Encode as UTF-8
							if (codepoint < 0x80) {
								result += static_cast<char>(codepoint);
							} else if (codepoint < 0x800) {
								result += static_cast<char>(0xC0 | (codepoint >> 6));
								result += static_cast<char>(0x80 | (codepoint & 0x3F));
							} else if (codepoint < 0x10000) {
								result += static_cast<char>(0xE0 | (codepoint >> 12));
								result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
								result += static_cast<char>(0x80 | (codepoint & 0x3F));
							} else {
								result += static_cast<char>(0xF0 | (codepoint >> 18));
								result += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
								result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
								result += static_cast<char>(0x80 | (codepoint & 0x3F));
							}
							i = semi + 1;
							continue;
						}
					}
				}
			}
			// Not a recognized entity or no semicolon — emit '&' literally
			result += text[i];
			++i;
		}
		return result;
	}

	// Collapse runs of whitespace to a single space and trim edges.
	// State-machine scan — no regex overhead.
	std::string normalizeWhitespace(const std::string& text) {
		std::string result;
		result.reserve(text.size());
		bool prev_ws = true; // Treat start as whitespace → trims leading
		for (char c : text) {
			if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
				if (!prev_ws) result += ' ';
				prev_ws = true;
			} else {
				result += c;
				prev_ws = false;
			}
		}
		// Trim trailing space
		if (!result.empty() && result.back() == ' ') {
			result.pop_back();
		}
		return result;
	}

	// Full cleaning pipeline
	std::string cleanText(const std::string& text) {
		std::string cleaned = stripHtmlTags(text);
		cleaned = decodeHtmlEntities(cleaned);
		// Issue #35: <s>/</s> markers stripped by stripHtmlTags (matches <X>)
		cleaned = normalizeWhitespace(cleaned);
		return cleaned;
	}

	unsigned int resolveTokenizerWorkerCount(
		const GRIM::Config::TokenizerConfig& config_tok,
		size_t corpus_size) {
		if (corpus_size == 0) return 1;
		if (!config_tok.enable_parallel_tokenization) return 1;

		const size_t threshold = static_cast<size_t>(
			std::max(1, config_tok.parallel_threshold));
		if (corpus_size < threshold) return 1;

		unsigned int workers = std::thread::hardware_concurrency();
		if (workers == 0) workers = 4;
		workers = std::max(1u, workers > 1 ? workers - 1 : 1);

		if (const char* env_workers = std::getenv("GRIM_TOKENIZER_WORKERS")) {
			try {
				const int parsed = std::stoi(env_workers);
				if (parsed > 0) {
					workers = static_cast<unsigned int>(parsed);
				}
			} catch (...) {
				// Ignore malformed env override and keep auto worker count.
			}
		}

		workers = std::min<unsigned int>(workers, static_cast<unsigned int>(corpus_size));
		return std::max(1u, workers);
	}

	size_t resolveTokenizerChunkSize(unsigned int workers, size_t corpus_size) {
		if (workers <= 1 || corpus_size == 0) return corpus_size;

		// Aim for multiple chunks per worker for load balancing while
		// keeping chunks large enough to amortize scheduler overhead.
		size_t chunk = (corpus_size + (workers * 8) - 1) / (workers * 8);
		chunk = std::max<size_t>(16, chunk);
		chunk = std::min<size_t>(512, chunk);
		return chunk;
	}
}

bool PrepareTrainingDataFromCache(
	const GrimTextPaths& paths,
	std::string& out_training_data_path,
	std::string& out_vocab_path,
	bool force_rebuild,
	bool clear_cache) {

	// Resolve primary paths from config
	if (!paths.training_data.empty()) {
		out_training_data_path = paths.training_data;
	}
	if (!paths.vocab.empty()) {
		out_vocab_path = paths.vocab;
	}

	if (out_training_data_path.empty()) {
		std::cerr << "[DataLoader] No training_data path configured; skipping cache preparation." << std::endl;
		return false;
	}

	GRIM::Config::TrainingHyperparameters train_config;
	if (!GRIM::Config::loadTrainingHyperparameters(train_config)) {
		std::cerr << "[DataLoader] FATAL: Could not load training config for atom layout" << std::endl;
		throw std::runtime_error("DataLoader: training config missing");
	}
	GRIM::Tokenizer::configureTokenLayout(GRIM::Tokenizer::kAtomTypeCount);
	std::cout << "[DataLoader] Atom token range fixed at " << GRIM::Tokenizer::ATOM_VOCAB_SIZE
	          << " type tokens" << std::endl;

	const bool training_exists = fs::exists(out_training_data_path);
	const bool vocab_exists = !out_vocab_path.empty() && fs::exists(out_vocab_path);
	fs::path vocab_path(out_vocab_path);
	bool grmt_version_mismatch = false;

	// Check if vocab and training data are in sync (always check, regardless of force_rebuild)
	bool vocab_mismatch = false;
	if (training_exists && vocab_exists) {
		// Read vocab size from GRMT file
		std::ifstream grmt_file(out_training_data_path, std::ios::binary);
		if (grmt_file.is_open()) {
	constexpr uint32_t kExpectedGrmtVersion = 8;  // GRMT v8: atom_flags side channel
			uint32_t magic = 0, version = 0, num_sequences = 0, grmt_vocab_size = 0;
			grmt_file.read(reinterpret_cast<char*>(&magic), 4);
			grmt_file.read(reinterpret_cast<char*>(&version), 4);
			grmt_file.read(reinterpret_cast<char*>(&num_sequences), 4);
			grmt_file.read(reinterpret_cast<char*>(&grmt_vocab_size), 4);
			grmt_file.close();
			if (!grmt_file) {
				// Truncated or corrupted GRMT file — force rebuild
				std::cerr << "[DataLoader] GRMT header read failed (truncated file?); forcing rebuild." << std::endl;
				grmt_version_mismatch = true;
			} else if (magic != 0x474D5254 || version != kExpectedGrmtVersion) {
				grmt_version_mismatch = true;
			}

			if (magic == 0x474D5254) { // "GRMT" in little-endian
				// Read vocab size from vocab.bin
				std::ifstream vocab_file(out_vocab_path, std::ios::binary);
				if (vocab_file.is_open()) {
					char vocab_magic[5] = {0};
					vocab_file.read(vocab_magic, 4);
					
						if (std::string(vocab_magic) == "KTMG") {
							uint16_t vocab_version = 0;
							vocab_file.read(reinterpret_cast<char*>(&vocab_version), 2);
							
							// Version 2+ required (Rule 20: no backwards compatibility)
							if (vocab_version >= 2) {
								// Vocab v2 header layout after magic(4) + version(2):
								//   checksum(4) + config_vocab_size(4) + max_length(4) + 3 bools(3) + actual_vocab_size(4)
								constexpr std::streamoff kVocabV2SkipBytes = 4 + 4 + 4 + 3; // 15 bytes to skip
								vocab_file.seekg(kVocabV2SkipBytes, std::ios::cur);
								uint32_t vocab_bin_size = 0;
								vocab_file.read(reinterpret_cast<char*>(&vocab_bin_size), 4);
								vocab_file.close();
								if (!vocab_file) {
									std::cerr << "[DataLoader] vocab.bin header read failed (truncated?); forcing rebuild." << std::endl;
									vocab_mismatch = true;
								} else if (grmt_vocab_size != vocab_bin_size) {
									std::cout << "[DataLoader] Vocab size mismatch detected!\n"
											  << "  training_data.grmt: " << grmt_vocab_size << " tokens\n"
											  << "  vocab.bin: " << vocab_bin_size << " tokens\n"
											  << "  Forcing rebuild to synchronize..." << std::endl;
									vocab_mismatch = true;
								}
							} else {
								std::cerr << "[DataLoader] vocab.bin has unsupported version " << vocab_version << ", minimum required is 2\n";
								vocab_mismatch = true; // Force rebuild — stale vocab format
							}
						}
				}
			}
		}
	}

	// Skip rebuild only if all required artifacts exist, match, and force_rebuild is false
	if (!force_rebuild && !vocab_mismatch && !grmt_version_mismatch && training_exists && vocab_exists) {
		std::cout << "[DataLoader] Existing training data + vocab found at '" << out_training_data_path
			  << "', vocab sizes match, skipping cache rebuild." << std::endl;
		return true;
	}

	// Log reason for rebuild
	if (force_rebuild) {
		std::cout << "[DataLoader] force_rebuild=true, rebuilding training data and vocab..." << std::endl;
	} else if (vocab_mismatch) {
		std::cout << "[DataLoader] Rebuilding due to vocab size mismatch..." << std::endl;
	} else if (grmt_version_mismatch) {
		std::cout << "[DataLoader] Rebuilding due to GRMT version mismatch..." << std::endl;
	} else if (training_exists && !vocab_exists) {
		std::cout << "[DataLoader] Training data present but vocab missing; rebuilding." << std::endl;
	} else if (!training_exists && vocab_exists) {
		std::cout << "[DataLoader] Vocab present but training data missing; rebuilding." << std::endl;
	} else {
		std::cout << "[DataLoader] Both files missing; building from cache..." << std::endl;
	}

	// Determine where the merged cache should live. We expect
	// grim_text.training_data in ai_config.json to point to the
	// desired final GRMT file; its parent directory is where
	// grim_data_pipeline wrote merged_verified_cache.jsonl.
	fs::path training_path(out_training_data_path);
	fs::path cache_dir = training_path.parent_path();
	fs::path cache_path = cache_dir / "merged_verified_cache.jsonl";

	if (!fs::exists(cache_path)) {
		std::cout << "[DataLoader] No merged_verified_cache.jsonl found at '"
				  << cache_path.string() << "'; nothing to prepare." << std::endl;
		return false;
	}

	std::cout << "[DataLoader] Preparing GRMT from cache: "
			  << cache_path.string() << std::endl;

	// Load cleaned texts from cache
	std::vector<std::string> texts;
	size_t malformed_lines = 0;
	{
		std::ifstream in(cache_path);
		if (!in.is_open()) {
			std::cerr << "[DataLoader] Failed to open cache file: "
					  << cache_path.string() << std::endl;
			return false;
		}
		std::string line;
		while (std::getline(in, line)) {
			if (line.empty()) continue;
			try {
				auto j = nlohmann::json::parse(line);
				if (j.contains("content")) {
					std::string raw_text = j["content"].get<std::string>();
					std::string cleaned = cleanText(raw_text);
					if (cleaned.length() >= kMinCleanedTextLength) {
						texts.push_back(cleaned);
					}
				}
			} catch (const std::exception&) {
				++malformed_lines;
				continue;
			}
		}
	}

	if (malformed_lines > 0) {
		std::cerr << "[DataLoader] Skipped " << malformed_lines 
				  << " malformed JSONL lines in cache file" << std::endl;
	}

	if (texts.empty()) {
		std::cout << "[DataLoader] Cache file is empty; nothing to tokenize." << std::endl;
		return false;
	}

	// No train/val/test split here — Phase1_Startup owns that decision.  
	// DataLoader writes ALL sequences to a single GRMT file.

	// Load tokenizer config from ai_config.json
	GRIM::Tokenizer::UniByteConfig tok_config;
	GRIM::Config::TokenizerConfig config_tok;
	if (!GRIM::Config::loadTokenizerConfig(config_tok)) {
		std::cerr << "[DataLoader] FATAL: ai_config.json missing; tokenizer config required" << std::endl;
		throw std::runtime_error("DataLoader: tokenizer config missing");
	}
	if (config_tok.vocab_size <= 0) {
		std::cerr << "[DataLoader] FATAL: tokenizer vocab_size must be > 0" << std::endl;
		throw std::runtime_error("DataLoader: invalid tokenizer vocab_size");
	}
	int target_vocab_size = config_tok.vocab_size;
	if (config_tok.max_vocab_size > 0 && target_vocab_size > config_tok.max_vocab_size) {
		std::cout << "[DataLoader] Clamping tokenizer target vocab_size "
				  << target_vocab_size << " -> " << config_tok.max_vocab_size
				  << " (max_vocab_size)" << std::endl;
		target_vocab_size = config_tok.max_vocab_size;
	}
	tok_config.target_vocab_size = target_vocab_size;
	tok_config.character_coverage = GRIM::HyperParameters::TOKENIZER_CHARACTER_COVERAGE;
	tok_config.min_subword_freq = config_tok.min_subword_freq;
	tok_config.prune_during_mining = config_tok.prune_during_mining;
	tok_config.enable_parallel_subword_mining = config_tok.enable_parallel_subword_mining;
	tok_config.subword_mining_workers = config_tok.subword_mining_workers;
	tok_config.subword_mining_max_bytes = config_tok.subword_mining_max_bytes;
	
	// Load scratch block reasoning settings from training hyperparameters (single source of truth)
	tok_config.enable_scratch_block_reasoning = train_config.tokenizer_enable_scratch_block_reasoning;
	tok_config.detect_numbers = train_config.tokenizer_detect_numbers;
	tok_config.detect_urls = train_config.tokenizer_detect_urls;
	tok_config.detect_emails = train_config.tokenizer_detect_emails;
	tok_config.detect_paths = train_config.tokenizer_detect_paths;
	tok_config.detect_dates = train_config.tokenizer_detect_dates;
	tok_config.detect_code_literals = train_config.tokenizer_detect_code_literals;
	
	tok_config.enable_byte_fallback = config_tok.enable_byte_fallback;
	tok_config.prefer_gpu = true;

	GRIM::Tokenizer::UniByte tokenizer(tok_config);

	// Resolve vocab path from config
	fs::create_directories(vocab_path.parent_path());
	bool vocab_loaded = false;
	bool save_text_vocab = false;
	
	// Check if we should save human-readable text vocab
	save_text_vocab = config_tok.save_text_vocab;
	
	if (!force_rebuild && !out_vocab_path.empty() && fs::exists(vocab_path)) {
		if (tokenizer.load(out_vocab_path)) {
			std::cout << "[DataLoader] Loaded existing vocab from "
					  << out_vocab_path << std::endl;
			vocab_loaded = true;
		}
	}
	if (!vocab_loaded) {
		std::cout << "[DataLoader] Training new tokenizer vocab from cache (target: " 
				  << target_vocab_size << " tokens)..." << std::endl;
		tokenizer.train(texts);
		if (!out_vocab_path.empty()) {
			std::cout << "[DataLoader] Saving vocab to " << out_vocab_path << "..." << std::endl << std::flush;
			if (!tokenizer.save(out_vocab_path, save_text_vocab)) {
				std::cerr << "[DataLoader] Failed to save vocab to "
						  << out_vocab_path << std::endl;
			} else {
				std::cout << "[DataLoader] Vocab saved successfully" << std::endl << std::flush;
				if (save_text_vocab) {
					std::cout << "[DataLoader] Also saved human-readable .txt vocab" << std::endl;
				}
			}
		}
	}

	struct TokenizedSequence {
		std::vector<int> token_ids;
		std::vector<int> targets;  // GRMT v7: pre-computed targets (shifted token_ids with masking)
		std::vector<float> numeric_values;
		std::vector<uint32_t> atom_flags;     // Per-token type-specific flags from AtomTable
		std::vector<uint16_t> text_features;  // [tokens * kTextFeatureDim] FP16
		std::vector<uint8_t> atom_mask;       // Unified per-token atom mask
		std::shared_ptr<GRIM::Tokenizer::AtomTable> atom_table;  // Per-sequence atom registry
		std::vector<uint32_t> atom_entry_ids;  // Per-token index into atom_table
	};

	// BOS/EOS are NOT added here — Phase1_Startup owns boundary token
	// insertion (add_bos, add_eos config flags) and target fixup for them.

	auto encode_texts = [&](const std::vector<std::string>& corpus) -> std::vector<TokenizedSequence> {
		std::vector<TokenizedSequence> tokens;
		tokens.reserve(corpus.size());

		auto build_sequence = [&](const std::string& text) -> std::optional<TokenizedSequence> {
			auto result = tokenizer.encodeWithMetadata(text);
			if (result.token_ids.empty()) {
				return std::nullopt; // Skip 0-length tokenizations
			}

			TokenizedSequence seq;
			seq.token_ids = std::move(result.token_ids);
			seq.numeric_values = std::move(result.token_numeric_values);
			seq.atom_flags = std::move(result.token_atom_flags);
			seq.text_features = std::move(result.token_text_features);
			seq.atom_mask = std::move(result.token_atom_mask);
			seq.atom_table = std::move(result.atom_table);
			seq.atom_entry_ids = std::move(result.atom_entry_ids);
			if (seq.numeric_values.size() != seq.token_ids.size() ||
				seq.atom_flags.size() != seq.token_ids.size() ||
				seq.text_features.size() != seq.token_ids.size() * GRIM::Tokenizer::kTextFeatureDim ||
				seq.atom_mask.size() != seq.token_ids.size() ||
				seq.atom_entry_ids.size() != seq.token_ids.size()) {
				throw std::runtime_error("[DataLoader] Token/side-channel length mismatch");
			}

			// GRMT v6: Pure next-token prediction targets.
			// target[j] = token_ids[j+1], last position = -1 (no next token).
			// Phase1_Startup handles BOS/EOS insertion and target fixup around
			// those boundary tokens.
			const size_t seq_len = seq.token_ids.size();
			seq.targets.resize(seq_len, -1); // Last position default: no next token
			for (size_t j = 0; j + 1 < seq_len; ++j) {
				seq.targets[j] = seq.token_ids[j + 1];
			}
			return seq;
		};

		const unsigned int workers = resolveTokenizerWorkerCount(config_tok, corpus.size());
		if (workers <= 1) {
			for (const auto& text : corpus) {
				auto seq = build_sequence(text);
				if (seq) {
					tokens.push_back(std::move(*seq));
				}
			}
			return tokens;
		}

		std::cout << "[DataLoader] Parallel tokenization: workers=" << workers
				  << " threshold=" << std::max(1, config_tok.parallel_threshold)
				  << " sequences=" << corpus.size() << std::endl;

		std::vector<std::unique_ptr<TokenizedSequence>> staged(corpus.size());

		const size_t chunk_size = resolveTokenizerChunkSize(workers, corpus.size());
		std::atomic<size_t> next_index{0};
		std::atomic<bool> abort{false};
		std::exception_ptr first_error = nullptr;
		std::mutex error_mutex;

		auto worker_fn = [&]() {
			while (!abort.load(std::memory_order_relaxed)) {
				const size_t begin = next_index.fetch_add(chunk_size, std::memory_order_relaxed);
				if (begin >= corpus.size()) break;
				const size_t end = std::min(begin + chunk_size, corpus.size());

				for (size_t i = begin; i < end; ++i) {
					if (abort.load(std::memory_order_relaxed)) return;
					try {
						auto seq = build_sequence(corpus[i]);
						if (seq) {
							staged[i] = std::make_unique<TokenizedSequence>(std::move(*seq));
						}
					} catch (...) {
						abort.store(true, std::memory_order_relaxed);
						std::lock_guard<std::mutex> lock(error_mutex);
						if (!first_error) {
							first_error = std::current_exception();
						}
						return;
					}
				}
			}
		};

		std::vector<std::thread> pool;
		pool.reserve(workers);
		for (unsigned int w = 0; w < workers; ++w) {
			pool.emplace_back(worker_fn);
		}
		for (auto& thread : pool) {
			thread.join();
		}

		if (first_error) {
			std::rethrow_exception(first_error);
		}

		// Preserve original sequence order for deterministic GRMT output.
		for (size_t i = 0; i < corpus.size(); ++i) {
			if (!staged[i]) continue;
			tokens.push_back(std::move(*staged[i]));
		}

		return tokens;
	};

	std::cout << "[DataLoader] Encoding " << texts.size() << " sequences..." << std::endl << std::flush;
	auto all_tokens = encode_texts(texts);

	// Write single GRMT file — Phase1_Startup handles train/val splitting
	fs::create_directories(cache_dir);
	fs::path train_grmt = training_path;

	// Log sequence statistics + atom diagnostics
	size_t total_tokens = 0;
	size_t encode_atom_tokens = 0;
	size_t encode_atom_sequences = 0;
	size_t encode_atom_entries = 0;
	for (const auto& seq : all_tokens) {
		total_tokens += seq.token_ids.size();
		bool seq_has_atoms = false;
		for (size_t j = 0; j < seq.token_ids.size(); ++j) {
			if (j < seq.atom_mask.size() && seq.atom_mask[j]) {
				encode_atom_tokens++;
				seq_has_atoms = true;
			}
			if (j < seq.atom_entry_ids.size() &&
				seq.atom_entry_ids[j] != GRIM::Tokenizer::kAtomEntryNone) {
				encode_atom_entries++;
			}
		}
		if (seq_has_atoms) encode_atom_sequences++;
	}
	std::cout << "[DataLoader] " << all_tokens.size() << " sequences, "
			  << total_tokens << " total tokens" << std::endl;
	std::cout << "[DataLoader] Atom encoding stats: "
			  << encode_atom_tokens << " atom tokens ("
			  << (total_tokens > 0 ? (100.0 * encode_atom_tokens / total_tokens) : 0.0)
			  << "%), " << encode_atom_sequences << "/" << all_tokens.size()
			  << " sequences with atoms, " << encode_atom_entries
			  << " AtomTable entries" << std::endl;
	if (encode_atom_tokens == 0) {
		std::cerr << "[DataLoader] WARNING: Zero atoms detected during encoding! "
				  << "Check scratch_block_reasoning.enabled in ai_config.json" << std::endl;
	}

	// Write all sequences to single GRMT file (no chunking — Phase1_Startup
	// handles sliding windows with stride and BOS prepending for long sequences)
	auto save_grmt = [&tokenizer](const fs::path& path,
		const std::vector<TokenizedSequence>& data) {
		std::ofstream file(path, std::ios::binary);
		if (!file.is_open()) return false;

		// Pre-scan: count valid sequences and warn about degenerate ones.
		// Skip 0-length sequences entirely — they cause division-by-zero
		// in loss reduction (valid_count=0).
		size_t valid_seq_count = 0;
		size_t sequences_with_no_valid_targets = 0;
		for (const auto& seq : data) {
			if (seq.token_ids.empty()) continue; // Will be skipped during write
			size_t valid = 0;
			for (int t : seq.targets) { if (t >= 0) valid++; }
			if (valid == 0) {
				sequences_with_no_valid_targets++;
				std::cerr << "[DataLoader] WARNING: Sequence with " << seq.token_ids.size()
						  << " tokens has 0 valid targets (all masked to -1) — skipping" << std::endl;
				continue; // Will be skipped during write
			}
			valid_seq_count++;
		}
		if (sequences_with_no_valid_targets > 0) {
			std::cerr << "[DataLoader] Dropped " << sequences_with_no_valid_targets
					  << " sequences with 0 valid targets" << std::endl;
		}

		uint32_t magic = 0x474D5254; // "GRMT"
		uint32_t version = 8;  // GRMT v8: atom_flags side channel
		uint32_t num_sequences = static_cast<uint32_t>(valid_seq_count);
		// CRITICAL: Use totalVocabSize() to include byte (256) + atom (256) + unigram tokens
		uint32_t vocab_size = static_cast<uint32_t>(tokenizer.totalVocabSize());

		file.write(reinterpret_cast<const char*>(&magic), 4);
		file.write(reinterpret_cast<const char*>(&version), 4);
		file.write(reinterpret_cast<const char*>(&num_sequences), 4);
		file.write(reinterpret_cast<const char*>(&vocab_size), 4);

		for (const auto& seq : data) {
			// Skip empty and target-less sequences
			if (seq.token_ids.empty()) continue;
			{ bool has_valid = false;
			  for (int t : seq.targets) { if (t >= 0) { has_valid = true; break; } }
			  if (!has_valid) continue;
			}

			uint32_t len = static_cast<uint32_t>(seq.token_ids.size());
			file.write(reinterpret_cast<const char*>(&len), 4);
			file.write(reinterpret_cast<const char*>(seq.token_ids.data()),
					len * sizeof(int));
			file.write(reinterpret_cast<const char*>(seq.targets.data()),
					len * sizeof(int));
			file.write(reinterpret_cast<const char*>(seq.numeric_values.data()),
					len * sizeof(float));
			file.write(reinterpret_cast<const char*>(seq.atom_mask.data()),
					len * sizeof(uint8_t));
			file.write(reinterpret_cast<const char*>(seq.text_features.data()),
					len * GRIM::Tokenizer::kTextFeatureDim * sizeof(uint16_t));
			// GRMT v8: atom_flags (type-specific metadata from AtomTable)
			file.write(reinterpret_cast<const char*>(seq.atom_flags.data()),
					len * sizeof(uint32_t));
			// GRMT v6: atom text (length-prefixed strings per token, reconstructed from AtomTable)
			for (uint32_t j = 0; j < len; ++j) {
				std::string s;
				if (seq.atom_table &&
					seq.atom_entry_ids[j] != GRIM::Tokenizer::kAtomEntryNone) {
					const auto* entry = seq.atom_table->getAtom(seq.atom_entry_ids[j]);
					if (entry) { s = seq.atom_table->atomToString(*entry); }
				}
				uint16_t slen = static_cast<uint16_t>(std::min<size_t>(s.size(), 65535));
				file.write(reinterpret_cast<const char*>(&slen), sizeof(uint16_t));
				if (slen > 0) {
					file.write(s.data(), slen);
				}
			}
		}
		return file.good();
	};

	if (!save_grmt(train_grmt, all_tokens)) {
		std::cerr << "[DataLoader] Failed to write GRMT file." << std::endl;
		return false;
	}

	std::cout << "[DataLoader] Wrote " << all_tokens.size() << " sequences to "
			  << train_grmt.string() << std::endl;

	// Optionally clear the cache now that it has been consumed.
	if (clear_cache) {
		std::error_code ec;
		fs::remove(cache_path, ec);
		if (ec) {
			std::cerr << "[DataLoader] Warning: failed to remove cache file: "
					  << cache_path.string() << " (" << ec.message() << ")" << std::endl;
		}
	}

	return true;
}

} // namespace GRIM

