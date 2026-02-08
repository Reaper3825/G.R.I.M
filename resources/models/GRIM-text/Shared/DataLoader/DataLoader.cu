#include "DataLoader.hpp"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <regex>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

#include <nlohmann/json.hpp>
#include "../../Shared/UnigramByte/UniByte.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../../../control/ai_config_paths.hpp"

namespace fs = std::filesystem;

namespace GRIM {

using GRIM::Config::GrimTextPaths;

// HTML cleaning utilities
namespace {
	// Strip HTML tags from text
	std::string stripHtmlTags(const std::string& text) {
		static const std::regex html_tag_regex("<[^>]+>");
		return std::regex_replace(text, html_tag_regex, " ");
	}
	
	// Basic HTML entity decoding (common entities)
	std::string decodeHtmlEntities(const std::string& text) {
		std::string result = text;
		
		// Common entities
		const std::vector<std::pair<std::string, std::string>> entities = {
			{"&lt;", "<"}, {"&gt;", ">"}, {"&amp;", "&"}, {"&quot;", "\""}, 
			{"&apos;", "'"}, {"&nbsp;", " "}, {"&#39;", "'"}, {"&#34;", "\""},
			{"&#60;", "<"}, {"&#62;", ">"}, {"&#38;", "&"}
		};
		
		for (const auto& [entity, replacement] : entities) {
			size_t pos = 0;
			while ((pos = result.find(entity, pos)) != std::string::npos) {
				result.replace(pos, entity.length(), replacement);
				pos += replacement.length();
			}
		}
		
		return result;
	}
	
	// Remove excessive whitespace
	std::string normalizeWhitespace(const std::string& text) {
		static const std::regex whitespace_regex("\\s+");
		std::string result = std::regex_replace(text, whitespace_regex, " ");
		
		// Trim leading/trailing whitespace
		size_t start = result.find_first_not_of(" \t\n\r");
		if (start == std::string::npos) return "";
		size_t end = result.find_last_not_of(" \t\n\r");
		return result.substr(start, end - start + 1);
	}
	
	// Full cleaning pipeline
	std::string cleanText(const std::string& text) {
		std::string cleaned = stripHtmlTags(text);
		cleaned = decodeHtmlEntities(cleaned);
		// Issue #35: <s>/</s> markers stripped by stripHtmlTags (matches <[^>]+>)
		cleaned = normalizeWhitespace(cleaned);
		return cleaned;
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
			constexpr uint32_t kExpectedGrmtVersion = 5;  // GRMT v5: pre-computed targets
			uint32_t magic = 0, version = 0, num_sequences = 0, grmt_vocab_size = 0;
			grmt_file.read(reinterpret_cast<char*>(&magic), 4);
			grmt_file.read(reinterpret_cast<char*>(&version), 4);
			grmt_file.read(reinterpret_cast<char*>(&num_sequences), 4);
			grmt_file.read(reinterpret_cast<char*>(&grmt_vocab_size), 4);
			grmt_file.close();
			if (magic != 0x474D5254 || version != kExpectedGrmtVersion) {
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
								// Skip checksum (4 bytes)
								vocab_file.seekg(4, std::ios::cur);
								// Skip config vocab_size (4 bytes) and max_length (4 bytes)
								vocab_file.seekg(8, std::ios::cur);
								// Skip 3 bools (3 bytes)
								vocab_file.seekg(3, std::ios::cur);
								// Read actual vocab_size (the real number of tokens)
								uint32_t vocab_bin_size = 0;
								vocab_file.read(reinterpret_cast<char*>(&vocab_bin_size), 4);
								vocab_file.close();
								if (grmt_vocab_size != vocab_bin_size) {
									std::cout << "[DataLoader] Vocab size mismatch detected!\n"
											  << "  training_data.grmt: " << grmt_vocab_size << " tokens\n"
											  << "  vocab.bin: " << vocab_bin_size << " tokens\n"
											  << "  Forcing rebuild to synchronize..." << std::endl;
									vocab_mismatch = true;
								}
							} else {
								std::cerr << "[DataLoader] vocab.bin has unsupported version " << vocab_version << ", minimum required is 2\n";
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
					// Skip if cleaning removed all content or text is too short
					if (cleaned.length() >= 20) {
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
	const int target_vocab_size = config_tok.vocab_size;
	tok_config.target_vocab_size = target_vocab_size;
	tok_config.character_coverage = GRIM::HyperParameters::TOKENIZER_CHARACTER_COVERAGE;
	tok_config.min_subword_freq = config_tok.min_subword_freq;
	tok_config.prune_during_mining = config_tok.prune_during_mining;
	
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
		std::vector<int> targets;  // GRMT v5: pre-computed targets (shifted token_ids with masking)
		std::vector<float> numeric_values;
		std::vector<uint8_t> numeric_mask;
		std::vector<uint16_t> text_features;  // [tokens * kTextFeatureDim] FP16
		std::vector<uint8_t> text_feature_mask;  // Per-token mask
	};

	// Get BOS/EOS IDs for target masking
	const int bos_id = tokenizer.bosId();
	const int eos_id = tokenizer.eosId();

	auto encode_texts = [&](const std::vector<std::string>& corpus) -> std::vector<TokenizedSequence> {
		std::vector<TokenizedSequence> tokens;
		tokens.reserve(corpus.size());

		for (const auto& text : corpus) {
			auto result = tokenizer.encodeWithMetadata(text);
			TokenizedSequence seq;
			seq.token_ids = std::move(result.token_ids);
			seq.numeric_values = std::move(result.token_numeric_values);
			seq.numeric_mask = std::move(result.token_numeric_mask);
			seq.text_features = std::move(result.token_text_features);
			seq.text_feature_mask = std::move(result.token_text_mask);
			if (seq.numeric_values.size() != seq.token_ids.size() ||
				seq.numeric_mask.size() != seq.token_ids.size() ||
				seq.text_features.size() != seq.token_ids.size() * GRIM::Tokenizer::kTextFeatureDim ||
				seq.text_feature_mask.size() != seq.token_ids.size()) {
				throw std::runtime_error("[DataLoader] Token/numeric side-channel length mismatch");
			}
			
			// GRMT v5: Create targets (next-token prediction)
			// Target[i] = token_ids[i+1], with masking for BOS/EOS boundaries
			const size_t seq_len = seq.token_ids.size();
			seq.targets.resize(seq_len, -1);  // Default to masked
			
			for (size_t j = 0; j < seq_len; ++j) {
				if (j + 1 < seq_len) {
					int next_token = seq.token_ids[j + 1];
					// BUG FIX Issue #35: Never train to predict EOS (or BOS) mid-sequence
					// This prevents mode collapse if EOS tokens somehow appear in data
					if (next_token == eos_id || next_token == bos_id) {
						seq.targets[j] = -1;  // Mask special token predictions
					} else {
						seq.targets[j] = next_token;
					}
				}
			}
			
			tokens.push_back(std::move(seq));
		}

		return tokens;
	};

	std::cout << "[DataLoader] Encoding " << texts.size() << " sequences..." << std::endl << std::flush;
	auto all_tokens = encode_texts(texts);

	// Write single GRMT file — Phase1_Startup handles train/val splitting
	fs::create_directories(cache_dir);
	fs::path train_grmt = training_path;

	// Log sequence statistics
	size_t total_tokens = 0;
	for (const auto& seq : all_tokens) {
		total_tokens += seq.token_ids.size();
	}
	std::cout << "[DataLoader] " << all_tokens.size() << " sequences, "
			  << total_tokens << " total tokens" << std::endl;

	// Write all sequences to single GRMT file (no chunking — Phase1_Startup
	// handles sliding windows with stride and BOS prepending for long sequences)
	auto save_grmt = [&tokenizer](const fs::path& path,
		const std::vector<TokenizedSequence>& data) {
		std::ofstream file(path, std::ios::binary);
		if (!file.is_open()) return false;

		size_t sequences_with_no_valid_targets = 0;
		for (const auto& seq : data) {
			size_t valid = 0;
			for (int t : seq.targets) { if (t >= 0) valid++; }
			if (valid == 0) {
				sequences_with_no_valid_targets++;
				std::cerr << "[DataLoader] WARNING: Sequence with " << seq.token_ids.size()
						  << " tokens has 0 valid targets (all masked to -1)" << std::endl;
			}
		}
		if (sequences_with_no_valid_targets > 0) {
			std::cerr << "[DataLoader] WARNING: " << sequences_with_no_valid_targets
					  << " sequences have NO valid targets - these will be skipped during training" << std::endl;
		}

		uint32_t magic = 0x474D5254; // "GRMT"
		uint32_t version = 5;  // GRMT v5: pre-computed targets
		uint32_t num_sequences = static_cast<uint32_t>(data.size());
		// CRITICAL: Use totalVocabSize() to include byte (256) + atom (256) + unigram tokens
		uint32_t vocab_size = static_cast<uint32_t>(tokenizer.totalVocabSize());

		file.write(reinterpret_cast<const char*>(&magic), 4);
		file.write(reinterpret_cast<const char*>(&version), 4);
		file.write(reinterpret_cast<const char*>(&num_sequences), 4);
		file.write(reinterpret_cast<const char*>(&vocab_size), 4);

		for (const auto& seq : data) {
			uint32_t len = static_cast<uint32_t>(seq.token_ids.size());
			file.write(reinterpret_cast<const char*>(&len), 4);
			file.write(reinterpret_cast<const char*>(seq.token_ids.data()),
					len * sizeof(int));
			file.write(reinterpret_cast<const char*>(seq.targets.data()),
					len * sizeof(int));
			file.write(reinterpret_cast<const char*>(seq.numeric_values.data()),
					len * sizeof(float));
			file.write(reinterpret_cast<const char*>(seq.numeric_mask.data()),
					len * sizeof(uint8_t));
			file.write(reinterpret_cast<const char*>(seq.text_features.data()),
					len * GRIM::Tokenizer::kTextFeatureDim * sizeof(uint16_t));
			file.write(reinterpret_cast<const char*>(seq.text_feature_mask.data()),
					len * sizeof(uint8_t));
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

