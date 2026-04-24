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
#include <unordered_set>
#include <cstdint>
#include <exception>
#include <cmath>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>
#include "../../Common/grim_model_serialization_version.hpp"
#include "../../Shared/UnigramByte/UniByte.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../../../../control/ai_config_paths.hpp"
#include "ConceptExecutionSequenceBuilder.hpp"

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

// ─── Concept blocks corpus loading ──────────────────────────────────────────
//
// Loads concept_blocks.jsonl and returns parsed JSON objects.
// If curriculum_manifest.json exists alongside the JSONL, only entries
// whose "id" appears in the manifest's concept_block_ids are kept.
// When no manifest is present, all entries are loaded (backward compat).
//
// The canonical builder (ConceptExecutionSequenceBuilder) handles all
// structured execution record building, text rendering, and payload
// compilation — no __SLOTS__ debug path.
//
namespace {

using json = nlohmann::json;

// Curriculum filter returned by loadCurriculumFilter().
// concept_ids: blocks that get canonical Q:/STATE0/EXP:/EXEC/A: formatting.
// plaintext_ids: blocks treated as raw text (pretraining mode).
// When has_filter is false, all entries are included as concept blocks.
struct CurriculumFilter {
	std::unordered_set<std::string> concept_ids;
	std::unordered_set<std::string> plaintext_ids;
	bool has_filter = false;
	bool format_as_concept = true;  // curriculum-level flag; false → all blocks render as plain text
};

// Load curriculum filter from curriculum_registry.json by name lookup.
// Falls back to {curriculum_name}.json or curriculum_manifest.json if registry
// doesn't contain the curriculum. THROWS when a named curriculum is specified
// but cannot be found anywhere (Rule 20: no silent fallbacks).
CurriculumFilter loadCurriculumFilter(const fs::path& dir, const std::string& curriculum_name) {
	CurriculumFilter filter;

	if (curriculum_name.empty()) {
		// No curriculum specified — try legacy curriculum_manifest.json
		fs::path manifest = dir / "curriculum_manifest.json";
		if (!fs::exists(manifest)) {
			std::cout << "[DataLoader] No curriculum specified; loading all blocks unfiltered." << std::endl;
			return filter;
		}
		std::ifstream in(manifest);
		if (!in.is_open()) {
			throw std::runtime_error(
				"[DataLoader] FATAL: cannot open curriculum manifest: " + manifest.string());
		}
		json j = json::parse(in);
		if (j.contains("concept_block_ids") && j["concept_block_ids"].is_array()) {
			for (const auto& id : j["concept_block_ids"])
				if (id.is_string()) filter.concept_ids.insert(id.get<std::string>());
		}
		if (j.contains("plaintext_block_ids") && j["plaintext_block_ids"].is_array()) {
			for (const auto& id : j["plaintext_block_ids"])
				if (id.is_string()) filter.plaintext_ids.insert(id.get<std::string>());
		}
		if (j.contains("format_as_concept") && j["format_as_concept"].is_boolean())
			filter.format_as_concept = j["format_as_concept"].get<bool>();
		filter.has_filter = !filter.concept_ids.empty() || !filter.plaintext_ids.empty();
		std::cout << "[DataLoader] Legacy manifest loaded: " << manifest.string() << std::endl;
		return filter;
	}

	// ── Primary path: look up curriculum by name in curriculum_registry.json ──
	fs::path registry_path = dir / "curriculum_registry.json";
	bool found_in_registry = false;

	if (fs::exists(registry_path)) {
		std::ifstream reg_in(registry_path);
		if (reg_in.is_open()) {
			try {
				json reg = json::parse(reg_in);
				if (reg.contains("curriculums") && reg["curriculums"].is_array()) {
					for (const auto& curr : reg["curriculums"]) {
						if (!curr.contains("name") || !curr["name"].is_string()) continue;
						if (curr["name"].get<std::string>() != curriculum_name) continue;

						// Found it — extract block IDs and flags
						found_in_registry = true;
						if (curr.contains("format_as_concept") && curr["format_as_concept"].is_boolean())
							filter.format_as_concept = curr["format_as_concept"].get<bool>();

						if (curr.contains("concept_block_ids") && curr["concept_block_ids"].is_array()) {
							for (const auto& id : curr["concept_block_ids"]) {
								if (id.is_string()) {
									if (filter.format_as_concept)
										filter.concept_ids.insert(id.get<std::string>());
									else
										filter.plaintext_ids.insert(id.get<std::string>());
								}
							}
						}
						// Mixed PT/concept curriculums: registry must honor plaintext_block_ids
						// with the same semantics as the per-curriculum manifest path.
						if (curr.contains("plaintext_block_ids") && curr["plaintext_block_ids"].is_array()) {
							for (const auto& id : curr["plaintext_block_ids"]) {
								if (id.is_string())
									filter.plaintext_ids.insert(id.get<std::string>());
							}
						}
						std::cout << "[DataLoader] Curriculum '" << curriculum_name
						          << "' loaded from registry: " << registry_path.string() << std::endl;
						break;
					}
				}
			} catch (const json::exception& e) {
				std::cerr << "[DataLoader] WARNING: failed to parse curriculum_registry.json: "
				          << e.what() << std::endl;
			}
		}
	}

	// ── Fallback: per-curriculum manifest file {name}.json ──
	if (!found_in_registry) {
		fs::path manifest = dir / (curriculum_name + ".json");
		if (!fs::exists(manifest)) {
			throw std::runtime_error(
				"[DataLoader] FATAL: curriculum '" + curriculum_name
				+ "' not found in curriculum_registry.json and no manifest at: "
				+ manifest.string());
		}
		std::ifstream in(manifest);
		if (!in.is_open()) {
			throw std::runtime_error(
				"[DataLoader] FATAL: cannot open curriculum manifest: " + manifest.string());
		}
		try {
			json j = json::parse(in);
			if (j.contains("concept_block_ids") && j["concept_block_ids"].is_array()) {
				for (const auto& id : j["concept_block_ids"])
					if (id.is_string()) filter.concept_ids.insert(id.get<std::string>());
			}
			if (j.contains("plaintext_block_ids") && j["plaintext_block_ids"].is_array()) {
				for (const auto& id : j["plaintext_block_ids"])
					if (id.is_string()) filter.plaintext_ids.insert(id.get<std::string>());
			}
			if (j.contains("format_as_concept") && j["format_as_concept"].is_boolean())
				filter.format_as_concept = j["format_as_concept"].get<bool>();

			if (!filter.format_as_concept && !filter.concept_ids.empty()) {
				for (const auto& id : filter.concept_ids)
					filter.plaintext_ids.insert(id);
				filter.concept_ids.clear();
			}
			std::cout << "[DataLoader] Curriculum '" << curriculum_name
			          << "' loaded from manifest: " << manifest.string() << std::endl;
		} catch (const json::exception& e) {
			throw std::runtime_error(
				"[DataLoader] FATAL: failed to parse " + manifest.string() + ": " + e.what());
		}
	}

	filter.has_filter = !filter.concept_ids.empty() || !filter.plaintext_ids.empty();

	// For a NAMED curriculum, an empty ID set is always a configuration error.
	// Without this, an empty/missing/typo'd ID list silently expands to the
	// full corpus because loadConceptBlocksJson() only filters when has_filter
	// is true. Rule 20: fail loud rather than train on the wrong data.
	if (!curriculum_name.empty() && !filter.has_filter) {
		throw std::runtime_error(
			"[DataLoader] FATAL: curriculum '" + curriculum_name +
			"' resolved to an empty filter (concept_block_ids and plaintext_block_ids "
			"are both empty/missing). Refusing to silently train on the entire corpus. "
			"Fix the curriculum definition or remove the curriculum name to opt in to "
			"full-corpus training.");
	}

	// Also force has_filter=true for any named curriculum so that the loader
	// applies an explicit (possibly all-rejecting) filter rather than falling
	// through to the unfiltered branch.
	if (!curriculum_name.empty()) {
		filter.has_filter = true;
	}

	std::cout << "[DataLoader]   format_as_concept=" << (filter.format_as_concept ? "true" : "false")
	          << ", concept_ids=" << filter.concept_ids.size()
	          << ", plaintext_ids=" << filter.plaintext_ids.size()
	          << ", has_filter=" << (filter.has_filter ? "true" : "false")
	          << std::endl;
	return filter;
}

void loadConceptBlocksJson(const fs::path& cache_dir,
                           std::vector<json>& out,
                           CurriculumFilter& out_filter,
                           const std::string& curriculum_name = "") {
	fs::path p = cache_dir / "concept_blocks.jsonl";
	std::ifstream in(p);
	if (!in.is_open()) {
		std::cout << "[DataLoader] No concept_blocks.jsonl at " << p.string()
		          << " (optional)\n";
		return;
	}

	// Load optional curriculum filter.
	out_filter = loadCurriculumFilter(cache_dir, curriculum_name);

	std::string line;
	size_t total = 0;
	size_t accepted = 0;
	while (std::getline(in, line)) {
		if (line.empty()) continue;
		try {
			auto j = json::parse(line);
			++total;
			if (out_filter.has_filter) {
				std::string id = j.value("id", std::string());
				if (out_filter.concept_ids.find(id) == out_filter.concept_ids.end() &&
				    out_filter.plaintext_ids.find(id) == out_filter.plaintext_ids.end())
					continue;
			}
			out.push_back(std::move(j));
			++accepted;
		} catch (const std::exception& e) {
			std::cerr << "[DataLoader] concept_blocks.jsonl skip line: " << e.what() << "\n";
		}
	}
	if (out_filter.has_filter) {
		std::cout << "[DataLoader] Loaded " << accepted << "/" << total
		          << " concept blocks (filtered by curriculum manifest) from "
		          << p.string() << std::endl;
	} else {
		std::cout << "[DataLoader] Loaded " << accepted
		          << " concept block entries from " << p.string() << std::endl;
	}
}

}  // namespace

bool PrepareTrainingDataFromCache(
	const GrimTextPaths& paths,
	std::string& out_training_data_path,
	std::string& out_vocab_path,
	bool force_rebuild) {

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
			} else if (magic != 0x474D5254 || version != GRIM::GRMT_FORMAT_VERSION) {
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

	// Derive the data directory from the configured GRMT path.
	fs::path training_path(out_training_data_path);
	fs::path cache_dir = training_path.parent_path();

	std::cout << "[DataLoader] Preparing GRMT from concept blocks in: "
			  << cache_dir.string() << std::endl;

	std::vector<nlohmann::json> concept_json_entries;
	CurriculumFilter curriculum_filter;
	loadConceptBlocksJson(cache_dir, concept_json_entries, curriculum_filter, train_config.current_curriculum);

	// ── Curriculum startup summary ──
	std::cout << "[DataLoader] ═══════════ Curriculum Config ═══════════" << std::endl;
	if (!train_config.current_curriculum.empty()) {
		std::cout << "[DataLoader]   curriculum        = " << train_config.current_curriculum << std::endl;
	} else {
		std::cout << "[DataLoader]   curriculum        = (NONE — loading ALL blocks unfiltered)" << std::endl;
	}
	std::cout << "[DataLoader]   format_as_concept = " << (curriculum_filter.format_as_concept ? "true" : "false") << std::endl;
	std::cout << "[DataLoader]   has_filter        = " << (curriculum_filter.has_filter ? "true" : "false") << std::endl;
	std::cout << "[DataLoader]   concept_ids       = " << curriculum_filter.concept_ids.size() << std::endl;
	std::cout << "[DataLoader]   plaintext_ids     = " << curriculum_filter.plaintext_ids.size() << std::endl;
	std::cout << "[DataLoader]   loaded blocks     = " << concept_json_entries.size() << std::endl;
	std::cout << "[DataLoader] ═══════════════════════════════════════" << std::endl;
	if (!train_config.current_model_training.empty()) {
		std::cout << "[DataLoader] Training model: " << train_config.current_model_training << std::endl;
	}

	if (concept_json_entries.empty()) {
		std::cerr << "[DataLoader] FATAL: No concept_blocks.jsonl entries found in "
				  << cache_dir.string()
				  << "; all training data must come from curriculum concept blocks."
				  << std::endl;
		throw std::runtime_error(
			"DataLoader: concept_blocks.jsonl is required but empty or missing");
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
		std::cout << "[DataLoader] Training new tokenizer vocab from concept blocks (target: " 
				  << target_vocab_size << " tokens)..." << std::endl;
		std::vector<std::string> vocab_corpus;
		vocab_corpus.reserve(concept_json_entries.size());
		for (const auto& cj : concept_json_entries) {
			std::string entry_id = cj.value("id", std::string());
			bool is_plaintext = !curriculum_filter.format_as_concept ||
			                    (curriculum_filter.has_filter &&
			                     curriculum_filter.plaintext_ids.count(entry_id) > 0);
			if (is_plaintext)
				vocab_corpus.push_back(GRIM::DataLoader::renderPlainText(cj, false));
			else
				vocab_corpus.push_back(GRIM::DataLoader::renderCanonicalText(cj));
		}
		tokenizer.train(vocab_corpus);
		if (!out_vocab_path.empty()) {
			std::cout << "[DataLoader] Saving vocab to " << out_vocab_path << "..." << std::endl << std::flush;
			if (!tokenizer.save(out_vocab_path, save_text_vocab, config_tok.vocab_score_multiplier)) {
				// Vocab is a hard dependency for Phase1; without it the GRMT we
				// would write next is unusable. Fail immediately rather than
				// returning true and leaving training data and vocab out of sync.
				std::cerr << "[DataLoader] FATAL: failed to save vocab to "
						  << out_vocab_path << std::endl;
				return false;
			}
			std::cout << "[DataLoader] Vocab saved successfully" << std::endl << std::flush;
			if (save_text_vocab) {
				std::cout << "[DataLoader] Also saved human-readable .txt vocab" << std::endl;
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
		std::vector<int32_t> token_exec_slots;   // -1 = none; else value-slot id for bootstrap

		// Compiled structured-execution payload (GRMT v11)
		bool execution_active = false;
		std::vector<GRIM::Execution::CompiledBootstrapBinding> compiled_bootstrap_bindings;
		std::vector<GRIM::Execution::TeacherStep> teacher_steps;
		std::vector<GRIM::Execution::SlotSelectionTarget> slot_selection_targets;
	};

	// BOS/EOS are NOT added here — Phase1_Startup owns boundary token
	// insertion (add_bos, add_eos config flags) and target fixup for them.

	auto build_sequence = [&](const std::string& text) -> std::optional<TokenizedSequence> {
		auto result = tokenizer.encodeWithMetadata(text);
		if (result.token_ids.empty()) {
			return std::nullopt;
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

		const size_t seq_len = seq.token_ids.size();
		seq.targets.resize(seq_len, -1);
		for (size_t j = 0; j + 1 < seq_len; ++j) {
			seq.targets[j] = seq.token_ids[j + 1];
		}
		seq.token_exec_slots.assign(seq_len, -1);
		return seq;
	};

	std::cout << "[DataLoader] Encoding " << concept_json_entries.size()
	          << " concept sequences..." << std::endl << std::flush;
	std::vector<TokenizedSequence> all_tokens;
	all_tokens.reserve(concept_json_entries.size());
	int concept_exec_base_slot = 0;
	if (const char* ev = std::getenv("GRIM_CONCEPT_EXEC_BASE_SLOT")) {
		try {
			concept_exec_base_slot = std::stoi(ev);
		} catch (...) {}
	}
	const int expected_exec_steps = train_config.execution_block_num_steps;
	size_t plaintext_count = 0;
	size_t concept_build_failures = 0;
	size_t selected_entries_skipped = 0;  // short text / encoder returned nullopt
	for (const auto& cj : concept_json_entries) {
		try {
			std::string entry_id = cj.value("id", std::string());
			bool is_plaintext = !curriculum_filter.format_as_concept ||
			                    (curriculum_filter.has_filter &&
			                     curriculum_filter.plaintext_ids.count(entry_id) > 0);

			if (is_plaintext) {
				// ── Pretraining path: plain text, no execution payload ──
				std::string text = GRIM::DataLoader::renderPlainText(cj, false);
				if (text.size() < kMinCleanedTextLength) { ++selected_entries_skipped; continue; }

				auto seq = build_sequence(text);
				if (!seq) { ++selected_entries_skipped; continue; }
				seq->execution_active = false;
				all_tokens.push_back(std::move(*seq));
				++plaintext_count;
				continue;
			}

			// ── Concept path: canonical formatting + execution payload ──
			auto built = GRIM::DataLoader::buildConceptSequence(cj, tokenizer, concept_exec_base_slot);
			if (built.canonical_text.size() < kMinCleanedTextLength) { ++selected_entries_skipped; continue; }

			if (built.payload.execution_active) {
				const int actual_steps = static_cast<int>(built.payload.teacher_steps.size());
				if (actual_steps == 0) {
					throw std::runtime_error(
						"DataLoader: execution-active concept entry has 0 teacher_steps "
						"— cannot pad from nothing");
				}
				if (actual_steps > expected_exec_steps) {
					std::string exec_entry_id = "(unknown)";
					if (cj.contains("id") && cj["id"].is_string())
						exec_entry_id = cj["id"].get<std::string>();
					else if (cj.contains("name") && cj["name"].is_string())
						exec_entry_id = cj["name"].get<std::string>();
					throw std::runtime_error(
						"DataLoader: execution-active concept entry \"" + exec_entry_id
						+ "\" has teacher_steps=" + std::to_string(actual_steps)
						+ " > execution_block_num_steps=" + std::to_string(expected_exec_steps)
						+ " — truncation would lose computation; fix data or increase config num_steps");
				}
				// Padding deferred to buildBatchPayload where step_mask is constructed.
				// GRMT stores original step count; batch builder pads + masks.
			}

			auto seq = build_sequence(built.canonical_text);
			if (!seq) { ++selected_entries_skipped; continue; }
			seq->execution_active = built.payload.execution_active;
			if (built.payload.execution_active) {
				seq->token_exec_slots = std::move(built.payload.token_exec_slots);
				seq->compiled_bootstrap_bindings = std::move(built.payload.compiled_bootstrap_bindings);
				seq->teacher_steps = std::move(built.payload.teacher_steps);
				seq->slot_selection_targets = std::move(built.payload.slot_selection_targets);
			}
			all_tokens.push_back(std::move(*seq));
		} catch (const std::exception& e) {
			++concept_build_failures;
			std::cerr << "[DataLoader] concept build failed: " << e.what() << "\n";
		}
	}
	if (plaintext_count > 0) {
		std::cout << "[DataLoader] Encoded " << plaintext_count << " plaintext (PT) + "
		          << (all_tokens.size() - plaintext_count) << " concept sequences" << std::endl;
	}

	// Refuse to write a zero-sequence GRMT — every selected entry failed or was
	// skipped, so there is nothing to train on. Better to fail here than to
	// return true and have Phase1 silently load an empty dataset.
	if (all_tokens.empty()) {
		std::cerr << "[DataLoader] FATAL: no sequences produced from "
		          << concept_json_entries.size() << " selected entries ("
		          << concept_build_failures << " build failures). "
		          << "Cannot write a zero-sequence GRMT." << std::endl;
		return false;
	}

	// For a filtered (named) curriculum, every selected entry was hand-picked
	// by config; an unexpected build failure on any of them is a data/config
	// bug, not noise to be swallowed. Fail loud so it gets fixed at the
	// source instead of producing a quietly-degraded GRMT. Silent skips
	// (short text, empty encoder output) count too — the curriculum names
	// the entries it expects to train on, so dropping any of them silently
	// is a partial GRMT.
	if (curriculum_filter.has_filter &&
	    (concept_build_failures > 0 || selected_entries_skipped > 0)) {
		std::cerr << "[DataLoader] FATAL: " << concept_build_failures
		          << " build failure(s) and " << selected_entries_skipped
		          << " silently-skipped selected entry/entries under a filtered "
		          << "curriculum. Refusing to produce a partial GRMT."
		          << std::endl;
		return false;
	}

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

		// Refuse to write a header with num_sequences=0. Without this, every
		// sequence being dropped here (e.g. all targets masked) still produces
		// a header-valid GRMT and a "successful" return.
		if (valid_seq_count == 0) {
			std::cerr << "[DataLoader] FATAL: save_grmt would write num_sequences=0 "
			          << "(all " << data.size() << " candidate sequences were dropped). "
			          << "Refusing to emit an empty GRMT." << std::endl;
			return false;
		}

		uint32_t magic = 0x474D5254; // "GRMT"
		uint32_t version = GRIM::GRMT_FORMAT_VERSION;
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
			// GRMT v11: compiled structured-execution payload
			// Order follows plan: exec_active, token_exec_slots, then bindings/steps/targets
			uint8_t exec_active = seq.execution_active ? 1 : 0;
			file.write(reinterpret_cast<const char*>(&exec_active), sizeof(uint8_t));

			std::vector<int32_t> slots = seq.token_exec_slots;
			if (slots.size() != len) {
				// Slot map MUST be aligned with the token stream. Silently rewriting
				// to all -1 while still emitting compiled_bootstrap_bindings and
				// teacher_steps produces a GRMT whose payload no longer agrees with
				// itself — downstream validation later flags it as broken with no
				// pointer to the real cause. Fail at the source instead.
				throw std::runtime_error(
					"[DataLoader] save_grmt: token_exec_slots.size()=" +
					std::to_string(seq.token_exec_slots.size()) +
					" != token_ids.size()=" + std::to_string(len) +
					(seq.execution_active
						? " (execution_active row \u2014 cannot recover)"
						: " (execution_inactive row \u2014 still a builder bug)"));
			}
			file.write(reinterpret_cast<const char*>(slots.data()), len * sizeof(int32_t));

			// Compiled bootstrap bindings
			uint32_t cbb_count = static_cast<uint32_t>(seq.compiled_bootstrap_bindings.size());
			file.write(reinterpret_cast<const char*>(&cbb_count), sizeof(uint32_t));
			static_assert(sizeof(GRIM::Execution::CompiledBootstrapBinding) == 12,
				"CompiledBootstrapBinding must be 12 bytes for bulk GRMT serialization");
			if (cbb_count > 0) {
				file.write(reinterpret_cast<const char*>(seq.compiled_bootstrap_bindings.data()),
					cbb_count * sizeof(GRIM::Execution::CompiledBootstrapBinding));
			}

			// Teacher steps
			uint32_t ts_count = static_cast<uint32_t>(seq.teacher_steps.size());
			file.write(reinterpret_cast<const char*>(&ts_count), sizeof(uint32_t));
			static_assert(sizeof(GRIM::Execution::TeacherStep) == 20,
				"TeacherStep must be 20 bytes for bulk GRMT serialization");
			if (ts_count > 0) {
				file.write(reinterpret_cast<const char*>(seq.teacher_steps.data()),
					ts_count * sizeof(GRIM::Execution::TeacherStep));
			}

			// Slot selection targets (field-by-field due to struct padding)
			uint32_t sst_count = static_cast<uint32_t>(seq.slot_selection_targets.size());
			file.write(reinterpret_cast<const char*>(&sst_count), sizeof(uint32_t));
			for (uint32_t si = 0; si < sst_count; ++si) {
				uint8_t kind = static_cast<uint8_t>(seq.slot_selection_targets[si].kind);
				file.write(reinterpret_cast<const char*>(&kind), sizeof(uint8_t));
				file.write(reinterpret_cast<const char*>(&seq.slot_selection_targets[si].slot_id),
					sizeof(int32_t));
			}
		}
		return file.good();
	};

	// Atomic write: serialize to a sibling temp path, then rename to the final
	// path only after success. Without this, any mid-write failure (slot-map
	// throw, disk error, etc.) leaves a header-valid but truncated GRMT at
	// `train_grmt`, and the freshness check on the next run accepts it.
	fs::path tmp_grmt = train_grmt;
	tmp_grmt += ".tmp";
	std::error_code ec;
	fs::remove(tmp_grmt, ec);  // best-effort cleanup of any prior crash residue

	bool wrote_ok = false;
	try {
		wrote_ok = save_grmt(tmp_grmt, all_tokens);
	} catch (const std::exception& e) {
		std::cerr << "[DataLoader] FATAL: exception while writing GRMT: " << e.what() << std::endl;
		wrote_ok = false;
	}

	if (!wrote_ok) {
		fs::remove(tmp_grmt, ec);  // never leave a partial temp file behind
		std::cerr << "[DataLoader] Failed to write GRMT file." << std::endl;
		return false;
	}

	// Atomically replace the destination. fs::rename overwrites on POSIX and
	// is atomic-on-same-filesystem on Windows for files (ReplaceFileW path).
	fs::rename(tmp_grmt, train_grmt, ec);
	if (ec) {
		fs::remove(tmp_grmt, ec);
		std::cerr << "[DataLoader] Failed to rename temp GRMT into place: "
		          << ec.message() << std::endl;
		return false;
	}

	std::cout << "[DataLoader] Wrote " << all_tokens.size() << " sequences to "
			  << train_grmt.string() << std::endl;

	return true;
}

} // namespace GRIM

