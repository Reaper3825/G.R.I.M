#include "verifier.hpp"
#include "web_training_data_generated.h"
#include <nlohmann/json.hpp>
#include <flatbuffers/flatbuffers.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <regex>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <optional>
#include <array>
#include <mutex>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {
constexpr const char* kDefaultSemanticModelRelPath =
    "resources/models/GRIM-text/quality/deberta-v3-base-mnli.onnx";
constexpr const char* kDefaultSemanticTokenizerRelPath =
    "resources/models/GRIM-text/quality/deberta-v3-base-mnli.spm";

struct SemanticEncodedInput {
    std::vector<int64_t> ids;
    std::vector<int64_t> attention;
    std::vector<int64_t> type_ids;
    bool valid = false;
};

class SemanticQualityModel {
public:
    explicit SemanticQualityModel(const Config& config)
        : max_seq_length_(std::max(64, config.semantic_max_seq_length)),
          use_gpu_(config.semantic_use_gpu),
          verbose_(config.verbose_logging) {
        positive_prompts_ = config.semantic_positive_prompts;
        if (positive_prompts_.empty()) {
            positive_prompts_ = {
                "This passage is factual, high quality, and well-written.",
                "This passage would improve a language model's knowledge and reasoning."
            };
        }

        negative_prompts_ = config.semantic_negative_prompts;
        if (negative_prompts_.empty()) {
            negative_prompts_ = {
                "This passage is spam, repetitive, or low effort content.",
                "This passage contains misinformation, hate, or unsafe instructions."
            };
        }

        initialized_ = initialize(config);
    }

    bool ready() const { return initialized_; }

    float scoreQuality(const std::string& text) const {
        if (!initialized_ || text.empty()) {
            return -1.0f;
        }

        float positive_sum = 0.0f;
        size_t positive_count = 0;
        for (const auto& prompt : positive_prompts_) {
            auto probs = infer(text, prompt);
            if (!probs) continue;
            positive_sum += (*probs)[2];  // Entailment probability
            positive_count++;
        }
        if (positive_count == 0) {
            return -1.0f;
        }
        float positive_avg = positive_sum / static_cast<float>(positive_count);

        float negative_sum = 0.0f;
        size_t negative_count = 0;
        for (const auto& prompt : negative_prompts_) {
            auto probs = infer(text, prompt);
            if (!probs) continue;
            negative_sum += (*probs)[2];
            negative_count++;
        }
        float negative_avg = negative_count > 0 ? (negative_sum / static_cast<float>(negative_count)) : 0.0f;

        float score = positive_avg * (1.0f - negative_avg);
        return std::clamp(score, 0.0f, 1.0f);
    }

private:
    bool initialize(const Config& config) {
        if (!loadTokenizer(config.semantic_tokenizer_path)) {
            std::cerr << "[Verifier] Semantic tokenizer unavailable, disabling semantic filter.\n";
            return false;
        }
        if (!loadSession(config.semantic_model_path)) {
            std::cerr << "[Verifier] Semantic ONNX model unavailable, disabling semantic filter.\n";
            return false;
        }
        return true;
    }

    bool loadTokenizer(const std::string& path) {
        if (path.empty()) {
            return false;
        }
        if (!fs::exists(path)) {
            std::cerr << "[Verifier] Semantic tokenizer not found at: " << path << "\n";
            return false;
        }

        auto status = tokenizer_.Load(path);
        if (!status.ok()) {
            std::cerr << "[Verifier] Failed to load tokenizer: " << status.ToString() << "\n";
            return false;
        }

        cls_id_ = tokenizer_.PieceToId("[CLS]");
        sep_id_ = tokenizer_.PieceToId("[SEP]");
        pad_id_ = tokenizer_.PieceToId("[PAD]");
        unk_id_ = tokenizer_.unk_id();

        if (cls_id_ < 0) cls_id_ = tokenizer_.bos_id();
        if (sep_id_ < 0) sep_id_ = tokenizer_.eos_id();
        if (pad_id_ < 0) pad_id_ = tokenizer_.pad_id();
        if (unk_id_ < 0) unk_id_ = 0;
        if (cls_id_ < 0) cls_id_ = unk_id_;
        if (sep_id_ < 0) sep_id_ = unk_id_;
        if (pad_id_ < 0) pad_id_ = 0;

        tokenizer_ready_ = true;
        return true;
    }

    bool loadSession(const std::string& path) {
        if (path.empty()) return false;
        if (!fs::exists(path)) {
            std::cerr << "[Verifier] Semantic ONNX not found at: " << path << "\n";
            return false;
        }

        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SemanticVerifier");
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options_->SetIntraOpNumThreads(2);

        if (use_gpu_) {
            try {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                session_options_->AppendExecutionProvider_CUDA(cuda_options);
                using_cuda_ = true;
                if (verbose_) {
                    std::cout << "[Verifier] Semantic filter using CUDA provider.\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "[Verifier] CUDA provider unavailable: " << e.what() << "\n";
            }
        }

        std::wstring model_path_w(path.begin(), path.end());
        session_ = std::make_unique<Ort::Session>(*env_, model_path_w.c_str(), *session_options_);

        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session_->GetInputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(name.get());
        }
        size_t num_outputs = session_->GetOutputCount();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(name.get());
        }

        return true;
    }

    SemanticEncodedInput encodePair(const std::string& premise, const std::string& hypothesis) const {
        SemanticEncodedInput encoded;
        if (!tokenizer_ready_) return encoded;

        std::vector<int> premise_tokens;
        std::vector<int> hypothesis_tokens;
        if (!tokenizer_.Encode(premise, &premise_tokens).ok()) return encoded;
        if (!tokenizer_.Encode(hypothesis, &hypothesis_tokens).ok()) return encoded;

        const size_t usable_tokens = max_seq_length_ > 3 ? (max_seq_length_ - 3) : 0;
        if (usable_tokens == 0) return encoded;

        size_t max_premise_tokens = usable_tokens > 32 ? (usable_tokens - 32) : (usable_tokens - 1);
        if (max_premise_tokens < 1) max_premise_tokens = usable_tokens / 2;
        size_t premise_len = std::min(premise_tokens.size(), max_premise_tokens);
        size_t remaining = usable_tokens - premise_len;
        if (remaining == 0) {
            remaining = 1;
            if (premise_len > 0) premise_len -= 1;
        }
        size_t hypothesis_len = std::min(hypothesis_tokens.size(), remaining);

        encoded.ids.reserve(max_seq_length_);
        encoded.attention.reserve(max_seq_length_);
        encoded.type_ids.reserve(max_seq_length_);

        auto push_token = [&](int64_t id, int64_t type, int64_t attn) {
            encoded.ids.push_back(id);
            encoded.type_ids.push_back(type);
            encoded.attention.push_back(attn);
        };

        push_token(cls_id_, 0, 1);
        for (size_t i = 0; i < premise_len; ++i) {
            push_token(static_cast<int64_t>(premise_tokens[i]), 0, 1);
        }
        push_token(sep_id_, 0, 1);
        for (size_t i = 0; i < hypothesis_len; ++i) {
            push_token(static_cast<int64_t>(hypothesis_tokens[i]), 1, 1);
        }
        push_token(sep_id_, 1, 1);

        while (encoded.ids.size() < static_cast<size_t>(max_seq_length_)) {
            push_token(pad_id_, 0, 0);
        }

        encoded.valid = encoded.ids.size() == static_cast<size_t>(max_seq_length_);
        return encoded;
    }

    static std::array<float, 3> softmax(const std::array<float, 3>& logits) {
        float max_logit = *std::max_element(logits.begin(), logits.end());
        std::array<float, 3> exp_vals{};
        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            exp_vals[i] = std::exp(logits[i] - max_logit);
            sum += exp_vals[i];
        }
        for (float& value : exp_vals) {
            value /= (sum + 1e-6f);
        }
        return exp_vals;
    }

    std::optional<std::array<float, 3>> infer(const std::string& premise,
                                              const std::string& hypothesis) const {
        if (!session_) {
            return std::nullopt;
        }

        auto encoded = encodePair(premise, hypothesis);
        if (!encoded.valid) {
            return std::nullopt;
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(max_seq_length_)};

        std::vector<Ort::Value> inputs;
        inputs.reserve(3);
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info,
            encoded.ids.data(),
            encoded.ids.size(),
            input_shape.data(),
            input_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info,
            encoded.attention.data(),
            encoded.attention.size(),
            input_shape.data(),
            input_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info,
            encoded.type_ids.data(),
            encoded.type_ids.size(),
            input_shape.data(),
            input_shape.size()));

        std::vector<const char*> input_name_ptrs;
        input_name_ptrs.reserve(input_names_.size());
        for (const auto& name : input_names_) {
            input_name_ptrs.push_back(name.c_str());
        }

        std::vector<const char*> output_name_ptrs;
        output_name_ptrs.reserve(output_names_.size());
        for (const auto& name : output_names_) {
            output_name_ptrs.push_back(name.c_str());
        }

        std::lock_guard<std::mutex> lock(run_mutex_);
        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                     input_name_ptrs.data(),
                                     inputs.data(),
                                     inputs.size(),
                                     output_name_ptrs.data(),
                                     output_name_ptrs.size());

        if (outputs.empty()) {
            return std::nullopt;
        }

        auto* logits = outputs[0].GetTensorMutableData<float>();
        std::array<float, 3> raw_logits{logits[0], logits[1], logits[2]};
        return softmax(raw_logits);
    }

    int max_seq_length_;
    bool tokenizer_ready_ = false;
    bool initialized_ = false;
    bool use_gpu_ = true;
    bool using_cuda_ = false;
    bool verbose_ = false;
    sentencepiece::SentencePieceProcessor tokenizer_;
    int cls_id_ = -1;
    int sep_id_ = -1;
    int pad_id_ = -1;
    int unk_id_ = 0;
    std::vector<std::string> positive_prompts_;
    std::vector<std::string> negative_prompts_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    mutable std::mutex run_mutex_;
};
} // namespace

class Verifier::Impl {
public:
    Config config;
    mutable Stats stats;
    mutable std::unordered_set<size_t> content_hashes;  // For duplicate detection
    std::unique_ptr<SemanticQualityModel> semantic_model_;

    explicit Impl(const Config& cfg) : config(cfg) {
        if (config.source_type_weights.empty()) {
            // Comprehensive source type weights for all our sources
            config.source_type_weights = {
                // Core academic/scholarly
                {"academic", 1.0f},
                {"academic_papers", 1.0f},
                {"philosophy", 0.95f},
                {"classical_texts", 0.95f},
                
                // Technical/Educational
                {"technical", 0.9f},
                {"tech_docs", 0.9f},
                {"open_books", 0.9f},
                {"erudite_writing", 0.9f},
                
                // Language/Linguistics
                {"linguistics", 0.9f},
                {"grammar", 0.9f},
                {"rhetoric", 0.9f},
                {"speech_corpus", 0.85f},
                
                // Logic/Reasoning
                {"logic", 0.95f},
                {"theoretical_reasoning", 0.95f},
                {"theoretical_science", 0.95f},
                
                // Reference/Data
                {"wikipedia", 0.8f},
                {"github", 0.85f},
                {"gutenberg", 0.9f},
                {"jstor_oa", 1.0f},
                
                // Hardware/Technical specs
                {"hardware_specs", 0.85f},
                
                // Other common types
                {"arxiv", 1.0f},
                {"stackoverflow", 0.75f},
                {"news_api", 0.7f},
                {"reddit", 0.6f},
                
                // Default for unknown
                {"unknown", 0.7f}
            };
        }

        if (config.enable_semantic_model) {
            auto ensure_path = [this](std::string& target, const std::string& fallback) {
                if (!target.empty()) {
                    auto resolved = resolveSemanticAsset(target);
                    if (!resolved.empty()) {
                        target = resolved;
                        return;
                    }
                }
                auto fallback_path = resolveSemanticAsset(fallback);
                if (!fallback_path.empty()) {
                    target = fallback_path;
                }
            };

            ensure_path(config.semantic_model_path, kDefaultSemanticModelRelPath);
            ensure_path(config.semantic_tokenizer_path, kDefaultSemanticTokenizerRelPath);

            semantic_model_ = std::make_unique<SemanticQualityModel>(config);
            if (!semantic_model_ || !semantic_model_->ready()) {
                semantic_model_.reset();
                std::cerr << "[Verifier] Semantic verifier disabled (model initialization failed)." << std::endl;
            } else if (config.verbose_logging) {
                std::cout << "[Verifier] Semantic verifier online using model: "
                          << config.semantic_model_path << std::endl;
            }
        }
    }

    size_t hash_content(const std::string& content) const {
        // Normalize and hash first 500 chars for fuzzy duplicate detection
        std::string normalized;
        normalized.reserve(std::min(content.length(), size_t(500)));
        
        for (char c : content) {
            if (std::isalnum(c) || std::isspace(c)) {
                normalized += std::tolower(c);
            }
            if (normalized.length() >= 500) break;
        }
        
        return std::hash<std::string>{}(normalized);
    }

    bool is_likely_english(const std::string& text) const {
        if (text.length() < 50) return true;  // Too short to judge
        
        // Count common English words
        static const std::vector<std::string> common_words = {
            " the ", " is ", " at ", " which ", " on ", " and ", " are ", " for ",
            " was ", " with ", " this ", " that ", " from ", " have ", " has "
        };
        
        std::string lower = " " + text.substr(0, 500) + " ";
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        
        int matches = 0;
        for (const auto& word : common_words) {
            if (lower.find(word) != std::string::npos) matches++;
        }
        
        // Need at least 4 common English words in first 500 chars
        return matches >= 4;
    }

    bool is_low_quality_content(const std::string& content) const {
        std::string lower = content;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        
        // Check for spam patterns
        static const std::vector<std::string> spam_patterns = {
            "buy now", "click here", "limited time", "act now", "offer expires",
            "subscribe now", "sign up today", "free trial"
        };
        
        for (const auto& pattern : spam_patterns) {
            if (lower.find(pattern) != std::string::npos) return true;
        }
        
        // Check for excessive repetition (same char 10+ times)
        char prev = '\0';
        int repeat_count = 0;
        for (char c : content) {
            if (c == prev) {
                repeat_count++;
                if (repeat_count >= 10) return true;
            } else {
                prev = c;
                repeat_count = 1;
            }
        }
        
        // Check for UI/navigation artifacts
        static const std::vector<std::string> ui_patterns = {
            "enable javascript", "404", "page not found", "error 404",
            "home | about | contact", "search...", "loading...",
            "menu", "login", "sign in", "register", "forgot password",
            "copyright ©", "all rights reserved", "terms of service"
        };
        
        for (const auto& pattern : ui_patterns) {
            if (lower.find(pattern) != std::string::npos) return true;
        }
        
        // Check for encoding errors (common malformed patterns)
        if (content.find("â€™") != std::string::npos ||  // Encoding of '
            content.find("â€œ") != std::string::npos ||  // Encoding of "
            content.find("Ã©") != std::string::npos) {   // Encoding of é
            return true;
        }
        
        return false;
    }

    float assess_content_quality(const std::string& content) const {
        float score = 0.5f;
        
        // Check for complete sentences (periods, question marks, exclamation)
        size_t sentence_endings = std::count(content.begin(), content.end(), '.') +
                                 std::count(content.begin(), content.end(), '?') +
                                 std::count(content.begin(), content.end(), '!');
        
        // Expect roughly 1 sentence per 80-120 characters
        float expected_sentences = content.length() / 100.0f;
        float sentence_ratio = std::min(sentence_endings / std::max(expected_sentences, 1.0f), 2.0f);
        
        if (sentence_ratio > 0.5f && sentence_ratio < 1.5f) {
            score += 0.15f;  // Good sentence structure
        }
        
        // Check for proper capitalization
        bool has_uppercase = std::any_of(content.begin(), content.end(), ::isupper);
        bool has_lowercase = std::any_of(content.begin(), content.end(), ::islower);
        
        if (has_uppercase && has_lowercase) {
            score += 0.1f;  // Mixed case (natural text)
        }
        
        // Penalize excessive special characters
        size_t special_chars = std::count_if(content.begin(), content.end(), 
            [](char c) { return !std::isalnum(c) && !std::isspace(c) && c != '.' && c != ',' && c != '!' && c != '?'; });
        
        float special_ratio = static_cast<float>(special_chars) / content.length();
        if (special_ratio > 0.15f) {
            score -= 0.2f;  // Too many special chars
        } else if (special_ratio < 0.05f) {
            score += 0.1f;  // Clean text
        }
        
        // Check for code blocks (good for technical content)
        if (content.find("```") != std::string::npos || 
            content.find("    ") != std::string::npos) {  // 4-space indentation
            score += 0.05f;
        }
        
        // Penalize excessive URLs
        size_t url_count = 0;
        size_t pos = 0;
        while ((pos = content.find("http", pos)) != std::string::npos) {
            url_count++;
            pos += 4;
        }
        
        if (url_count > 5) {
            score -= 0.15f;  // Probably scraped link list
        }
        
        return std::clamp(score, 0.0f, 1.0f);
    }

    std::string extract_domain(const std::string& url) const {
        std::regex domain_regex(R"((?:https?://)?(?:www\.)?([^/]+))");
        std::smatch match;
        if (std::regex_search(url, match, domain_regex) && match.size() > 1) {
            return match[1].str();
        }
        return "";
    }

    bool is_domain_approved(const std::string& domain) const {
        if (config.domain_whitelist.empty()) return true;
        return std::find(config.domain_whitelist.begin(), 
                        config.domain_whitelist.end(), 
                        domain) != config.domain_whitelist.end();
    }

    float calculate_reliability(const UnverifiedEntry& entry) const {
        float base_score = 0.7f;  // Default score for unknown types
        
        auto it = config.source_type_weights.find(entry.source_type);
        if (it != config.source_type_weights.end()) {
            base_score = it->second;
        }
        
        // Progressive filtering - more forgiving approach
        if (config.progressive_filtering) {
            // Apply length penalties more gradually
            if (entry.content.length() < config.min_length) {
                float ratio = static_cast<float>(entry.content.length()) / config.min_length;
                base_score *= (0.5f + 0.5f * ratio);  // 50-100% of score based on how close
            } else if (entry.content.length() > config.max_length) {
                // Only slight penalty for very long content
                base_score *= 0.9f;
            }
            
            // Bonus for good content indicators
            if (entry.content.find("http") != std::string::npos) {
                base_score *= 0.95f;  // Small penalty for URLs (might be noise)
            }
            
            // Word count check (more forgiving)
            size_t word_count = std::count(entry.content.begin(), entry.content.end(), ' ') + 1;
            if (word_count < 10) {
                base_score *= 0.6f;  // Short entries get lower score
            } else if (word_count > 100) {
                base_score *= 1.05f;  // Bonus for substantial content
            }
        } else {
            // Original stricter logic
            if (entry.content.length() < config.min_length) {
                base_score *= 0.7f;
            } else if (entry.content.length() > config.max_length) {
                base_score *= 0.8f;
            }
        }
        
        return std::min(1.0f, std::max(0.0f, base_score));
    }

    VerifiedEntry verify_entry(const UnverifiedEntry& entry) const {
        VerifiedEntry verified;
        verified.content = entry.content;
        verified.source_url = entry.source_url;
        verified.source_type = entry.source_type;
        verified.author = entry.author;
        verified.metadata = entry.metadata;
        verified.reliability_score = calculate_reliability(entry);
        verified.verification_time = std::time(nullptr);
        return verified;
    }

    std::string resolveSemanticAsset(const std::string& relative) const {
        if (relative.empty()) return "";

        fs::path candidate(relative);
        if (candidate.is_absolute()) {
            return fs::exists(candidate) ? candidate.string() : "";
        }

        fs::path root = fs::current_path();
        for (int i = 0; i < 6 && root.has_parent_path(); ++i) {
            if (fs::exists(root / "resources")) break;
            root = root.parent_path();
        }

        candidate = root / relative;
        if (fs::exists(candidate)) {
            return candidate.string();
        }
        return "";
    }

    float semantic_quality_score(const std::string& content) const {
        if (!semantic_model_) return -1.0f;
        return semantic_model_->scoreQuality(content);
    }
};

Verifier::Verifier(const Config& config) 
    : pImpl(std::make_unique<Impl>(config)) {}

Verifier::~Verifier() = default;

std::vector<UnverifiedEntry> Verifier::load_unverified_entries() const {
    std::vector<UnverifiedEntry> entries;
    
    if (!fs::exists(pImpl->config.input_dir)) {
        std::cerr << "Input directory does not exist: " << pImpl->config.input_dir << std::endl;
        return entries;
    }
    
    for (const auto& entry : fs::directory_iterator(pImpl->config.input_dir)) {
        if (entry.path().extension() == ".jsonl") {
            std::ifstream file(entry.path());
            std::string line;
            while (std::getline(file, line)) {
                try {
                    auto j = json::parse(line);
                    UnverifiedEntry unverified;
                    unverified.content = j.value("content", "");
                    unverified.source_url = j.value("source_url", "unknown");
                    unverified.source_type = j.value("source_type", "unknown");
                    unverified.author = j.value("author", "unknown");
                    
                    // Handle metadata (could be string or object)
                    if (j.contains("metadata")) {
                        if (j["metadata"].is_string()) {
                            unverified.metadata = j["metadata"].get<std::string>();
                        } else {
                            unverified.metadata = j["metadata"].dump();
                        }
                    } else {
                        unverified.metadata = "";
                    }
                    
                    // Only add if content exists
                    if (!unverified.content.empty()) {
                        entries.push_back(unverified);
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing JSON: " << e.what() << std::endl;
                }
            }
        }
    }
    
    return entries;
}

std::vector<VerifiedEntry> Verifier::verify_entries(const std::vector<UnverifiedEntry>& entries) const {
    std::vector<VerifiedEntry> verified;
    std::vector<VerifiedEntry> rejected;  // For optional saving
    
    for (const auto& entry : entries) {
        pImpl->stats.total_processed++;
        
        std::string rejection_reason;
        bool should_accept = true;
        
        // Check 1: Duplicate detection (hash-based)
        size_t content_hash = pImpl->hash_content(entry.content);
        if (pImpl->content_hashes.count(content_hash)) {
            pImpl->stats.duplicate_rejected++;
            rejection_reason = "duplicate_content";
            should_accept = false;
            
            if (pImpl->config.verbose_logging) {
                std::cerr << "REJECTED (duplicate): hash=" << content_hash << "\n";
            }
        } else {
            pImpl->content_hashes.insert(content_hash);
        }
        
        // Check 2: Language detection
        if (should_accept && !pImpl->is_likely_english(entry.content)) {
            pImpl->stats.quality_rejected++;
            rejection_reason = "non_english_content";
            should_accept = false;
            
            if (pImpl->config.verbose_logging) {
                std::cerr << "REJECTED (language): not English\n";
            }
        }
        
        // Check 3: Low-quality content filter
        if (should_accept && pImpl->is_low_quality_content(entry.content)) {
            pImpl->stats.quality_rejected++;
            rejection_reason = "low_quality_patterns";
            should_accept = false;
            
            if (pImpl->config.verbose_logging) {
                std::cerr << "REJECTED (quality): spam/UI/malformed content detected\n";
            }
        }
        
        // Check 4: Domain validation (if whitelist exists)
        if (should_accept) {
            std::string domain = pImpl->extract_domain(entry.source_url);
            if (!pImpl->is_domain_approved(domain)) {
                pImpl->stats.domain_rejected++;
                rejection_reason = "domain_not_approved";
                should_accept = false;
                
                if (pImpl->config.verbose_logging) {
                    std::cerr << "REJECTED (domain): " << domain << "\n";
                }
            }
        }
        
        // Check 5: Length validation (with more forgiving bounds)
        if (should_accept) {
            if (entry.content.length() < pImpl->config.min_length) {
                pImpl->stats.quality_rejected++;
                rejection_reason = "too_short";
                should_accept = false;
                
                if (pImpl->config.verbose_logging) {
                    std::cerr << "REJECTED (too short): " << entry.content.length() 
                              << " < " << pImpl->config.min_length << "\n";
                }
            } else if (entry.content.length() > pImpl->config.max_length) {
                // Don't reject - just truncate or note it
                if (pImpl->config.progressive_filtering) {
                    // Allow but note in metadata
                    if (pImpl->config.verbose_logging) {
                        std::cout << "WARNING (long): " << entry.content.length() 
                                  << " > " << pImpl->config.max_length << " (accepting anyway)\n";
                    }
                } else {
                    pImpl->stats.quality_rejected++;
                    rejection_reason = "too_long";
                    should_accept = false;
                }
            }
        }

        float semantic_score = -1.0f;
        if (should_accept && pImpl->config.enable_semantic_model && pImpl->semantic_model_) {
            semantic_score = pImpl->semantic_quality_score(entry.content);
            if (semantic_score >= 0.0f && pImpl->config.semantic_hard_filter &&
                semantic_score < pImpl->config.semantic_min_score) {
                pImpl->stats.semantic_rejected++;
                pImpl->stats.quality_rejected++;
                rejection_reason = "semantic_low_confidence";
                should_accept = false;

                if (pImpl->config.verbose_logging) {
                    std::cerr << "REJECTED (semantic): score=" << semantic_score
                              << " < " << pImpl->config.semantic_min_score << "\n";
                }
            }
        }

        // Check 3: Reliability score
        if (should_accept) {
            VerifiedEntry ve = pImpl->verify_entry(entry);

            // Apply content quality assessment
            float content_quality = pImpl->assess_content_quality(entry.content);
            ve.reliability_score = (ve.reliability_score * 0.7f) + (content_quality * 0.3f);  // Weighted average

            if (semantic_score >= 0.0f && pImpl->config.semantic_quality_weight > 0.0f) {
                float weight = std::clamp(pImpl->config.semantic_quality_weight, 0.05f, 0.75f);
                ve.reliability_score = (ve.reliability_score * (1.0f - weight)) + (semantic_score * weight);
            }

            if (ve.reliability_score >= pImpl->config.reliability_threshold) {
                // Classify by quality tier
                if (ve.reliability_score >= pImpl->config.high_quality_threshold) {
                    pImpl->stats.high_quality_count++;
                } else if (ve.reliability_score >= pImpl->config.medium_quality_threshold) {
                    pImpl->stats.medium_quality_count++;
                } else {
                    pImpl->stats.low_quality_count++;
                }
                
                verified.push_back(ve);
                pImpl->stats.passed_verification++;
            } else {
                pImpl->stats.failed_verification++;
                rejection_reason = "low_reliability_score";
                should_accept = false;
                
                if (pImpl->config.verbose_logging) {
                    std::cerr << "REJECTED (reliability): " << ve.reliability_score 
                              << " < " << pImpl->config.reliability_threshold << "\n";
                }
                
                if (pImpl->config.save_rejected) {
                    rejected.push_back(ve);
                }
            }
        }
        
        // Track rejection reason
        if (!should_accept && !rejection_reason.empty()) {
            pImpl->stats.rejection_reasons[rejection_reason]++;
        }
    }
    
    // Optionally save rejected entries for analysis
    if (pImpl->config.save_rejected && !rejected.empty()) {
        std::string rejected_path = pImpl->config.output_dir + "/rejected";
        std::filesystem::create_directories(rejected_path);
        
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
        
        std::string filename = rejected_path + "/rejected_" + ss.str() + ".jsonl";
        std::ofstream file(filename);
        
        for (const auto& entry : rejected) {
            json j;
            j["content"] = entry.content.substr(0, 500);  // Truncate for storage
            j["source_url"] = entry.source_url;
            j["source_type"] = entry.source_type;
            j["reliability_score"] = entry.reliability_score;
            file << j.dump() << std::endl;
        }
        file.close();
        
        std::cout << "Saved " << rejected.size() << " rejected entries to " << filename << "\n";
    }
    
    return verified;
}

bool Verifier::save_verified_entries(const std::vector<VerifiedEntry>& entries) const {
    if (!fs::exists(pImpl->config.output_dir)) {
        fs::create_directories(pImpl->config.output_dir);
    }
    
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    
    std::string filename = pImpl->config.output_dir + "/verified_" + ss.str() + ".jsonl";
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return false;
    }
    
    for (const auto& entry : entries) {
        json j;
        j["content"] = entry.content;
        j["source_url"] = entry.source_url;
        j["source_type"] = entry.source_type;
        j["author"] = entry.author;
        j["metadata"] = entry.metadata;
        j["reliability_score"] = entry.reliability_score;
        j["verification_time"] = entry.verification_time;
        
        file << j.dump() << std::endl;
    }
    
    file.close();
    std::cout << "Saved " << entries.size() << " verified entries to " << filename << std::endl;
    return true;
}

Stats Verifier::get_stats() const {
    return pImpl->stats;
}

void Stats::writeSummaryToLog(const std::string& log_path) const {
    namespace fs = std::filesystem;
    
    // Ensure logs directory exists
    fs::path logFile(log_path);
    if (logFile.has_parent_path()) {
        fs::create_directories(logFile.parent_path());
    }
    
    // Write human-readable log file
    std::ofstream log(log_path, std::ios::app);
    if (log.is_open()) {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        
        log << "\n=== Verification Statistics ===\n";
        log << "Timestamp: " << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << "\n";
        log << "Total Processed:    " << total_processed << "\n";
        log << "Passed:             " << passed_verification << " ("
            << (total_processed > 0 ? (passed_verification * 100.0 / total_processed) : 0)
            << "%)\n";
        log << "Failed:             " << failed_verification << "\n";
        log << "  - Domain:         " << domain_rejected << "\n";
        log << "  - Quality:        " << quality_rejected << "\n";
        log << "  - Duplicates:     " << duplicate_rejected << "\n";
        log << "  - Semantic:       " << semantic_rejected << "\n";
        
        if (high_quality_count + medium_quality_count + low_quality_count > 0) {
            log << "\nQuality Distribution:\n";
            log << "  - High:           " << high_quality_count << "\n";
            log << "  - Medium:         " << medium_quality_count << "\n";
            log << "  - Low:            " << low_quality_count << "\n";
        }
        
        if (!rejection_reasons.empty()) {
            log << "\nRejection Reasons:\n";
            for (const auto& [reason, count] : rejection_reasons) {
                log << "  - " << reason << ": " << count << "\n";
            }
        }
        log << "================================\n\n";
        log.close();
    }
    
    // IMPORTANT: Write FlatBuffer binary file for UI panel (space-efficient)
    std::string fb_path = "resources/models/GRIM-text/training/data/verification_stats.bin";
    fs::create_directories(fs::path(fb_path).parent_path());
    
    flatbuffers::FlatBufferBuilder builder(512);
    
    auto stats_fb = GRIMWebTraining::CreateVerificationInfo(
        builder,
        GRIMWebTraining::VerificationStatus_VERIFIED,
        passed_verification > 0 ? static_cast<float>(passed_verification) / total_processed : 0.0f,
        0,  // num_cross_refs
        0,  // cross_ref_urls
        std::chrono::system_clock::now().time_since_epoch().count(),
        builder.CreateString("automatic"),
        0.0f,  // content_quality_score
        0.0f,  // factual_consistency
        false,  // is_duplicate
        false,  // is_malformed
        true    // passed_filters
    );
    
    builder.Finish(stats_fb);
    
    std::ofstream fb_file(fb_path, std::ios::binary);
    if (fb_file.is_open()) {
        fb_file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
        fb_file.close();
    }
    
    // Also write minimal JSON for backward compatibility with UI
    std::string json_path = "resources/models/GRIM-text/training/data/verification_stats.json";
    std::ofstream json_file(json_path);
    if (json_file.is_open()) {
        json j;
        j["total_processed"] = total_processed;
        j["passed_verification"] = passed_verification;
        j["failed_verification"] = failed_verification;
        j["domain_rejected"] = domain_rejected;
        j["quality_rejected"] = quality_rejected;
        j["duplicate_rejected"] = duplicate_rejected;
        j["semantic_rejected"] = semantic_rejected;
        j["high_quality_count"] = high_quality_count;
        j["medium_quality_count"] = medium_quality_count;
        j["low_quality_count"] = low_quality_count;
        
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
        j["timestamp"] = ss.str();
        
        json_file << j.dump(2);
        json_file.close();
    }
}
