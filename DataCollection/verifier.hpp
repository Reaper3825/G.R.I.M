#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <iostream>
#include "../../../control/ai_config_paths.hpp"

struct UnverifiedEntry {
    std::string content;
    std::string source_url;
    std::string source_type;
    std::string author;
    std::string metadata;
};

struct VerifiedEntry {
    std::string content;
    std::string source_url;
    std::string source_type;
    std::string author;
    std::string metadata;
    float reliability_score;
    time_t verification_time;
};

struct Config {
    std::string input_dir;   // Loaded from ai_config by constructor
    std::string output_dir;  // Loaded from ai_config by constructor
    float reliability_threshold = 0.6f;  // More forgiving - lowered from 0.6
    int min_cross_references = 2;
    bool enable_cross_check = true;
    size_t min_length = 75;              // More forgiving - lowered from 100
    size_t max_length = 75000;          // More forgiving - increased from 50000
    std::vector<std::string> domain_whitelist;
    std::unordered_map<std::string, float> source_type_weights;
    
    // Enhanced: Progressive filtering
    bool progressive_filtering = true;   // Apply filters gradually
    bool save_rejected = false;          // Save rejected entries for analysis
    bool verbose_logging = false;        // Log rejection reasons
    
    // Enhanced: Quality tiers
    float high_quality_threshold = 0.8f;
    float medium_quality_threshold = 0.6f;
    float low_quality_threshold = 0.4f;

    // Semantic verifier (DeBERTa ONNX)
    // DISABLED by default - causes crashes when running in GRIM.exe
    bool enable_semantic_model = false;
    bool semantic_use_gpu = false;
    bool semantic_hard_filter = true;
    int semantic_max_seq_length = 512;
    float semantic_min_score = 0.55f;
    float semantic_quality_weight = 0.35f;
    // Default paths point to the downloaded quality model layout under resources/
    std::string semantic_model_path = "resources/models/GRIM-text/quality/deberta-v3-base-mnli/model.onnx";
    std::string semantic_tokenizer_path = "resources/models/GRIM-text/quality/deberta-v3-base-mnli/spm.model";
    std::vector<std::string> semantic_positive_prompts;
    std::vector<std::string> semantic_negative_prompts;
};

struct Stats {
    size_t total_processed = 0;
    size_t passed_verification = 0;
    size_t failed_verification = 0;
    size_t domain_rejected = 0;
    size_t quality_rejected = 0;
    size_t duplicate_rejected = 0;
    size_t semantic_rejected = 0;
    
    // Enhanced: Quality tier tracking
    size_t high_quality_count = 0;
    size_t medium_quality_count = 0;
    size_t low_quality_count = 0;
    
    // Enhanced: Rejection reason tracking
    std::unordered_map<std::string, size_t> rejection_reasons;
    
    void writeSummaryToLog(const std::string& log_path = "logs/verification_stats.log") const;
};

class Verifier {
public:
    explicit Verifier(const Config& config);
    ~Verifier();
    
    // Progress callback type
    using ProgressCallback = std::function<void(float progress, size_t processed, size_t total)>;
    
    std::vector<UnverifiedEntry> load_unverified_entries() const;
    std::vector<VerifiedEntry> verify_entries(const std::vector<UnverifiedEntry>& entries) const;
    std::vector<VerifiedEntry> verify_entries(const std::vector<UnverifiedEntry>& entries, ProgressCallback callback) const;
    bool save_verified_entries(const std::vector<VerifiedEntry>& entries) const;
    Stats get_stats() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};
