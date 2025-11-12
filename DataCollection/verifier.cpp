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

namespace fs = std::filesystem;
using json = nlohmann::json;

class Verifier::Impl {
public:
    Config config;
    mutable Stats stats;

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
        
        // Check 1: Domain validation (if whitelist exists)
        std::string domain = pImpl->extract_domain(entry.source_url);
        if (!pImpl->is_domain_approved(domain)) {
            pImpl->stats.domain_rejected++;
            rejection_reason = "domain_not_approved";
            should_accept = false;
            
            if (pImpl->config.verbose_logging) {
                std::cerr << "REJECTED (domain): " << domain << "\n";
            }
        }
        
        // Check 2: Length validation (with more forgiving bounds)
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
        
        // Check 3: Reliability score
        if (should_accept) {
            VerifiedEntry ve = pImpl->verify_entry(entry);
            
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
