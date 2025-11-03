#include "verifier.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <regex>
#include <cmath>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace grim {
namespace training {

// Simple content fingerprinting for similarity detection
struct ContentFingerprint {
    std::string source_url;
    std::vector<std::string> key_phrases;
    std::string normalized_content;
};

class Verifier::Impl {
public:
    VerifierConfig config;
    Stats stats;
    std::vector<ContentFingerprint> similarity_index;

    // Normalize text for comparison
    std::string normalize_text(const std::string& text) const {
        std::string normalized = text;
        
        // Convert to lowercase
        std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
        
        // Remove extra whitespace
        normalized = std::regex_replace(normalized, std::regex("\\s+"), " ");
        
        // Trim
        normalized.erase(0, normalized.find_first_not_of(" \t\n\r"));
        normalized.erase(normalized.find_last_not_of(" \t\n\r") + 1);
        
        return normalized;
    }

    // Extract key phrases from content
    std::vector<std::string> extract_key_phrases(const std::string& content) const {
        std::vector<std::string> phrases;
        
        // Simple extraction: sentences or noun phrases
        std::regex sentence_regex(R"([A-Z][^.!?]*[.!?])");
        auto sentences_begin = std::sregex_iterator(content.begin(), content.end(), sentence_regex);
        auto sentences_end = std::sregex_iterator();
        
        for (auto it = sentences_begin; it != sentences_end; ++it) {
            std::string sentence = it->str();
            if (sentence.length() > 20 && sentence.length() < 200) {
                phrases.push_back(normalize_text(sentence));
            }
        }
        
        return phrases;
    }

    // Calculate Jaccard similarity between two sets of phrases
    double calculate_similarity(const std::vector<std::string>& phrases1,
                               const std::vector<std::string>& phrases2) const {
        if (phrases1.empty() || phrases2.empty()) return 0.0;
        
        std::unordered_set<std::string> set1(phrases1.begin(), phrases1.end());
        std::unordered_set<std::string> set2(phrases2.begin(), phrases2.end());
        
        size_t intersection = 0;
        for (const auto& phrase : set1) {
            if (set2.count(phrase)) {
                intersection++;
            }
        }
        
        size_t union_size = set1.size() + set2.size() - intersection;
        return union_size > 0 ? static_cast<double>(intersection) / union_size : 0.0;
    }
};

Verifier::Verifier() : pImpl(std::make_unique<Impl>()) {
    pImpl->config = VerifierConfig{};
    
    // Set default source type weights
    pImpl->config.source_type_weights["news_api"] = 0.7;
    pImpl->config.source_type_weights["github"] = 0.9;
    pImpl->config.source_type_weights["tech_docs"] = 0.95;
}

Verifier::Verifier(const VerifierConfig& config) : pImpl(std::make_unique<Impl>()) {
    pImpl->config = config;
}

Verifier::~Verifier() = default;

size_t Verifier::verify_sources() {
    pImpl->stats = Stats{};  // Reset stats
    
    // Create output directory
    fs::create_directories(pImpl->config.output_dir);
    
    // Load all raw entries
    auto raw_entries = load_raw_entries();
    pImpl->stats.total_processed = raw_entries.size();
    
    if (raw_entries.empty()) {
        return 0;
    }
    
    // Build similarity index for cross-checking
    if (pImpl->config.require_cross_check) {
        build_similarity_index(raw_entries);
    }
    
    // Verify each entry
    std::vector<VerifiedEntry> verified_entries;
    
    for (const auto& entry : raw_entries) {
        auto verified = verify_entry(entry);
        
        if (verified) {
            verified_entries.push_back(*verified);
            pImpl->stats.passed++;
        } else {
            pImpl->stats.failed++;
        }
    }
    
    // Save verified entries
    if (!verified_entries.empty()) {
        save_verified_entries(verified_entries);
    }
    
    return pImpl->stats.passed;
}

std::unique_ptr<VerifiedEntry> Verifier::verify_entry(const UnverifiedEntry& entry) {
    // Step 1: Check domain whitelist/blacklist
    if (!is_domain_approved(entry.source_url)) {
        pImpl->stats.domain_rejected++;
        return nullptr;
    }
    
    // Step 2: Validate content format
    if (!is_well_formed(entry.content)) {
        pImpl->stats.malformed++;
        return nullptr;
    }
    
    // Step 3: Calculate base reliability score
    double base_score = calculate_base_score(entry);
    
    // Step 4: Cross-check with other sources if enabled
    std::vector<std::string> cross_refs;
    double cross_check_boost = 0.0;
    
    if (pImpl->config.require_cross_check) {
        cross_check_boost = cross_check_reliability(entry.content, cross_refs);
        
        // Require minimum cross-references
        if (cross_refs.size() < static_cast<size_t>(pImpl->config.min_cross_references)) {
            cross_check_boost *= 0.5;  // Penalize lack of corroboration
        }
    }
    
    // Step 5: Calculate final reliability score
    double final_score = std::min(1.0, base_score + cross_check_boost * 0.3);
    
    // Step 6: Check against threshold
    if (final_score < pImpl->config.min_reliability_threshold) {
        pImpl->stats.low_reliability++;
        return nullptr;
    }
    
    // Create verified entry
    auto verified = std::make_unique<VerifiedEntry>();
    verified->content = entry.content;
    verified->source_url = entry.source_url;
    verified->source_type = entry.source_type;
    verified->author = entry.author;
    verified->reliability_score = final_score;
    verified->cross_references = cross_refs;
    verified->metadata = entry.metadata;
    
    if (cross_refs.size() >= static_cast<size_t>(pImpl->config.min_cross_references)) {
        verified->verification_method = "cross_referenced";
    } else {
        verified->verification_method = "domain_trusted";
    }
    
    return verified;
}

bool Verifier::load_whitelist(const std::string& filepath) {
    try {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;
        
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line[0] != '#') {
                pImpl->config.domain_whitelist.insert(line);
            }
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

bool Verifier::load_blacklist(const std::string& filepath) {
    try {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;
        
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line[0] != '#') {
                pImpl->config.domain_blacklist.insert(line);
            }
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

void Verifier::add_to_whitelist(const std::string& domain) {
    pImpl->config.domain_whitelist.insert(domain);
}

void Verifier::add_to_blacklist(const std::string& domain) {
    pImpl->config.domain_blacklist.insert(domain);
}

void Verifier::set_source_type_weight(const std::string& source_type, double weight) {
    pImpl->config.source_type_weights[source_type] = std::clamp(weight, 0.0, 1.0);
}

Verifier::Stats Verifier::get_stats() const {
    return pImpl->stats;
}

void Verifier::reset_stats() {
    pImpl->stats = Stats{};
}

bool Verifier::is_domain_approved(const std::string& url) const {
    std::string domain = extract_domain(url);
    
    // Check blacklist first
    if (pImpl->config.domain_blacklist.count(domain)) {
        return false;
    }
    
    // If whitelist is empty, approve all (except blacklisted)
    if (pImpl->config.domain_whitelist.empty()) {
        return true;
    }
    
    // Check whitelist
    return pImpl->config.domain_whitelist.count(domain) > 0;
}

std::string Verifier::extract_domain(const std::string& url) const {
    // Simple domain extraction
    std::regex domain_regex(R"(^(?:https?://)?(?:www\.)?([^/]+))");
    std::smatch match;
    
    if (std::regex_search(url, match, domain_regex) && match.size() > 1) {
        return match[1].str();
    }
    
    return url;
}

double Verifier::calculate_base_score(const UnverifiedEntry& entry) const {
    double score = 0.5;  // Base neutral score
    
    // Apply source type weight
    auto it = pImpl->config.source_type_weights.find(entry.source_type);
    if (it != pImpl->config.source_type_weights.end()) {
        score = it->second;
    }
    
    // Boost for known authors
    if (!entry.author.empty() && entry.author != "Unknown") {
        score += 0.1;
    }
    
    // Boost for longer, more detailed content
    if (entry.content.length() > 500) {
        score += 0.05;
    }
    if (entry.content.length() > 1000) {
        score += 0.05;
    }
    
    return std::min(1.0, score);
}

double Verifier::cross_check_reliability(const std::string& content,
                                        std::vector<std::string>& cross_refs) const {
    auto similar_sources = find_similar_content(content);
    cross_refs = similar_sources;
    
    // More corroborating sources = higher boost
    double boost = std::min(0.3, similar_sources.size() * 0.1);
    
    return boost;
}

bool Verifier::is_well_formed(const std::string& content) const {
    // Check minimum length
    if (content.length() < 50) {
        return false;
    }
    
    // Check for excessive special characters (likely corrupted)
    size_t special_count = 0;
    for (char c : content) {
        if (!std::isalnum(c) && !std::isspace(c) && c != '.' && c != ',' && c != '-') {
            special_count++;
        }
    }
    
    if (special_count > content.length() / 4) {
        return false;  // More than 25% special chars
    }
    
    // Check for reasonable sentence structure
    size_t period_count = std::count(content.begin(), content.end(), '.');
    if (period_count == 0 && content.length() > 500) {
        return false;  // Long text with no sentences
    }
    
    return true;
}

std::vector<UnverifiedEntry> Verifier::load_raw_entries() const {
    std::vector<UnverifiedEntry> entries;
    
    if (!fs::exists(pImpl->config.input_dir)) {
        return entries;
    }
    
    for (const auto& entry : fs::directory_iterator(pImpl->config.input_dir)) {
        if (!entry.is_regular_file()) continue;
        
        std::string filepath = entry.path().string();
        
        // Only process .jsonl files
        if (filepath.find(".jsonl") == std::string::npos) continue;
        
        std::ifstream file(filepath);
        if (!file.is_open()) continue;
        
        std::string line;
        while (std::getline(file, line)) {
            try {
                json j = json::parse(line);
                
                UnverifiedEntry uentry;
                uentry.content = j.value("content", "");
                uentry.source_url = j.value("source_url", "");
                uentry.source_type = j.value("source_type", "");
                uentry.author = j.value("author", "");
                uentry.metadata = j.contains("metadata") ? j["metadata"].dump() : "{}";
                
                if (!uentry.content.empty()) {
                    entries.push_back(uentry);
                }
                
            } catch (const json::parse_error&) {
                continue;
            }
        }
    }
    
    return entries;
}

bool Verifier::save_verified_entries(const std::vector<VerifiedEntry>& entries) const {
    try {
        std::string filepath = pImpl->config.output_dir + "/verified.jsonl";
        std::ofstream outfile(filepath);
        
        if (!outfile.is_open()) return false;
        
        for (const auto& entry : entries) {
            json j;
            j["content"] = entry.content;
            j["source_url"] = entry.source_url;
            j["source_type"] = entry.source_type;
            j["author"] = entry.author;
            j["reliability_score"] = entry.reliability_score;
            j["verification_method"] = entry.verification_method;
            j["cross_references"] = entry.cross_references;
            j["metadata"] = json::parse(entry.metadata.empty() ? "{}" : entry.metadata);
            
            outfile << j.dump() << "\n";
        }
        
        outfile.close();
        return true;
        
    } catch (...) {
        return false;
    }
}

void Verifier::build_similarity_index(const std::vector<UnverifiedEntry>& entries) {
    pImpl->similarity_index.clear();
    
    for (const auto& entry : entries) {
        ContentFingerprint fp;
        fp.source_url = entry.source_url;
        fp.normalized_content = pImpl->normalize_text(entry.content);
        fp.key_phrases = pImpl->extract_key_phrases(entry.content);
        
        pImpl->similarity_index.push_back(fp);
    }
}

std::vector<std::string> Verifier::find_similar_content(const std::string& content) const {
    std::vector<std::string> similar_sources;
    
    auto key_phrases = pImpl->extract_key_phrases(content);
    
    for (const auto& fp : pImpl->similarity_index) {
        double similarity = pImpl->calculate_similarity(key_phrases, fp.key_phrases);
        
        if (similarity > 0.3) {  // 30% similarity threshold
            similar_sources.push_back(fp.source_url);
        }
    }
    
    return similar_sources;
}

} // namespace training
} // namespace grim
