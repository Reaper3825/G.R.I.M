#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <functional>

namespace grim {
namespace training {

/**
 * @brief Entry from raw data that needs verification
 */
struct UnverifiedEntry {
    std::string content;
    std::string source_url;
    std::string source_type;
    std::string author;
    std::string metadata;
};

/**
 * @brief Verified data entry with reliability scoring
 */
struct VerifiedEntry {
    std::string content;
    std::string source_url;
    std::string source_type;
    std::string author;
    double reliability_score;  // 0.0 to 1.0
    std::string verification_method;
    std::vector<std::string> cross_references;  // URLs of corroborating sources
    std::string metadata;
};

/**
 * @brief Configuration for the verification process
 */
struct VerifierConfig {
    std::string input_dir = "data/raw";
    std::string output_dir = "data/verified";
    double min_reliability_threshold = 0.8;
    bool require_cross_check = true;
    int min_cross_references = 2;
    std::unordered_set<std::string> domain_whitelist;
    std::unordered_set<std::string> domain_blacklist;
    std::unordered_map<std::string, double> source_type_weights;  // Inherent trust per source type
};

/**
 * @brief Stage 2: Data Verifier
 * 
 * Filters and validates information accuracy and reliability.
 * Cross-checks claims, validates sources, and assigns reliability scores.
 */
class Verifier {
public:
    Verifier();
    explicit Verifier(const VerifierConfig& config);
    ~Verifier();

    /**
     * @brief Main entry point: Verify collected data sources
     * 
     * Checks source domains against whitelist, cross-checks identical claims
     * across multiple sources, drops low-confidence or malformed text.
     * Keeps entries with reliability score >= threshold.
     * 
     * @return Number of entries that passed verification
     */
    size_t verify_sources();

    /**
     * @brief Verify a single entry
     * 
     * @param entry Unverified data entry
     * @return Verified entry with reliability score, or nullptr if failed
     */
    std::unique_ptr<VerifiedEntry> verify_entry(const UnverifiedEntry& entry);

    /**
     * @brief Load domain whitelist from file
     */
    bool load_whitelist(const std::string& filepath);

    /**
     * @brief Load domain blacklist from file
     */
    bool load_blacklist(const std::string& filepath);

    /**
     * @brief Add domain to whitelist
     */
    void add_to_whitelist(const std::string& domain);

    /**
     * @brief Add domain to blacklist
     */
    void add_to_blacklist(const std::string& domain);

    /**
     * @brief Set source type weight (inherent trust level)
     */
    void set_source_type_weight(const std::string& source_type, double weight);

    /**
     * @brief Get verification statistics
     */
    struct Stats {
        size_t total_processed = 0;
        size_t passed = 0;
        size_t failed = 0;
        size_t domain_rejected = 0;
        size_t low_reliability = 0;
        size_t malformed = 0;
    };
    Stats get_stats() const;

    /**
     * @brief Reset statistics
     */
    void reset_stats();

private:
    /**
     * @brief Check if domain is whitelisted
     */
    bool is_domain_approved(const std::string& url) const;

    /**
     * @brief Extract domain from URL
     */
    std::string extract_domain(const std::string& url) const;

    /**
     * @brief Calculate base reliability score from source metadata
     */
    double calculate_base_score(const UnverifiedEntry& entry) const;

    /**
     * @brief Cross-check content against other sources
     */
    double cross_check_reliability(const std::string& content, 
                                   std::vector<std::string>& cross_refs) const;

    /**
     * @brief Validate content format and quality
     */
    bool is_well_formed(const std::string& content) const;

    /**
     * @brief Load all entries from input directory
     */
    std::vector<UnverifiedEntry> load_raw_entries() const;

    /**
     * @brief Save verified entries to output file
     */
    bool save_verified_entries(const std::vector<VerifiedEntry>& entries) const;

    /**
     * @brief Build content similarity index for cross-checking
     */
    void build_similarity_index(const std::vector<UnverifiedEntry>& entries);

    /**
     * @brief Find similar content in the index
     */
    std::vector<std::string> find_similar_content(const std::string& content) const;

    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace training
} // namespace grim
