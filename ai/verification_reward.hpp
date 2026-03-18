#pragma once
/**
 * @file verification_reward.hpp
 * @brief Integration between data verifier and RL reward system
 * 
 * Uses verification reliability scores to provide rewards for:
 * - Information quality assessment
 * - Source credibility evaluation
 * - Content accuracy validation
 */

#include <string>
#include <memory>
#include <vector>

namespace GRIM::RL {

/**
 * @brief Verification-based reward calculation
 * 
 * Integrates the data verifier's reliability scoring into the RL reward system.
 * Rewards GRIM for:
 * - Gathering information from reliable sources
 * - Producing factually consistent content
 * - Cross-referencing claims
 * - Avoiding low-quality or malformed information
 */
class VerificationReward {
public:
    VerificationReward();
    ~VerificationReward();

    /**
     * @brief Evaluate text content and assign reward
     * 
     * @param content Text to verify
     * @param source_url Optional source URL for domain checking
     * @param source_type Type of source (e.g., "tech_docs", "arxiv")
     * @return Reward value 0.0 to 1.0 (reliability score)
     */
    float evaluateContent(
        const std::string& content,
        const std::string& source_url = "",
        const std::string& source_type = ""
    );

    /**
     * @brief Batch evaluation for multiple content pieces
     * 
     * @param contents Vector of text contents
     * @param source_urls Optional vector of source URLs
     * @return Vector of reliability scores
     */
    std::vector<float> evaluateBatch(
        const std::vector<std::string>& contents,
        const std::vector<std::string>& source_urls = {}
    );

    /**
     * @brief Check if source domain is whitelisted
     * 
     * @param url Source URL to check
     * @return true if domain is approved, false otherwise
     */
    bool isSourceTrusted(const std::string& url);

    /**
     * @brief Get detailed verification breakdown
     * 
     * Returns detailed scoring:
     * - base_score: Content quality (0.0-1.0)
     * - domain_score: Source trustworthiness (0.0-1.0)
     * - cross_ref_score: Cross-reference validation (0.0-1.0)
     * - final_score: Weighted combination
     */
    struct VerificationScore {
        float base_score = 0.0f;
        float domain_score = 0.0f;
        float cross_ref_score = 0.0f;
        float content_quality = 0.0f;
        float final_score = 0.0f;
        
        bool is_malformed = false;
        bool is_trusted_domain = false;
        int cross_references_found = 0;
        
        std::string verification_method;
    };

    VerificationScore getDetailedScore(
        const std::string& content,
        const std::string& source_url = "",
        const std::string& source_type = ""
    );

    /**
     * @brief Configure verification parameters
     */
    struct Config {
        float min_content_length = 100.0f;
        float max_content_length = 50000.0f;
        float min_word_count = 20.0f;
        
        // Reward weights
        float domain_weight = 0.4f;        // Weight for trusted domain
        float quality_weight = 0.4f;       // Weight for content quality
        float source_weight = 0.2f;        // Weight for source type
        
        // Penalties
        float malformed_penalty = 0.5f;    // Penalty for malformed content
        float untrusted_domain_penalty = 0.3f;  // Penalty for unknown domains
        
        bool enable_cross_check = false;   // Cross-checking disabled (standalone)
    };

    void setConfig(const Config& config);
    Config getConfig() const;

    /**
     * @brief Add domain to whitelist
     */
    void addTrustedDomain(const std::string& domain);

    /**
     * @brief Add domain to blacklist
     */
    void blockDomain(const std::string& domain);

    /**
     * @brief Set source type weight (inherent trust level)
     * 
     * @param source_type Type identifier (e.g., "arxiv", "wikipedia")
     * @param weight Trust weight 0.0-1.0
     */
    void setSourceTypeWeight(const std::string& source_type, float weight);

    /**
     * @brief Get verification statistics
     */
    struct Stats {
        size_t total_evaluated = 0;
        size_t high_quality = 0;      // score >= 0.8
        size_t medium_quality = 0;    // score >= 0.5
        size_t low_quality = 0;       // score < 0.5
        
        float avg_score = 0.0f;
        float avg_content_quality = 0.0f;
        float avg_domain_score = 0.0f;
    };

    Stats getStats() const;
    void resetStats();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * @brief Global verification reward instance
 */
VerificationReward& getVerificationReward();

/**
 * @brief Export functions for C API
 */
extern "C" {
    /**
     * @brief Evaluate content and return reward score
     * 
     * @param content Text content to evaluate
     * @param source_url Optional source URL (can be NULL)
     * @param source_type Optional source type (can be NULL)
     * @return Reward score 0.0 to 1.0
     */
    #if defined(_WIN32)
    __declspec(dllexport)
    #else
    __attribute__((visibility("default")))
    #endif
    float GRIM_VerifyContent(
        const char* content,
        const char* source_url,
        const char* source_type
    );

    /**
     * @brief Check if domain is trusted
     * 
     * @param url URL to check
     * @return 1 if trusted, 0 otherwise
     */
    #if defined(_WIN32)
    __declspec(dllexport)
    #else
    __attribute__((visibility("default")))
    #endif
    int GRIM_IsTrustedSource(const char* url);

    /**
     * @brief Add domain to trusted list
     * 
     * @param domain Domain to add (e.g., "arxiv.org")
     */
    #if defined(_WIN32)
    __declspec(dllexport)
    #else
    __attribute__((visibility("default")))
    #endif
    void GRIM_AddTrustedDomain(const char* domain);
}

} // namespace GRIM::RL
