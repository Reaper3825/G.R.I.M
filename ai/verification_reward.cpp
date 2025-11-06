/**
 * @file verification_reward.cpp
 * @brief Self-contained verification-based RL rewards (no training dependencies)
 */

#include "verification_reward.hpp"
#include <algorithm>
#include <regex>
#include <cctype>
#include <unordered_set>
#include <unordered_map>

namespace GRIM::RL {

// Domain whitelist - trusted sources for verification
static const std::unordered_set<std::string> TRUSTED_DOMAINS = {
    "github.com", "arxiv.org", "wikipedia.org", "stackoverflow.com",
    "huggingface.co", "python.org", "pytorch.org", "tensorflow.org",
    "gutenberg.org", "ocw.mit.edu", "plato.stanford.edu", "iep.utm.edu",
    "philpapers.org", "ncbi.nlm.nih.gov", "ark.intel.com", "nvidia.com",
    "techpowerup.com", "wikichip.org", "developer.mozilla.org",
    "doc.rust-lang.org", "golang.org", "docs.microsoft.com",
    "glottolog.org", "english-corpora.org", "wordnik.com",
    "americanrhetoric.com", "avalon.law.yale.edu", "owl.purdue.edu",
    "writingcenter.unc.edu", "earlymoderntexts.com",
    "philsci-archive.pitt.edu", "openstax.org", "openslr.org",
    "datashare.ed.ac.uk", "commonvoice.mozilla.org",
    "classics.mit.edu", "openlogicproject.org",
    "doabooks.org", "jstor.org",
    "public.oed.com", "dictionary.cambridge.org", "bartleby.com",
    "chicagomanualofstyle.org", "nature.com", "sciencedirect.com",
    "academic.oup.com", "cambridge.org", "archive.org", "muse.jhu.edu",
    "linguisticsociety.org", "quickanddirtytips.com", "merriam-webster.com",
    "springer.com", "oxfordhandbooks.com"
};

// Source type reliability weights
static const std::unordered_map<std::string, float> SOURCE_TYPE_WEIGHTS = {
    {"arxiv", 0.95f},
    {"philosophy", 0.90f},
    {"academic_papers", 0.90f},
    {"tech_docs", 0.90f},
    {"classical_texts", 0.95f},
    {"logic", 0.95f},
    {"rhetoric", 0.85f},
    {"gutenberg", 0.90f},
    {"wikipedia", 0.70f},
    {"stackoverflow", 0.75f},
    {"github", 0.80f},
    {"hardware_specs", 0.85f},
    {"linguistics", 0.85f},
    {"grammar", 0.95f},
    {"theoretical_reasoning", 0.98f},
    {"theoretical_science", 0.97f},
    {"erudite_writing", 0.92f}
};

class VerificationReward::Impl {
public:
    Config config;
    Stats stats;
    
    Impl() = default;
    
    float calculateContentQuality(const std::string& content) {
        float quality = 1.0f;
        
        // Length checks
        if (content.length() < config.min_content_length) {
            quality *= (static_cast<float>(content.length()) / config.min_content_length);
        }
        if (content.length() > config.max_content_length) {
            quality *= 0.8f;  // Slight penalty for very long content
        }
        
        // Word count
        size_t word_count = std::count_if(content.begin(), content.end(), 
            [](char c) { return std::isspace(c); }) + 1;
        
        if (word_count < config.min_word_count) {
            quality *= (static_cast<float>(word_count) / config.min_word_count);
        }
        
        // Check for excessive repetition
        std::regex repeated_chars(R"((.)\1{10,})");
        if (std::regex_search(content, repeated_chars)) {
            quality *= 0.5f;
        }
        
        // Check for proper sentence structure
        size_t sentence_endings = std::count(content.begin(), content.end(), '.') +
                                 std::count(content.begin(), content.end(), '!') +
                                 std::count(content.begin(), content.end(), '?');
        if (sentence_endings > 0) {
            float avg_sentence_length = static_cast<float>(word_count) / sentence_endings;
            if (avg_sentence_length < 5.0f || avg_sentence_length > 100.0f) {
                quality *= 0.7f;  // Too short or too long sentences
            }
        }
        
        return std::clamp(quality, 0.0f, 1.0f);
    }
    
    std::string extractDomain(const std::string& url) {
        std::regex domain_regex(R"(^(?:https?://)?(?:www\.)?([^/]+))");
        std::smatch match;
        if (std::regex_search(url, match, domain_regex) && match.size() > 1) {
            return match[1].str();
        }
        return "";
    }
    
    bool isDomainTrusted(const std::string& domain) {
        return TRUSTED_DOMAINS.find(domain) != TRUSTED_DOMAINS.end();
    }
    
    float getSourceTypeWeight(const std::string& source_type) {
        auto it = SOURCE_TYPE_WEIGHTS.find(source_type);
        if (it != SOURCE_TYPE_WEIGHTS.end()) {
            return it->second;
        }
        return 0.7f;  // Default weight for unknown source types
    }
};

VerificationReward::VerificationReward() : pImpl(std::make_unique<Impl>()) {
}

VerificationReward::~VerificationReward() = default;

float VerificationReward::evaluateContent(
    const std::string& content,
    const std::string& source_url,
    const std::string& source_type
) {
    auto score = getDetailedScore(content, source_url, source_type);
    
    // Update stats
    pImpl->stats.total_evaluated++;
    pImpl->stats.avg_score = 
        (pImpl->stats.avg_score * (pImpl->stats.total_evaluated - 1) + score.final_score) 
        / pImpl->stats.total_evaluated;
    
    if (score.final_score >= 0.8f) {
        pImpl->stats.high_quality++;
    } else if (score.final_score >= 0.5f) {
        pImpl->stats.medium_quality++;
    } else {
        pImpl->stats.low_quality++;
    }
    
    return score.final_score;
}

std::vector<float> VerificationReward::evaluateBatch(
    const std::vector<std::string>& contents,
    const std::vector<std::string>& source_urls
) {
    std::vector<float> scores;
    scores.reserve(contents.size());
    
    for (size_t i = 0; i < contents.size(); ++i) {
        std::string url = (i < source_urls.size()) ? source_urls[i] : "";
        scores.push_back(evaluateContent(contents[i], url));
    }
    
    return scores;
}

bool VerificationReward::isSourceTrusted(const std::string& url) {
    if (url.empty()) return false;
    
    std::string domain = pImpl->extractDomain(url);
    return pImpl->isDomainTrusted(domain);
}

VerificationReward::VerificationScore VerificationReward::getDetailedScore(
    const std::string& content,
    const std::string& source_url,
    const std::string& source_type
) {
    VerificationScore score;
    
    // 1. Content quality check
    score.content_quality = pImpl->calculateContentQuality(content);
    
    // 2. Domain trust check
    if (!source_url.empty()) {
        std::string domain = pImpl->extractDomain(source_url);
        score.is_trusted_domain = pImpl->isDomainTrusted(domain);
        score.domain_score = score.is_trusted_domain ? 1.0f : 0.3f;
    } else {
        score.domain_score = 0.5f;  // Neutral for unknown source
    }
    
    // 3. Source type weight
    float source_weight = pImpl->getSourceTypeWeight(source_type);
    score.base_score = source_weight;
    
    // 4. Compute final weighted score
    float domain_weight = pImpl->config.domain_weight;
    float quality_weight = pImpl->config.quality_weight;
    float source_weight_factor = pImpl->config.source_weight;
    
    score.final_score = 
        (score.domain_score * domain_weight) +
        (score.content_quality * quality_weight) +
        (source_weight * source_weight_factor);
    
    score.final_score = std::clamp(score.final_score, 0.0f, 1.0f);
    
    // Set verification method
    if (score.is_trusted_domain) {
        score.verification_method = "domain_whitelist";
    } else if (!source_type.empty()) {
        score.verification_method = "source_type_weight";
    } else {
        score.verification_method = "content_quality_only";
    }
    
    return score;
}

void VerificationReward::setConfig(const Config& config) {
    pImpl->config = config;
}

VerificationReward::Config VerificationReward::getConfig() const {
    return pImpl->config;
}

VerificationReward::Stats VerificationReward::getStats() const {
    return pImpl->stats;
}

void VerificationReward::resetStats() {
    pImpl->stats = Stats{};
}

// Global instance
static VerificationReward* g_verification_reward = nullptr;

VerificationReward& getVerificationReward() {
    if (!g_verification_reward) {
        g_verification_reward = new VerificationReward();
    }
    return *g_verification_reward;
}

} // namespace GRIM::RL

// =========================================================
// C API for external integration
// =========================================================

extern "C" {

float GRIM_VerifyContent(const char* content, const char* source_url, const char* source_type) {
    if (!content) return 0.0f;
    
    std::string content_str(content);
    std::string url_str = source_url ? source_url : "";
    std::string type_str = source_type ? source_type : "";
    
    return GRIM::RL::getVerificationReward().evaluateContent(content_str, url_str, type_str);
}

int GRIM_IsTrustedSource(const char* url) {
    if (!url) return 0;
    return GRIM::RL::getVerificationReward().isSourceTrusted(url) ? 1 : 0;
}

void GRIM_GetVerificationStats(
    size_t* total_evaluated,
    size_t* high_quality,
    size_t* medium_quality,
    size_t* low_quality,
    float* avg_score
) {
    auto stats = GRIM::RL::getVerificationReward().getStats();
    
    if (total_evaluated) *total_evaluated = stats.total_evaluated;
    if (high_quality) *high_quality = stats.high_quality;
    if (medium_quality) *medium_quality = stats.medium_quality;
    if (low_quality) *low_quality = stats.low_quality;
    if (avg_score) *avg_score = stats.avg_score;
}

void GRIM_ResetVerificationStats() {
    GRIM::RL::getVerificationReward().resetStats();
}

} // extern "C"
