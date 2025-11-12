//======================================================//
//  Data Preprocessor for Neural Network Training
//  Cleans and prepares web-collected data
//======================================================//

#pragma once
#include <string>
#include <vector>
#include <regex>
#include <unordered_set>
#include <algorithm>
#include <sstream>

namespace GRIM {
namespace Training {

struct PreprocessorConfig {
    // Text cleaning
    bool remove_html = true;
    bool normalize_urls = true;
    bool normalize_whitespace = true;
    bool remove_control_chars = true;
    
    // Quality filtering
    int min_length = 50;           // characters
    int max_length = 10000;        // characters
    int min_words = 10;
    float max_repetition_ratio = 0.3f;  // % of repeated tokens
    float min_alpha_ratio = 0.7f;  // % of alphabetic chars
    
    // Deduplication
    bool deduplicate = true;
    int dedup_ngram_size = 13;     // 13-gram deduplication
    float dedup_threshold = 0.8f;   // 80% overlap = duplicate
    
    // Special tokens
    std::string bos_token = "<|startoftext|>";
    std::string eos_token = "<|endoftext|>";
    std::string pad_token = "<|pad|>";
    std::string unk_token = "<|unk|>";
};

class DataPreprocessor {
public:
    explicit DataPreprocessor(const PreprocessorConfig& config = PreprocessorConfig())
        : config_(config) {}
    
    // Main preprocessing pipeline
    std::string preprocess(const std::string& text) {
        std::string result = text;
        
        if (config_.remove_html) {
            result = removeHTML(result);
        }
        
        if (config_.normalize_urls) {
            result = normalizeURLs(result);
        }
        
        if (config_.remove_control_chars) {
            result = removeControlChars(result);
        }
        
        if (config_.normalize_whitespace) {
            result = normalizeWhitespace(result);
        }
        
        return result;
    }
    
    // Quality filtering
    bool passesQualityFilter(const std::string& text) {
        // Length check
        if (text.length() < static_cast<size_t>(config_.min_length) ||
            text.length() > static_cast<size_t>(config_.max_length)) {
            return false;
        }
        
        // Word count check
        int word_count = countWords(text);
        if (word_count < config_.min_words) {
            return false;
        }
        
        // Alphabetic ratio check
        float alpha_ratio = computeAlphaRatio(text);
        if (alpha_ratio < config_.min_alpha_ratio) {
            return false;
        }
        
        // Repetition check
        float rep_ratio = computeRepetitionRatio(text);
        if (rep_ratio > config_.max_repetition_ratio) {
            return false;
        }
        
        return true;
    }
    
    // Deduplication
    bool isDuplicate(const std::string& text) {
        auto ngrams = extractNgrams(text, config_.dedup_ngram_size);
        
        // Check overlap with seen ngrams
        int overlap_count = 0;
        for (const auto& ngram : ngrams) {
            if (seen_ngrams_.count(ngram)) {
                overlap_count++;
            }
        }
        
        if (ngrams.empty()) return false;
        
        float overlap_ratio = static_cast<float>(overlap_count) / ngrams.size();
        
        if (overlap_ratio > config_.dedup_threshold) {
            return true;  // Duplicate
        }
        
        // Add to seen set
        for (const auto& ngram : ngrams) {
            seen_ngrams_.insert(ngram);
        }
        
        return false;
    }
    
    // Add special tokens
    std::string addSpecialTokens(const std::string& text) {
        return config_.bos_token + " " + text + " " + config_.eos_token;
    }
    
    // Reset deduplication state
    void resetDeduplication() {
        seen_ngrams_.clear();
    }
    
private:
    PreprocessorConfig config_;
    std::unordered_set<std::string> seen_ngrams_;
    
    std::string removeHTML(const std::string& text) {
        std::string result = text;
        
        // Remove script and style tags with their content
        static std::regex script_regex("<script[^>]*>.*?</script>", std::regex::icase);
        static std::regex style_regex("<style[^>]*>.*?</style>", std::regex::icase);
        result = std::regex_replace(result, script_regex, " ");
        result = std::regex_replace(result, style_regex, " ");
        
        // Remove HTML comments
        static std::regex comment_regex("<!--.*?-->");
        result = std::regex_replace(result, comment_regex, " ");
        
        // Convert common block tags to newlines for better text flow
        static std::regex block_tags("<(?:p|br|div|h[1-6]|li|tr)[^>]*>", std::regex::icase);
        result = std::regex_replace(result, block_tags, "\n");
        
        // Remove all remaining HTML tags
        static std::regex html_tag_regex("<[^>]*>");
        result = std::regex_replace(result, html_tag_regex, " ");
        
        // Decode common HTML entities
        static std::vector<std::pair<std::string, std::string>> entities = {
            {"&nbsp;", " "}, {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, 
            {"&quot;", "\""}, {"&#39;", "'"}, {"&apos;", "'"}, {"&mdash;", "—"},
            {"&ndash;", "–"}, {"&hellip;", "..."}, {"&bull;", "•"}
        };
        for (const auto& [entity, replacement] : entities) {
            size_t pos = 0;
            while ((pos = result.find(entity, pos)) != std::string::npos) {
                result.replace(pos, entity.length(), replacement);
                pos += replacement.length();
            }
        }
        
        // Remove remaining HTML entities
        static std::regex html_entity_regex("&[a-zA-Z0-9#]+;");
        result = std::regex_replace(result, html_entity_regex, " ");
        
        // Clean up excessive whitespace
        static std::regex multi_space_regex("  +");
        result = std::regex_replace(result, multi_space_regex, " ");
        
        static std::regex multi_newline_regex("\n\n+");
        result = std::regex_replace(result, multi_newline_regex, "\n\n");
        
        return result;
    }
    
    std::string normalizeURLs(const std::string& text) {
        static std::regex url_regex(R"(https?://[^\s]+)");
        return std::regex_replace(text, url_regex, "<URL>");
    }
    
    std::string removeControlChars(const std::string& text) {
        std::string result;
        result.reserve(text.size());
        
        for (char c : text) {
            // Keep printable chars, tabs, newlines
            if (c >= 32 || c == '\t' || c == '\n' || c == '\r') {
                result += c;
            }
        }
        
        return result;
    }
    
    std::string normalizeWhitespace(const std::string& text) {
        static std::regex multi_space_regex("[ \\t]+");
        static std::regex multi_newline_regex("\\n{3,}");
        
        std::string result = std::regex_replace(text, multi_space_regex, " ");
        result = std::regex_replace(result, multi_newline_regex, "\n\n");
        
        // Trim leading/trailing whitespace
        result.erase(0, result.find_first_not_of(" \t\n\r"));
        result.erase(result.find_last_not_of(" \t\n\r") + 1);
        
        return result;
    }
    
    int countWords(const std::string& text) {
        std::istringstream iss(text);
        std::string word;
        int count = 0;
        while (iss >> word) {
            count++;
        }
        return count;
    }
    
    float computeAlphaRatio(const std::string& text) {
        if (text.empty()) return 0.0f;
        
        int alpha_count = 0;
        for (char c : text) {
            if (std::isalpha(static_cast<unsigned char>(c))) {
                alpha_count++;
            }
        }
        
        return static_cast<float>(alpha_count) / text.length();
    }
    
    float computeRepetitionRatio(const std::string& text) {
        auto words = tokenizeWords(text);
        if (words.size() < 10) return 0.0f;
        
        std::unordered_set<std::string> unique_words(words.begin(), words.end());
        
        // Check for repeated 3-grams
        int repeated_trigrams = 0;
        std::unordered_set<std::string> seen_trigrams;
        
        for (size_t i = 0; i + 2 < words.size(); ++i) {
            std::string trigram = words[i] + " " + words[i+1] + " " + words[i+2];
            if (seen_trigrams.count(trigram)) {
                repeated_trigrams++;
            }
            seen_trigrams.insert(trigram);
        }
        
        if (words.size() < 3) return 0.0f;
        return static_cast<float>(repeated_trigrams) / (words.size() - 2);
    }
    
    std::vector<std::string> tokenizeWords(const std::string& text) {
        std::vector<std::string> words;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            words.push_back(word);
        }
        return words;
    }
    
    std::vector<std::string> extractNgrams(const std::string& text, int n) {
        std::vector<std::string> ngrams;
        auto words = tokenizeWords(text);
        
        if (words.size() < static_cast<size_t>(n)) {
            return ngrams;
        }
        
        for (size_t i = 0; i <= words.size() - n; ++i) {
            std::string ngram;
            for (int j = 0; j < n; ++j) {
                if (j > 0) ngram += " ";
                ngram += words[i + j];
            }
            ngrams.push_back(ngram);
        }
        
        return ngrams;
    }
};

} // namespace Training
} // namespace GRIM
