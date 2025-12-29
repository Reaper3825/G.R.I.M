//======================================================//
//  Data Preprocessor for Neural Network Training
//  Cleans and prepares web-collected data
//======================================================//

#pragma once
#include <string>
#include <vector>
#include <unordered_set>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstring>

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
    
    // Token-based length limit (critical for model compatibility)
    // Rough estimate: 1 token ≈ 4-5 characters for English text
    // With max_seq_len=900, limit to ~3500 chars to stay safe
    int max_token_estimate_chars = 3500;  // characters (approx max_seq_len * 4)
    
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
        // Length check (character-based)
        if (text.length() < static_cast<size_t>(config_.min_length) ||
            text.length() > static_cast<size_t>(config_.max_length)) {
            return false;
        }
        
        // Token-estimate length check (critical for model compatibility)
        // This prevents sequences from exceeding model's max_seq_len after tokenization
        if (config_.max_token_estimate_chars > 0 &&
            text.length() > static_cast<size_t>(config_.max_token_estimate_chars)) {
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
    
    // Split long text into chunks that fit within token limits
    // This preserves valuable long-form content instead of discarding it
    std::vector<std::string> chunkLongText(const std::string& text) {
        std::vector<std::string> chunks;
        
        // If text is within limit, return as-is
        if (text.length() <= static_cast<size_t>(config_.max_token_estimate_chars)) {
            chunks.push_back(text);
            return chunks;
        }
        
        // Split on sentence boundaries (., !, ?) to keep coherent chunks
        size_t target_chunk_size = config_.max_token_estimate_chars;
        size_t overlap = 100;  // 100 char overlap between chunks for context
        
        size_t pos = 0;
        while (pos < text.length()) {
            size_t chunk_end = std::min(pos + target_chunk_size, text.length());
            
            // If not at end, try to break on sentence boundary
            if (chunk_end < text.length()) {
                // Look back up to 200 chars for a sentence ending
                size_t search_start = (chunk_end > 200) ? chunk_end - 200 : pos;
                size_t best_break = std::string::npos;
                
                // Start from the last valid character before chunk_end.
                for (size_t i = (chunk_end == 0 ? 0 : chunk_end - 1); i > search_start; --i) {
                    char c = text[i];
                    if (c == '.' || c == '!' || c == '?') {
                        // Check if followed by space or end
                        if (i + 1 >= text.length() || text[i + 1] == ' ' || text[i + 1] == '\n') {
                            best_break = i + 1;
                            break;
                        }
                    }
                }
                
                if (best_break != std::string::npos) {
                    chunk_end = best_break;
                }
            }
            
            std::string chunk = text.substr(pos, chunk_end - pos);
            
            // Only add if chunk meets minimum quality
            if (chunk.length() >= static_cast<size_t>(config_.min_length)) {
                chunks.push_back(chunk);
            }
            
            // If we reached the end, we're done. This prevents an infinite loop
            // when the remaining tail is <= overlap (chunk_end-overlap == pos).
            if (chunk_end >= text.length()) {
                break;
            }

            // Move forward, with overlap for context continuity.
            size_t next_pos = (chunk_end > overlap) ? (chunk_end - overlap) : chunk_end;
            if (next_pos <= pos) {
                // Safety: guarantee forward progress.
                next_pos = chunk_end;
            }
            pos = next_pos;
            
            // Prevent infinite loop
            if (pos >= text.length()) break;
        }
        
        return chunks;
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

    static inline char toLowerAscii(char c) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    static inline bool startsWithIgnoreCase(const std::string& s, size_t pos, const char* literal) {
        for (size_t i = 0; literal[i] != '\0'; ++i) {
            if (pos + i >= s.size()) return false;
            if (toLowerAscii(s[pos + i]) != toLowerAscii(literal[i])) return false;
        }
        return true;
    }

    static inline size_t findIgnoreCase(const std::string& s, const char* needle, size_t startPos) {
        if (!needle || needle[0] == '\0') return startPos;
        for (size_t i = startPos; i < s.size(); ++i) {
            if (startsWithIgnoreCase(s, i, needle)) return i;
        }
        return std::string::npos;
    }

    static inline bool isTagNameChar(char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == ':';
    }
    
    std::string removeHTML(const std::string& text) {
        // NOTE: std::regex can exhibit pathological slowdowns on some large HTML blobs.
        // This implementation uses linear scans to avoid apparent "hangs".
        std::string out;
        out.reserve(text.size());

        size_t i = 0;
        while (i < text.size()) {
            const char c = text[i];
            if (c != '<') {
                out.push_back(c);
                ++i;
                continue;
            }

            // HTML comment: <!-- ... -->
            if (startsWithIgnoreCase(text, i, "<!--")) {
                const size_t end = text.find("-->", i + 4);
                i = (end == std::string::npos) ? text.size() : (end + 3);
                out.push_back(' ');
                continue;
            }

            // Parse tag name
            size_t namePos = i + 1;
            while (namePos < text.size() && (text[namePos] == '/' || std::isspace(static_cast<unsigned char>(text[namePos])))) {
                ++namePos;
            }
            size_t nameEnd = namePos;
            while (nameEnd < text.size() && isTagNameChar(text[nameEnd])) {
                ++nameEnd;
            }

            std::string tag;
            tag.reserve(nameEnd > namePos ? (nameEnd - namePos) : 0);
            for (size_t k = namePos; k < nameEnd; ++k) {
                tag.push_back(toLowerAscii(text[k]));
            }

            // Find end of this tag
            const size_t close = text.find('>', i + 1);
            if (close == std::string::npos) {
                // Malformed, treat '<' as text
                out.push_back('<');
                ++i;
                continue;
            }

            // Drop script/style bodies entirely
            if (tag == "script" || tag == "style") {
                const char* closer = (tag == "script") ? "</script" : "</style";
                const size_t bodyEndStart = findIgnoreCase(text, closer, close + 1);
                if (bodyEndStart == std::string::npos) {
                    i = text.size();
                } else {
                    const size_t bodyEndClose = text.find('>', bodyEndStart);
                    i = (bodyEndClose == std::string::npos) ? text.size() : (bodyEndClose + 1);
                }
                out.push_back(' ');
                continue;
            }

            // Convert common block-ish tags to newlines
            if (tag == "p" || tag == "br" || tag == "div" || tag == "li" || tag == "tr" ||
                (tag.size() == 2 && tag[0] == 'h' && tag[1] >= '1' && tag[1] <= '6')) {
                out.push_back('\n');
            } else {
                out.push_back(' ');
            }

            i = close + 1;
        }

        // Decode a small set of common HTML entities (fast path)
        auto replaceAll = [&](const char* needle, const char* replacement) {
            size_t pos = 0;
            const size_t needleLen = std::strlen(needle);
            const size_t replLen = std::strlen(replacement);
            while ((pos = out.find(needle, pos)) != std::string::npos) {
                out.replace(pos, needleLen, replacement);
                pos += replLen;
            }
        };
        replaceAll("&nbsp;", " ");
        replaceAll("&amp;", "&");
        replaceAll("&lt;", "<");
        replaceAll("&gt;", ">");
        replaceAll("&quot;", "\"");
        replaceAll("&#39;", "'");
        replaceAll("&apos;", "'");
        replaceAll("&mdash;", "—");
        replaceAll("&ndash;", "–");
        replaceAll("&hellip;", "...");
        replaceAll("&bull;", "•");

        // Replace any remaining "&...;" entities with a space (bounded, linear-time)
        for (size_t p = 0; p < out.size(); ++p) {
            if (out[p] != '&') continue;

            size_t semi = std::string::npos;
            const size_t maxScan = (p + 32 < out.size()) ? (p + 32) : (out.size() - 1);
            for (size_t q = p + 1; q <= maxScan; ++q) {
                if (out[q] == ';') {
                    semi = q;
                    break;
                }
                // Early stop on whitespace; it's not an entity.
                if (out[q] == ' ' || out[q] == '\n' || out[q] == '\t') break;
            }

            if (semi != std::string::npos) {
                for (size_t q = p; q <= semi; ++q) out[q] = ' ';
                p = semi;
            }
        }

        return out;
    }
    
    std::string normalizeURLs(const std::string& text) {
        // Fast URL normalization (avoids regex worst-cases)
        std::string out;
        out.reserve(text.size());

        size_t i = 0;
        while (i < text.size()) {
            if (startsWithIgnoreCase(text, i, "http://") || startsWithIgnoreCase(text, i, "https://")) {
                out += "<URL>";
                // Skip until whitespace
                while (i < text.size() && !std::isspace(static_cast<unsigned char>(text[i]))) {
                    ++i;
                }
                continue;
            }
            out.push_back(text[i]);
            ++i;
        }

        return out;
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
        std::string out;
        out.reserve(text.size());

        bool in_space = false;
        int newline_run = 0;

        for (char c : text) {
            const unsigned char uc = static_cast<unsigned char>(c);

            if (c == '\r') continue;

            if (c == '\n') {
                newline_run++;
                in_space = false;
                if (newline_run <= 2) {
                    out.push_back('\n');
                }
                continue;
            }

            newline_run = 0;

            if (c == ' ' || c == '\t') {
                if (!in_space) {
                    out.push_back(' ');
                    in_space = true;
                }
                continue;
            }

            in_space = false;
            out.push_back(static_cast<char>(uc));
        }

        // Trim leading/trailing whitespace
        size_t start = 0;
        while (start < out.size() && (out[start] == ' ' || out[start] == '\n' || out[start] == '\t')) start++;
        size_t end = out.size();
        while (end > start && (out[end - 1] == ' ' || out[end - 1] == '\n' || out[end - 1] == '\t')) end--;

        return out.substr(start, end - start);
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
