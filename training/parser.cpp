#include "parser.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <regex>
#include <ctime>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace grim {
namespace training {

class Parser::Impl {
public:
    ParserConfig config;
    Stats stats;
    
    // Simple tokenizer (word-based for now)
    std::vector<int> simple_tokenize(const std::string& text) const {
        std::vector<int> tokens;
        std::istringstream iss(text);
        std::string word;
        
        while (iss >> word) {
            // Simple hash-based token ID
            size_t hash = std::hash<std::string>{}(word);
            tokens.push_back(static_cast<int>(hash % 50000));
        }
        
        return tokens;
    }

    // Extract sentences from text
    std::vector<std::string> extract_sentences(const std::string& text) const {
        std::vector<std::string> sentences;
        std::regex sentence_regex(R"([^.!?]+[.!?]+)");
        
        auto begin = std::sregex_iterator(text.begin(), text.end(), sentence_regex);
        auto end = std::sregex_iterator();
        
        for (auto it = begin; it != end; ++it) {
            std::string sentence = it->str();
            // Trim whitespace
            sentence.erase(0, sentence.find_first_not_of(" \t\n\r"));
            sentence.erase(sentence.find_last_not_of(" \t\n\r") + 1);
            
            if (!sentence.empty()) {
                sentences.push_back(sentence);
            }
        }
        
        return sentences;
    }

    // Calculate TF-IDF like importance score for sentences
    double calculate_importance(const std::string& sentence, const std::string& full_text) const {
        // Simple heuristic: length + word frequency
        double score = 0.0;
        
        // Prefer medium-length sentences
        if (sentence.length() > 20 && sentence.length() < 150) {
            score += 1.0;
        }
        
        // Check for important keywords
        std::vector<std::string> important_words = {
            "important", "key", "critical", "essential", "significant",
            "demonstrates", "shows", "indicates", "reveals", "suggests"
        };
        
        for (const auto& word : important_words) {
            if (sentence.find(word) != std::string::npos) {
                score += 0.5;
            }
        }
        
        return score;
    }
};

Parser::Parser() : pImpl(std::make_unique<Impl>()) {
    pImpl->config = ParserConfig{};
}

Parser::Parser(const ParserConfig& config) : pImpl(std::make_unique<Impl>()) {
    pImpl->config = config;
}

Parser::~Parser() = default;

size_t Parser::parse_verified_data() {
    pImpl->stats = Stats{};  // Reset stats
    
    // Create output directory
    fs::create_directories(pImpl->config.output_dir);
    
    // Load verified entries
    auto verified_entries = load_verified_entries();
    pImpl->stats.total_verified_entries = verified_entries.size();
    
    if (verified_entries.empty()) {
        return 0;
    }
    
    // Parse all entries
    std::vector<ParsedExample> all_examples;
    
    for (const auto& entry : verified_entries) {
        auto examples = parse_entry(entry);
        
        // Validate tokenization for each example
        for (const auto& example : examples) {
            if (validate_tokenization(example)) {
                all_examples.push_back(example);
            } else {
                pImpl->stats.tokenization_failures++;
            }
        }
    }
    
    pImpl->stats.total_parsed_examples = all_examples.size();
    
    // Save parsed examples
    if (!all_examples.empty()) {
        save_parsed_examples(all_examples);
    }
    
    return pImpl->stats.total_parsed_examples;
}

std::vector<ParsedExample> Parser::parse_entry(const VerifiedData& entry) {
    switch (pImpl->config.strategy) {
        case ParserConfig::ParseStrategy::QUESTION_ANSWER:
            return parse_as_qa(entry);
        case ParserConfig::ParseStrategy::SUMMARIZATION:
            return parse_as_summarization(entry);
        case ParserConfig::ParseStrategy::COMPLETION:
            return parse_as_completion(entry);
        case ParserConfig::ParseStrategy::INSTRUCTION:
        default:
            return parse_as_instruction(entry);
    }
}

bool Parser::load_tokenizer(const std::string& tokenizer_path) {
    pImpl->config.tokenizer_path = tokenizer_path;
    // In a full implementation, load actual tokenizer model here
    return fs::exists(tokenizer_path);
}

bool Parser::validate_tokenization(const ParsedExample& example) {
    // Tokenize the input text
    auto tokens = tokenize(example.input_text);
    
    // Check if within token limit
    if (tokens.size() > pImpl->config.max_token_length) {
        return false;
    }
    
    // Check if tokenization produced reasonable output
    if (tokens.empty() && !example.input_text.empty()) {
        return false;
    }
    
    return true;
}

void Parser::set_strategy(ParserConfig::ParseStrategy strategy) {
    pImpl->config.strategy = strategy;
}

Parser::Stats Parser::get_stats() const {
    return pImpl->stats;
}

void Parser::reset_stats() {
    pImpl->stats = Stats{};
}

std::vector<VerifiedData> Parser::load_verified_entries() const {
    std::vector<VerifiedData> entries;
    
    std::string filepath = pImpl->config.input_dir + "/verified.jsonl";
    
    if (!fs::exists(filepath)) {
        return entries;
    }
    
    std::ifstream file(filepath);
    if (!file.is_open()) return entries;
    
    std::string line;
    while (std::getline(file, line)) {
        try {
            json j = json::parse(line);
            
            VerifiedData data;
            data.content = j.value("content", "");
            data.source_url = j.value("source_url", "");
            data.source_type = j.value("source_type", "");
            data.author = j.value("author", "");
            data.reliability_score = j.value("reliability_score", 0.0);
            data.verification_method = j.value("verification_method", "");
            
            if (j.contains("cross_references") && j["cross_references"].is_array()) {
                for (const auto& ref : j["cross_references"]) {
                    data.cross_references.push_back(ref.get<std::string>());
                }
            }
            
            data.metadata = j.contains("metadata") ? j["metadata"].dump() : "{}";
            
            if (!data.content.empty()) {
                entries.push_back(data);
            }
            
        } catch (const json::parse_error&) {
            continue;
        }
    }
    
    return entries;
}

bool Parser::save_parsed_examples(const std::vector<ParsedExample>& examples) const {
    try {
        std::string filepath = pImpl->config.output_dir + "/parsed.jsonl";
        std::ofstream outfile(filepath);
        
        if (!outfile.is_open()) return false;
        
        for (const auto& example : examples) {
            json j;
            j["topic"] = example.topic;
            j["summary"] = example.summary;
            j["full_content"] = example.full_content;
            j["source_url"] = example.source_url;
            j["source_type"] = example.source_type;
            j["author"] = example.author;
            j["reliability_score"] = example.reliability_score;
            j["keywords"] = example.keywords;
            j["entities"] = example.entities;
            j["timestamp"] = example.timestamp;
            j["input_text"] = example.input_text;
            j["target_text"] = example.target_text;
            j["metadata"] = json::parse(example.metadata.empty() ? "{}" : example.metadata);
            
            outfile << j.dump() << "\n";
        }
        
        outfile.close();
        return true;
        
    } catch (...) {
        return false;
    }
}

std::string Parser::extract_topic(const std::string& content) const {
    // Extract first sentence or first 100 chars as topic
    auto sentences = pImpl->extract_sentences(content);
    
    if (!sentences.empty()) {
        std::string topic = sentences[0];
        if (topic.length() > 100) {
            topic = topic.substr(0, 97) + "...";
        }
        return topic;
    }
    
    return content.substr(0, std::min(content.length(), size_t(100)));
}

std::string Parser::generate_summary(const std::string& content) const {
    auto sentences = pImpl->extract_sentences(content);
    
    if (sentences.size() <= 3) {
        return content;
    }
    
    // Score sentences and pick top 3
    std::vector<std::pair<double, std::string>> scored_sentences;
    
    for (const auto& sentence : sentences) {
        double score = pImpl->calculate_importance(sentence, content);
        scored_sentences.push_back({score, sentence});
    }
    
    // Sort by score
    std::sort(scored_sentences.begin(), scored_sentences.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Take top 3
    std::stringstream summary;
    for (size_t i = 0; i < std::min(size_t(3), scored_sentences.size()); ++i) {
        summary << scored_sentences[i].second << " ";
    }
    
    return summary.str();
}

std::vector<std::string> Parser::extract_keywords(const std::string& content) const {
    std::vector<std::string> keywords;
    
    // Simple keyword extraction: find capitalized words and technical terms
    std::regex keyword_regex(R"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b)");
    
    auto begin = std::sregex_iterator(content.begin(), content.end(), keyword_regex);
    auto end = std::sregex_iterator();
    
    std::unordered_map<std::string, int> freq;
    
    for (auto it = begin; it != end; ++it) {
        std::string word = it->str();
        freq[word]++;
    }
    
    // Sort by frequency and take top keywords
    std::vector<std::pair<int, std::string>> sorted_keywords;
    for (const auto& kv : freq) {
        if (kv.second > 1) {  // Must appear at least twice
            sorted_keywords.push_back({kv.second, kv.first});
        }
    }
    
    std::sort(sorted_keywords.begin(), sorted_keywords.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    for (size_t i = 0; i < std::min(size_t(10), sorted_keywords.size()); ++i) {
        keywords.push_back(sorted_keywords[i].second);
    }
    
    pImpl->stats.keyword_extraction_count += keywords.size();
    
    return keywords;
}

std::vector<std::string> Parser::extract_entities(const std::string& content) const {
    std::vector<std::string> entities;
    
    // Simple NER: extract proper nouns and organizations
    std::regex entity_regex(R"(\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3}\b)");
    
    auto begin = std::sregex_iterator(content.begin(), content.end(), entity_regex);
    auto end = std::sregex_iterator();
    
    std::unordered_set<std::string> unique_entities;
    
    for (auto it = begin; it != end; ++it) {
        std::string entity = it->str();
        if (entity.length() > 2) {  // Ignore single letters
            unique_entities.insert(entity);
        }
    }
    
    entities.assign(unique_entities.begin(), unique_entities.end());
    pImpl->stats.entity_extraction_count += entities.size();
    
    return entities;
}

std::vector<ParsedExample> Parser::parse_as_qa(const VerifiedData& entry) {
    std::vector<ParsedExample> examples;
    
    auto sentences = pImpl->extract_sentences(entry.content);
    
    // Generate Q&A pairs from sentences
    for (size_t i = 0; i < sentences.size(); ++i) {
        if (sentences[i].length() < 20) continue;
        
        ParsedExample example;
        example.topic = extract_topic(entry.content);
        example.summary = generate_summary(entry.content);
        example.full_content = entry.content;
        example.source_url = entry.source_url;
        example.source_type = entry.source_type;
        example.author = entry.author;
        example.reliability_score = entry.reliability_score;
        example.timestamp = std::to_string(std::time(nullptr));
        example.metadata = entry.metadata;
        
        if (pImpl->config.extract_keywords) {
            example.keywords = extract_keywords(entry.content);
        }
        
        if (pImpl->config.extract_entities) {
            example.entities = extract_entities(entry.content);
        }
        
        // Generate question from sentence
        example.input_text = "Question: What does the text say about " + 
                           example.topic + "?\nContext: " + entry.content;
        example.target_text = sentences[i];
        
        examples.push_back(example);
        
        if (examples.size() >= 5) break;  // Limit Q&A pairs per entry
    }
    
    return examples;
}

std::vector<ParsedExample> Parser::parse_as_summarization(const VerifiedData& entry) {
    std::vector<ParsedExample> examples;
    
    ParsedExample example;
    example.topic = extract_topic(entry.content);
    example.summary = generate_summary(entry.content);
    example.full_content = entry.content;
    example.source_url = entry.source_url;
    example.source_type = entry.source_type;
    example.author = entry.author;
    example.reliability_score = entry.reliability_score;
    example.timestamp = std::to_string(std::time(nullptr));
    example.metadata = entry.metadata;
    
    if (pImpl->config.extract_keywords) {
        example.keywords = extract_keywords(entry.content);
    }
    
    if (pImpl->config.extract_entities) {
        example.entities = extract_entities(entry.content);
    }
    
    example.input_text = "Summarize the following text:\n" + entry.content;
    example.target_text = example.summary;
    
    examples.push_back(example);
    
    return examples;
}

std::vector<ParsedExample> Parser::parse_as_completion(const VerifiedData& entry) {
    std::vector<ParsedExample> examples;
    
    auto sentences = pImpl->extract_sentences(entry.content);
    
    // Create completion examples
    for (size_t i = 0; i < sentences.size() - 1; ++i) {
        ParsedExample example;
        example.topic = extract_topic(entry.content);
        example.summary = generate_summary(entry.content);
        example.full_content = entry.content;
        example.source_url = entry.source_url;
        example.source_type = entry.source_type;
        example.author = entry.author;
        example.reliability_score = entry.reliability_score;
        example.timestamp = std::to_string(std::time(nullptr));
        example.metadata = entry.metadata;
        
        if (pImpl->config.extract_keywords) {
            example.keywords = extract_keywords(entry.content);
        }
        
        if (pImpl->config.extract_entities) {
            example.entities = extract_entities(entry.content);
        }
        
        // Use first N sentences as input, next sentence as target
        example.input_text = sentences[i];
        example.target_text = sentences[i + 1];
        
        examples.push_back(example);
        
        if (examples.size() >= 10) break;
    }
    
    return examples;
}

std::vector<ParsedExample> Parser::parse_as_instruction(const VerifiedData& entry) {
    std::vector<ParsedExample> examples;
    
    ParsedExample example;
    example.topic = extract_topic(entry.content);
    example.summary = generate_summary(entry.content);
    example.full_content = entry.content;
    example.source_url = entry.source_url;
    example.source_type = entry.source_type;
    example.author = entry.author;
    example.reliability_score = entry.reliability_score;
    example.timestamp = std::to_string(std::time(nullptr));
    example.metadata = entry.metadata;
    
    if (pImpl->config.extract_keywords) {
        example.keywords = extract_keywords(entry.content);
    }
    
    if (pImpl->config.extract_entities) {
        example.entities = extract_entities(entry.content);
    }
    
    // Format as instruction-following
    std::stringstream instruction;
    instruction << "### Instruction:\n";
    instruction << "Provide information about: " << example.topic << "\n\n";
    instruction << "### Response:\n";
    
    example.input_text = instruction.str();
    example.target_text = entry.content;
    
    examples.push_back(example);
    
    return examples;
}

std::vector<int> Parser::tokenize(const std::string& text) const {
    return pImpl->simple_tokenize(text);
}

std::string Parser::truncate_to_tokens(const std::string& text, size_t max_tokens) const {
    auto tokens = tokenize(text);
    
    if (tokens.size() <= max_tokens) {
        return text;
    }
    
    // Simple truncation: split by words and take first N
    std::istringstream iss(text);
    std::vector<std::string> words;
    std::string word;
    
    while (iss >> word && words.size() < max_tokens) {
        words.push_back(word);
    }
    
    std::stringstream result;
    for (const auto& w : words) {
        result << w << " ";
    }
    
    return result.str();
}

} // namespace training
} // namespace grim
