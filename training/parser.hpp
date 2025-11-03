#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace grim {
namespace training {

/**
 * @brief Verified entry ready for parsing
 */
struct VerifiedData {
    std::string content;
    std::string source_url;
    std::string source_type;
    std::string author;
    double reliability_score;
    std::string verification_method;
    std::vector<std::string> cross_references;
    std::string metadata;
};

/**
 * @brief Parsed training example ready for model training
 */
struct ParsedExample {
    std::string topic;
    std::string summary;
    std::string full_content;
    std::string source_url;
    std::string source_type;
    std::string author;
    double reliability_score;
    std::vector<std::string> keywords;
    std::vector<std::string> entities;  // Named entities
    std::string timestamp;
    std::string metadata;
    
    // Training-specific fields
    std::string input_text;   // What goes into the model
    std::string target_text;  // Expected output (for supervised learning)
    std::vector<float> embedding;  // Optional pre-computed embedding
};

/**
 * @brief Configuration for the parsing process
 */
struct ParserConfig {
    std::string input_dir = "data/verified";
    std::string output_dir = "data/parsed";
    std::string tokenizer_path = "";  // Path to tokenizer model
    size_t max_token_length = 2048;
    bool extract_entities = true;
    bool extract_keywords = true;
    bool compute_embeddings = false;
    bool save_as_flatbuffer = false;  // If true, use FlatBuffer; else JSONL
    
    // Parsing strategy
    enum class ParseStrategy {
        QUESTION_ANSWER,  // Generate Q&A pairs
        SUMMARIZATION,    // Extract summaries
        COMPLETION,       // Text completion tasks
        INSTRUCTION       // Instruction-following format
    };
    ParseStrategy strategy = ParseStrategy::INSTRUCTION;
};

/**
 * @brief Stage 3: Data Parser
 * 
 * Converts verified text into structured, model-trainable examples.
 * Extracts key fields, formats records, and prepares data for fine-tuning.
 */
class Parser {
public:
    Parser();
    explicit Parser(const ParserConfig& config);
    ~Parser();

    /**
     * @brief Main entry point: Parse verified data into training examples
     * 
     * Extracts key fields (topic, summary, source, timestamp, reliability),
     * formats each record as JSONL or FlatBuffer message, and runs tokenizer
     * sanity checks.
     * 
     * @return Number of examples successfully parsed
     */
    size_t parse_verified_data();

    /**
     * @brief Parse a single verified entry
     * 
     * @param entry Verified data entry
     * @return Vector of parsed examples (may generate multiple from one entry)
     */
    std::vector<ParsedExample> parse_entry(const VerifiedData& entry);

    /**
     * @brief Load tokenizer for validation
     */
    bool load_tokenizer(const std::string& tokenizer_path);

    /**
     * @brief Validate that parsed text is tokenizable
     */
    bool validate_tokenization(const ParsedExample& example);

    /**
     * @brief Set parsing strategy
     */
    void set_strategy(ParserConfig::ParseStrategy strategy);

    /**
     * @brief Get parsing statistics
     */
    struct Stats {
        size_t total_verified_entries = 0;
        size_t total_parsed_examples = 0;
        size_t tokenization_failures = 0;
        size_t entity_extraction_count = 0;
        size_t keyword_extraction_count = 0;
    };
    Stats get_stats() const;

    /**
     * @brief Reset statistics
     */
    void reset_stats();

private:
    /**
     * @brief Load verified entries from input directory
     */
    std::vector<VerifiedData> load_verified_entries() const;

    /**
     * @brief Save parsed examples to output file
     */
    bool save_parsed_examples(const std::vector<ParsedExample>& examples) const;

    /**
     * @brief Extract topic from content
     */
    std::string extract_topic(const std::string& content) const;

    /**
     * @brief Generate summary from content
     */
    std::string generate_summary(const std::string& content) const;

    /**
     * @brief Extract keywords from content
     */
    std::vector<std::string> extract_keywords(const std::string& content) const;

    /**
     * @brief Extract named entities from content
     */
    std::vector<std::string> extract_entities(const std::string& content) const;

    /**
     * @brief Parse using Question-Answer strategy
     */
    std::vector<ParsedExample> parse_as_qa(const VerifiedData& entry);

    /**
     * @brief Parse using Summarization strategy
     */
    std::vector<ParsedExample> parse_as_summarization(const VerifiedData& entry);

    /**
     * @brief Parse using Completion strategy
     */
    std::vector<ParsedExample> parse_as_completion(const VerifiedData& entry);

    /**
     * @brief Parse using Instruction-following strategy
     */
    std::vector<ParsedExample> parse_as_instruction(const VerifiedData& entry);

    /**
     * @brief Tokenize text and check length
     */
    std::vector<int> tokenize(const std::string& text) const;

    /**
     * @brief Truncate text to fit within token limit
     */
    std::string truncate_to_tokens(const std::string& text, size_t max_tokens) const;

    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace training
} // namespace grim
