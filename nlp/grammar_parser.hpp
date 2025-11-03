#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <regex>
#include <nlohmann/json.hpp>

namespace GRIM {

// Grammar component definition
struct GrammarComponent {
    std::string name;
    std::string description;
    std::vector<std::string> patterns;
    bool optional = false;
    bool capture = false;
    bool requiresContext = false;
};

// Command verb definition
struct CommandVerb {
    std::string canonical;
    std::vector<std::string> synonyms;
    std::string intent;
    bool requiresObject = false;
};

// Command object definition
struct CommandObject {
    std::string canonical;
    std::vector<std::string> synonyms;
    std::string category;
};

// Sentence template
struct SentenceTemplate {
    std::string name;
    std::string structure;  // e.g., "[greeting] <verb> [article] <object>"
    std::vector<std::string> examples;
    bool requiresContext = false;
    std::string category;
};

// Parse result
struct GrammarParseResult {
    bool matched = false;
    std::string intent;
    std::unordered_map<std::string, std::string> slots;
 double confidence = 0.0;
    std::string matchedTemplate;
    std::vector<std::string> matchedComponents;
    
    // For multi-command support
    std::vector<GrammarParseResult> subCommands;
};

// Grammar-based NLP parser
class GrammarParser {
public:
    GrammarParser();
    ~GrammarParser() = default;
    
    // Load grammar from JSON file
    bool load(const std::string& path);
    
    // Load grammar from FlatBuffer binary file
    bool loadBinary(const std::string& path);
    
    // Parse input using grammar rules
    GrammarParseResult parse(const std::string& input);
    
    // Parse with context awareness
    GrammarParseResult parseWithContext(
        const std::string& input,
     const std::unordered_map<std::string, std::string>& context
    );
    
    // Add custom verb/synonym mapping
    bool addVerb(const std::string& verb, const std::string& intent, 
      const std::vector<std::string>& synonyms = {});
    
    // Add custom object/synonym mapping
    bool addObject(const std::string& object, const std::string& category,
 const std::vector<std::string>& synonyms = {});
    
    // Get statistics
    nlohmann::json getStats() const;
    
private:
    // Grammar data
    std::unordered_map<std::string, GrammarComponent> components;
    std::unordered_map<std::string, CommandVerb> verbs;
    std::unordered_map<std::string, CommandObject> objects;
    std::vector<SentenceTemplate> templates;
    nlohmann::json contextRules;
    nlohmann::json learningConfig;
  
    // Parsing helpers
    std::optional<std::string> matchComponent(
      const std::string& input,
        const GrammarComponent& component,
        size_t& position
    );
    
    std::optional<std::string> matchVerb(
  const std::string& input,
 size_t& position,
        CommandVerb& outVerb
    );

    std::optional<std::string> matchObject(
        const std::string& input,
size_t& position,
        CommandObject& outObject
    );
  
    bool matchTemplate(
 const std::string& input,
  const SentenceTemplate& tmpl,
        GrammarParseResult& result
    );
    
    std::vector<std::string> tokenize(const std::string& input);
    
    double calculateConfidence(const GrammarParseResult& result);
    
 // Multi-command detection
std::vector<std::string> splitCommands(const std::string& input);
    
    // Fuzzy matching
    double levenshteinSimilarity(const std::string& a, const std::string& b);
    std::optional<std::string> fuzzyMatchVerb(const std::string& token);
    std::optional<std::string> fuzzyMatchObject(const std::string& token);
    
  // Statistics
    struct {
        int totalParses = 0;
        int successfulParses = 0;
        int failedParses = 0;
  int multiCommandParses = 0;
        int contextualParses = 0;
        int fuzzyMatches = 0;
    } stats;
};

} // namespace GRIM
