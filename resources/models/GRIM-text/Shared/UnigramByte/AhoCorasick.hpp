//======================================================//
//  AhoCorasick.hpp
//  Aho-Corasick DFA for Fast Multi-Pattern Matching
//  
//  Replaces std::regex for structural detection patterns.
//  Provides O(n) matching for multiple patterns simultaneously.
//  
//  Features:
//  - Fast multi-pattern matching in single pass
//  - No regex overhead (10-100x faster than std::regex)
//  - GPU-friendly DFA representation
//  - Cache-aligned transition tables
//  
//  Usage:
//    AhoCorasick ac;
//    ac.addPattern("http://", AtomType::ATOM_URL);
//    ac.addPattern("https://", AtomType::ATOM_URL);
//    ac.build();
//    
//    auto matches = ac.search(text);
//  
//  Author: GRIM Team
//  Date: December 2025
//======================================================//

#pragma once

#include "Unigram.hpp"  // For AtomType
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <queue>
#include <memory>
#include <ostream>

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  Match Result
//======================================================//

struct AhoCorasickMatch {
    size_t start;         // Start position in text
    size_t end;           // End position (exclusive)
    size_t length;        // Match length
    uint32_t pattern_id;  // Which pattern matched
    AtomType atom_type;   // Atom type for this pattern
    
    AhoCorasickMatch()
        : start(0), end(0), length(0), pattern_id(0), atom_type(AtomType::ATOM_IDENTIFIER) {}
    
    AhoCorasickMatch(size_t s, size_t e, uint32_t pid, AtomType type)
        : start(s), end(e), length(e - s), pattern_id(pid), atom_type(type) {}
};

//======================================================//
//  DFA Node
//======================================================//

struct AhoCorasickNode {
    // Transition table (char → next state)
    std::unordered_map<char, uint32_t> transitions;
    
    // Failure link (suffix link for mismatch)
    uint32_t failure_link;
    
    // Output patterns at this node
    std::vector<uint32_t> outputs;  // Pattern IDs that end here
    
    // Parent info (for building failure links)
    uint32_t parent;
    char parent_char;
    
    AhoCorasickNode()
        : failure_link(0), parent(0), parent_char('\0') {}
};

//======================================================//
//  Aho-Corasick Automaton
//======================================================//

class AhoCorasick {
public:
    AhoCorasick();
    ~AhoCorasick() = default;
    
    // Disable copy (expensive)
    AhoCorasick(const AhoCorasick&) = delete;
    AhoCorasick& operator=(const AhoCorasick&) = delete;
    
    // Move support
    AhoCorasick(AhoCorasick&&) noexcept = default;
    AhoCorasick& operator=(AhoCorasick&&) noexcept = default;
    
    //--------------------------------------------------//
    // Pattern Management
    //--------------------------------------------------//
    
    // Add pattern to match
    // Returns pattern ID
    uint32_t addPattern(std::string_view pattern, AtomType atom_type);

    // Configure ASCII case-insensitive matching (must be set before addPattern)
    void setCaseInsensitive(bool enabled);
    bool isCaseInsensitive() const { return case_insensitive_; }
    
    // Build DFA (must call after adding all patterns)
    void build();
    
    // Check if built
    bool isBuilt() const { return built_; }
    
    // Clear all patterns and reset
    void clear();
    
    //--------------------------------------------------//
    // Searching
    //--------------------------------------------------//
    
    // Find all matches in text
    std::vector<AhoCorasickMatch> search(std::string_view text) const;
    
    // Find first match only (faster)
    bool findFirst(std::string_view text, AhoCorasickMatch& out_match) const;
    
    // Check if text contains any pattern
    bool contains(std::string_view text) const;

    // Export DFA as Graphviz DOT (for visualization)
    bool saveDot(const std::string& path) const;
    bool writeDot(std::ostream& out) const;
    
    //--------------------------------------------------//
    // Statistics
    //--------------------------------------------------//
    
    size_t getPatternCount() const { return patterns_.size(); }
    size_t getNodeCount() const { return nodes_.size(); }
    size_t getMaxDepth() const;
    
    // Get pattern by ID
    const std::string& getPattern(uint32_t pattern_id) const;
    AtomType getPatternType(uint32_t pattern_id) const;
    
    //--------------------------------------------------//
    // GPU Transfer (for future CUDA kernels)
    //--------------------------------------------------//
    
    struct GPUDFAData {
        // Flattened transition table (cache-aligned)
        uint32_t* d_transitions;      // [num_nodes * 256] - full ASCII table per node
        uint32_t* d_failure_links;    // [num_nodes] - failure links
        uint32_t* d_output_counts;    // [num_nodes] - number of outputs per node
        uint32_t* d_output_lists;     // [total_outputs] - flattened output lists
        uint32_t* d_output_offsets;   // [num_nodes] - offset into output_lists
        uint32_t num_nodes;
        uint32_t total_outputs;
        
        GPUDFAData()
            : d_transitions(nullptr), d_failure_links(nullptr)
            , d_output_counts(nullptr), d_output_lists(nullptr)
            , d_output_offsets(nullptr), num_nodes(0), total_outputs(0) {}
    };
    
    // Upload DFA to GPU (for parallel matching)
    bool uploadToGPU(GPUDFAData& out_data) const;
    static void freeGPUData(GPUDFAData& data);
    
private:
    std::vector<AhoCorasickNode> nodes_;
    std::vector<std::string> patterns_;
    std::vector<AtomType> pattern_types_;
    bool built_;
    bool case_insensitive_;
    
    // Internal methods
    uint32_t createNode();
    void buildFailureLinks();
    uint32_t getTransition(uint32_t state, char ch) const;
    void computeOutputClosure();
};

//======================================================//
//  Common Pattern Sets
//======================================================//

class StructuralPatterns {
public:
    // Pre-built pattern sets for common structural atoms
    static AhoCorasick createURLPrefixes();        // http://, https://, ftp://, etc.
    static AhoCorasick createDateSeparators();     // -, /, . in date contexts
    static AhoCorasick createTimeSeparators();     // :, . in time contexts
    static AhoCorasick createNumberPrefixes();     // 0x, 0b, etc.
    static AhoCorasick createPathSeparators();     // /, \, etc.
    
    // All-in-one for maximum performance
    static AhoCorasick createAllStructuralPatterns();
};

} // namespace Tokenizer
} // namespace GRIM
