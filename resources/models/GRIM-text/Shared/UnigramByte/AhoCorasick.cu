//======================================================//
//  AhoCorasick.cu
//  Implementation of Aho-Corasick DFA
//======================================================//

#include "AhoCorasick.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace GRIM {
namespace Tokenizer {

namespace {

char toLowerAscii(char ch) {
    if (ch >= 'A' && ch <= 'Z') {
        return static_cast<char>(ch + ('a' - 'A'));
    }
    return ch;
}

std::string normalizePattern(std::string_view pattern, bool case_insensitive) {
    std::string out(pattern);
    if (!case_insensitive) {
        return out;
    }
    for (char& ch : out) {
        ch = toLowerAscii(ch);
    }
    return out;
}

std::string escapeDotLabel(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (ch < 32 || ch > 126) {
                    std::ostringstream oss;
                    oss << "0x" << std::hex << std::setw(2) << std::setfill('0')
                        << static_cast<int>(ch);
                    out += oss.str();
                } else {
                    out.push_back(static_cast<char>(ch));
                }
                break;
        }
    }
    return out;
}

std::string escapeDotChar(unsigned char ch) {
    if (ch == '\\') return "\\\\";
    if (ch == '"') return "\\\"";
    if (ch == '\t') return "\\t";
    if (ch == '\n') return "\\n";
    if (ch == '\r') return "\\r";
    if (ch >= 32 && ch <= 126) return std::string(1, static_cast<char>(ch));
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<int>(ch);
    return oss.str();
}

} // namespace

//======================================================//
//  AhoCorasick Implementation
//======================================================//

AhoCorasick::AhoCorasick()
    : built_(false)
    , case_insensitive_(false)
{
    // Create root node (state 0)
    nodes_.emplace_back();
}

void AhoCorasick::setCaseInsensitive(bool enabled) {
    if (!patterns_.empty() || built_) {
        throw std::runtime_error("setCaseInsensitive must be called before addPattern/build");
    }
    case_insensitive_ = enabled;
}

uint32_t AhoCorasick::addPattern(std::string_view pattern, AtomType atom_type) {
    if (built_) {
        throw std::runtime_error("Cannot add patterns after build()");
    }
    
    if (pattern.empty()) {
        throw std::invalid_argument("Pattern cannot be empty");
    }
    
    uint32_t pattern_id = static_cast<uint32_t>(patterns_.size());
    std::string normalized = normalizePattern(pattern, case_insensitive_);
    patterns_.emplace_back(normalized);
    pattern_types_.push_back(atom_type);
    
    // Insert pattern into trie
    uint32_t current = 0;  // Start at root
    
    for (char ch : normalized) {
        auto it = nodes_[current].transitions.find(ch);
        
        if (it == nodes_[current].transitions.end()) {
            // Create new node (may invalidate references!)
            uint32_t next_state = createNode();
            // Re-access node after potential reallocation
            nodes_[current].transitions[ch] = next_state;
            nodes_[next_state].parent = current;
            nodes_[next_state].parent_char = ch;
            current = next_state;
        } else {
            current = it->second;
        }
    }
    
    // Mark end of pattern
    nodes_[current].outputs.push_back(pattern_id);
    
    return pattern_id;
}

void AhoCorasick::build() {
    if (built_) {
        return;  // Already built
    }
    
    buildFailureLinks();
    computeOutputClosure();
    built_ = true;
}

void AhoCorasick::clear() {
    nodes_.clear();
    patterns_.clear();
    pattern_types_.clear();
    built_ = false;
    
    // Re-create root
    nodes_.emplace_back();
}

std::vector<AhoCorasickMatch> AhoCorasick::search(std::string_view text) const {
    if (!built_) {
        throw std::runtime_error("Must call build() before search()");
    }
    
    std::vector<AhoCorasickMatch> matches;
    matches.reserve(64);  // Pre-allocate for typical case
    
    uint32_t state = 0;
    
    for (size_t i = 0; i < text.size(); ++i) {
        char ch = text[i];
        
        // Follow transitions (with failure links)
        state = getTransition(state, ch);
        
        // Check for matches at this state
        const auto& outputs = nodes_[state].outputs;
        for (uint32_t pattern_id : outputs) {
            size_t pattern_len = patterns_[pattern_id].size();
            size_t start = i + 1 - pattern_len;
            
            matches.emplace_back(start, i + 1, pattern_id, pattern_types_[pattern_id]);
        }
    }
    
    return matches;
}

bool AhoCorasick::findFirst(std::string_view text, AhoCorasickMatch& out_match) const {
    if (!built_) {
        throw std::runtime_error("Must call build() before findFirst()");
    }
    
    uint32_t state = 0;
    
    for (size_t i = 0; i < text.size(); ++i) {
        char ch = text[i];
        state = getTransition(state, ch);
        
        const auto& outputs = nodes_[state].outputs;
        if (!outputs.empty()) {
            uint32_t pattern_id = outputs[0];
            size_t pattern_len = patterns_[pattern_id].size();
            size_t start = i + 1 - pattern_len;
            
            out_match = AhoCorasickMatch(start, i + 1, pattern_id, pattern_types_[pattern_id]);
            return true;
        }
    }
    
    return false;
}

bool AhoCorasick::contains(std::string_view text) const {
    AhoCorasickMatch dummy;
    return findFirst(text, dummy);
}

bool AhoCorasick::saveDot(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }
    return writeDot(file);
}

bool AhoCorasick::writeDot(std::ostream& out) const {
    if (!built_) {
        return false;
    }

    out << "digraph AhoCorasick {\n";
    out << "  rankdir=LR;\n";
    out << "  node [shape=circle,fontname=\"Consolas\",fontsize=10];\n";
    out << "  edge [fontname=\"Consolas\",fontsize=9];\n";

    for (size_t i = 0; i < nodes_.size(); ++i) {
        out << "  " << i << " [label=\"" << i;
        if (!nodes_[i].outputs.empty()) {
            out << "\\n";
            for (size_t j = 0; j < nodes_[i].outputs.size(); ++j) {
                uint32_t pid = nodes_[i].outputs[j];
                const std::string& pattern = patterns_[pid];
                out << pid << ":" << escapeDotLabel(pattern);
                if (j + 1 < nodes_[i].outputs.size()) {
                    out << ",";
                }
            }
        }
        out << "\"];\n";
    }

    for (size_t i = 0; i < nodes_.size(); ++i) {
        for (const auto& [ch, next_state] : nodes_[i].transitions) {
            out << "  " << i << " -> " << next_state
                << " [label=\"" << escapeDotChar(static_cast<unsigned char>(ch)) << "\"];\n";
        }
    }

    for (size_t i = 1; i < nodes_.size(); ++i) {
        out << "  " << i << " -> " << nodes_[i].failure_link
            << " [style=dashed,color=\"gray\",label=\"fail\"];\n";
    }

    out << "}\n";
    return out.good();
}

size_t AhoCorasick::getMaxDepth() const {
    size_t max_depth = 0;
    
    // BFS to find maximum depth
    std::queue<std::pair<uint32_t, size_t>> q;
    q.push({0, 0});
    
    while (!q.empty()) {
        auto [state, depth] = q.front();
        q.pop();
        
        max_depth = std::max(max_depth, depth);
        
        for (const auto& [ch, next_state] : nodes_[state].transitions) {
            q.push({next_state, depth + 1});
        }
    }
    
    return max_depth;
}

const std::string& AhoCorasick::getPattern(uint32_t pattern_id) const {
    if (pattern_id >= patterns_.size()) {
        throw std::out_of_range("Invalid pattern ID");
    }
    return patterns_[pattern_id];
}

AtomType AhoCorasick::getPatternType(uint32_t pattern_id) const {
    if (pattern_id >= pattern_types_.size()) {
        throw std::out_of_range("Invalid pattern ID");
    }
    return pattern_types_[pattern_id];
}

bool AhoCorasick::uploadToGPU(GPUDFAData& out_data) const {
    if (!built_) {
        return false;
    }
    
    const size_t num_nodes = nodes_.size();
    const size_t transition_table_size = num_nodes * 256;  // Full ASCII per node
    
    // Count total outputs
    size_t total_outputs = 0;
    for (const auto& node : nodes_) {
        total_outputs += node.outputs.size();
    }
    
    // Allocate GPU memory
    cudaError_t err;
    
    err = cudaMalloc(&out_data.d_transitions, transition_table_size * sizeof(uint32_t));
    if (err != cudaSuccess) return false;
    
    err = cudaMalloc(&out_data.d_failure_links, num_nodes * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(out_data.d_transitions);
        return false;
    }
    
    err = cudaMalloc(&out_data.d_output_counts, num_nodes * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(out_data.d_transitions);
        cudaFree(out_data.d_failure_links);
        return false;
    }
    
    err = cudaMalloc(&out_data.d_output_lists, total_outputs * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(out_data.d_transitions);
        cudaFree(out_data.d_failure_links);
        cudaFree(out_data.d_output_counts);
        return false;
    }
    
    err = cudaMalloc(&out_data.d_output_offsets, num_nodes * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(out_data.d_transitions);
        cudaFree(out_data.d_failure_links);
        cudaFree(out_data.d_output_counts);
        cudaFree(out_data.d_output_lists);
        return false;
    }
    
    // Pack data on host
    std::vector<uint32_t> h_transitions(transition_table_size, 0);
    std::vector<uint32_t> h_failure_links(num_nodes);
    std::vector<uint32_t> h_output_counts(num_nodes);
    std::vector<uint32_t> h_output_lists;
    std::vector<uint32_t> h_output_offsets(num_nodes);
    
    h_output_lists.reserve(total_outputs);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        const auto& node = nodes_[i];
        
        // Pack transitions (256 entries per node)
        for (const auto& [ch, next] : node.transitions) {
            h_transitions[i * 256 + static_cast<unsigned char>(ch)] = next;
        }
        
        // Failure links
        h_failure_links[i] = node.failure_link;
        
        // Outputs
        h_output_counts[i] = static_cast<uint32_t>(node.outputs.size());
        h_output_offsets[i] = static_cast<uint32_t>(h_output_lists.size());
        
        for (uint32_t output_id : node.outputs) {
            h_output_lists.push_back(output_id);
        }
    }
    
    // Upload to GPU
    cudaMemcpy(out_data.d_transitions, h_transitions.data(),
               transition_table_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(out_data.d_failure_links, h_failure_links.data(),
               num_nodes * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(out_data.d_output_counts, h_output_counts.data(),
               num_nodes * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(out_data.d_output_lists, h_output_lists.data(),
               total_outputs * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(out_data.d_output_offsets, h_output_offsets.data(),
               num_nodes * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    out_data.num_nodes = static_cast<uint32_t>(num_nodes);
    out_data.total_outputs = static_cast<uint32_t>(total_outputs);
    
    return cudaGetLastError() == cudaSuccess;
}

void AhoCorasick::freeGPUData(GPUDFAData& data) {
    if (data.d_transitions) cudaFree(data.d_transitions);
    if (data.d_failure_links) cudaFree(data.d_failure_links);
    if (data.d_output_counts) cudaFree(data.d_output_counts);
    if (data.d_output_lists) cudaFree(data.d_output_lists);
    if (data.d_output_offsets) cudaFree(data.d_output_offsets);
    
    data = GPUDFAData{};
}

uint32_t AhoCorasick::createNode() {
    uint32_t id = static_cast<uint32_t>(nodes_.size());
    nodes_.emplace_back();
    return id;
}

void AhoCorasick::buildFailureLinks() {
    // BFS to build failure links
    std::queue<uint32_t> q;
    
    // Root failure link points to itself
    nodes_[0].failure_link = 0;
    
    // Depth-1 nodes fail to root
    for (const auto& [ch, child] : nodes_[0].transitions) {
        nodes_[child].failure_link = 0;
        q.push(child);
    }
    
    // BFS for deeper nodes
    while (!q.empty()) {
        uint32_t state = q.front();
        q.pop();
        
        for (const auto& [ch, child] : nodes_[state].transitions) {
            // Find failure link for child
            uint32_t failure = nodes_[state].failure_link;
            
            while (failure != 0 && nodes_[failure].transitions.find(ch) == nodes_[failure].transitions.end()) {
                failure = nodes_[failure].failure_link;
            }
            
            auto it = nodes_[failure].transitions.find(ch);
            if (it != nodes_[failure].transitions.end() && it->second != child) {
                nodes_[child].failure_link = it->second;
            } else {
                nodes_[child].failure_link = 0;
            }
            
            q.push(child);
        }
    }
}

uint32_t AhoCorasick::getTransition(uint32_t state, char ch) const {
    const char key = case_insensitive_ ? toLowerAscii(ch) : ch;
    // Follow transitions with failure links
    while (state != 0) {
        auto it = nodes_[state].transitions.find(key);
        if (it != nodes_[state].transitions.end()) {
            return it->second;
        }
        state = nodes_[state].failure_link;
    }
    
    // Check root
    auto it = nodes_[0].transitions.find(key);
    return (it != nodes_[0].transitions.end()) ? it->second : 0;
}

void AhoCorasick::computeOutputClosure() {
    // Add outputs from failure links in BFS order so failure outputs are already closed.
    std::queue<uint32_t> q;
    q.push(0);

    while (!q.empty()) {
        uint32_t state = q.front();
        q.pop();

        if (state != 0) {
            uint32_t failure = nodes_[state].failure_link;
            const auto& failure_outputs = nodes_[failure].outputs;
            nodes_[state].outputs.insert(nodes_[state].outputs.end(),
                                         failure_outputs.begin(),
                                         failure_outputs.end());
        }

        for (const auto& [ch, next_state] : nodes_[state].transitions) {
            q.push(next_state);
        }
    }
}

//======================================================//
//  StructuralPatterns Implementation
//======================================================//

AhoCorasick StructuralPatterns::createURLPrefixes() {
    AhoCorasick ac;
    ac.build();
    return ac;
}

AhoCorasick StructuralPatterns::createDateSeparators() {
    AhoCorasick ac;
    ac.build();
    return ac;
}

AhoCorasick StructuralPatterns::createTimeSeparators() {
    AhoCorasick ac;
    ac.build();
    return ac;
}

AhoCorasick StructuralPatterns::createNumberPrefixes() {
    AhoCorasick ac;
    ac.addPattern("0x", AtomType::ATOM_NUM);
    ac.addPattern("0X", AtomType::ATOM_NUM);
    ac.addPattern("0b", AtomType::ATOM_NUM);
    ac.addPattern("0B", AtomType::ATOM_NUM);
    ac.build();
    return ac;
}

AhoCorasick StructuralPatterns::createPathSeparators() {
    AhoCorasick ac;
    ac.build();
    return ac;
}

AhoCorasick StructuralPatterns::createAllStructuralPatterns() {
    AhoCorasick ac;
    ac.addPattern("0x", AtomType::ATOM_NUM);
    ac.addPattern("0X", AtomType::ATOM_NUM);
    ac.addPattern("0b", AtomType::ATOM_NUM);
    ac.addPattern("0B", AtomType::ATOM_NUM);
    ac.build();
    return ac;
}

} // namespace Tokenizer
} // namespace GRIM
