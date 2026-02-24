//======================================================//
//  AtomTable.cu
//  CUDA implementation of Atom Registry
//======================================================//

#include "AtomTable.hpp"
#include "UniByte.hpp"  // For StructuralSpan

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string_view>

// NOTE: NO std::regex - banned for performance reasons

namespace GRIM {
namespace Tokenizer {

namespace {

const char* atomCategoryName(AtomCategory category) {
    switch (category) {
        case AtomCategory::NUMERIC: return "NUMERIC";
        case AtomCategory::TEMPORAL: return "TEMPORAL";
        case AtomCategory::STRUCTURAL: return "STRUCTURAL";
        case AtomCategory::STRING: return "STRING";
        case AtomCategory::IDENTIFIER: return "IDENTIFIER";
        case AtomCategory::SYSTEM: return "SYSTEM";
        case AtomCategory::GENERIC: return "GENERIC";
        default: return "UNKNOWN";
    }
}

const char* atomOriginName(AtomOrigin origin) {
    switch (origin) {
        case AtomOrigin::USER_INPUT: return "USER_INPUT";
        case AtomOrigin::MODEL_GENERATED: return "MODEL_GENERATED";
        case AtomOrigin::SYSTEM: return "SYSTEM";
        case AtomOrigin::TOOL_DERIVED: return "TOOL_DERIVED";
        case AtomOrigin::UNKNOWN: return "UNKNOWN";
        default: return "UNKNOWN";
    }
}

std::string escapeForTsv(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (char c : value) {
        switch (c) {
            case '\t': out += "\\t"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

} // namespace

//======================================================//
//  CUDA Kernels
//======================================================//

// Kernel: Unpack atom numeric values for computation
__global__ void kernelUnpackAtomNumerics(
    const float* __restrict__ numeric_values,
    const uint32_t* __restrict__ types,
    size_t num_atoms,
    float* __restrict__ output_floats,
    int64_t* __restrict__ output_ints,
    bool* __restrict__ is_integer
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_atoms) return;
    
    uint32_t type = types[idx];
    float val = numeric_values[idx];
    
    // Check if integer type
    bool is_int = (type == static_cast<uint32_t>(AtomType::ATOM_INTEGER) ||
                   type == static_cast<uint32_t>(AtomType::ATOM_HEX) ||
                   type == static_cast<uint32_t>(AtomType::ATOM_BINARY));
    
    is_integer[idx] = is_int;
    output_floats[idx] = val;
    output_ints[idx] = static_cast<int64_t>(val);
}

// Kernel: Pack atom data for embedding lookup
__global__ void kernelPackAtomEmbeddings(
    const float* __restrict__ numeric_values,
    const uint32_t* __restrict__ types,
    const uint32_t* __restrict__ flags,
    size_t num_atoms,
    int embedding_dim,
    float* __restrict__ output_embeddings  // [num_atoms, embedding_dim]
) {
    const size_t atom_idx = blockIdx.x;
    const size_t dim_idx = threadIdx.x;
    
    if (atom_idx >= num_atoms || dim_idx >= embedding_dim) return;
    
    float value = 0.0f;
    
    // First few dimensions encode the atom type
    if (dim_idx < 16) {
        uint32_t type = types[atom_idx];
        value = (type == dim_idx) ? 1.0f : 0.0f;
    }
    // Next dimensions encode numeric value (log-scaled)
    else if (dim_idx < 32) {
        float num_val = numeric_values[atom_idx];
        int bit = dim_idx - 16;
        
        if (num_val != 0.0f) {
            // Encode sign
            if (bit == 0) {
                value = (num_val < 0) ? 1.0f : 0.0f;
            }
            // Encode magnitude in log scale
            else {
                float log_val = log2f(fabsf(num_val) + 1.0f);
                value = fmodf(log_val / (float)bit, 1.0f);
            }
        }
    }
    // Remaining dimensions from flags
    else if (dim_idx < 64) {
        uint32_t flag_bit = dim_idx - 32;
        value = ((flags[atom_idx] >> flag_bit) & 1) ? 1.0f : 0.0f;
    }
    
    output_embeddings[atom_idx * embedding_dim + dim_idx] = value;
}

//======================================================//
//  AtomTable Implementation
//======================================================//

AtomTable::AtomTable() {
    entries_.reserve(64);
    string_pool_.reserve(4096);  // Start with 4KB string pool
    hash_to_id_.reserve(64);
}

AtomTable::~AtomTable() = default;

AtomTable::AtomTable(AtomTable&& other) noexcept
    : entries_(std::move(other.entries_))
    , string_pool_(std::move(other.string_pool_))
    , hash_to_id_(std::move(other.hash_to_id_))
    , type_index_(std::move(other.type_index_))
    , next_id_(other.next_id_)
    , dedup_hits_(other.dedup_hits_)
    , total_queries_(other.total_queries_)
    , pending_gpu_upload_(std::move(other.pending_gpu_upload_))
    , gpu_dirty_(other.gpu_dirty_)
{
    other.next_id_ = 0;
    other.dedup_hits_ = 0;
    other.total_queries_ = 0;
    other.gpu_dirty_ = false;
}

AtomTable& AtomTable::operator=(AtomTable&& other) noexcept {
    if (this != &other) {
        entries_ = std::move(other.entries_);
        string_pool_ = std::move(other.string_pool_);
        hash_to_id_ = std::move(other.hash_to_id_);
        type_index_ = std::move(other.type_index_);
        pending_gpu_upload_ = std::move(other.pending_gpu_upload_);
        next_id_ = other.next_id_;
        dedup_hits_ = other.dedup_hits_;
        total_queries_ = other.total_queries_;
        gpu_dirty_ = other.gpu_dirty_;
        
        other.next_id_ = 0;
        other.dedup_hits_ = 0;
        other.total_queries_ = 0;
        other.gpu_dirty_ = false;
    }
    return *this;
}

void AtomTable::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    string_pool_.clear();
    hash_to_id_.clear();
    type_index_.clear();
    pending_gpu_upload_.clear();
    next_id_ = 0;
    dedup_hits_ = 0;
    total_queries_ = 0;
    gpu_dirty_ = false;
}

void AtomTable::reserve(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.reserve(count);
    hash_to_id_.reserve(count);
}

bool AtomTable::saveToFile(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    const char magic[4] = {'A', 'T', 'M', 'B'};
    const uint32_t version = 1;
    const uint32_t entry_count = static_cast<uint32_t>(entries_.size());
    const uint32_t pool_size = static_cast<uint32_t>(string_pool_.size());

    file.write(magic, sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&entry_count), sizeof(entry_count));
    file.write(reinterpret_cast<const char*>(&pool_size), sizeof(pool_size));

    if (entry_count > 0) {
        file.write(reinterpret_cast<const char*>(entries_.data()),
                   entry_count * sizeof(AtomEntry));
    }
    if (pool_size > 0) {
        file.write(reinterpret_cast<const char*>(string_pool_.data()),
                   pool_size * sizeof(char));
    }

    return file.good();
}

bool AtomTable::saveToTextFile(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    file << "id\t"
         << "type\t"
         << "category\t"
         << "origin\t"
         << "raw_len\t"
         << "parsed_len\t"
         << "raw_text\t"
         << "parsed_text\t"
         << "numeric_value\t"
         << "flags\t"
         << "confidence\t"
         << "created_at\t"
         << "source_start\t"
         << "source_end\t"
         << "hash"
         << "\n";

    for (const auto& entry : entries_) {
        std::string_view raw = getString(entry.raw_text_ref);
        std::string_view parsed = getString(entry.parsed_ref);

        std::ostringstream hash_stream;
        hash_stream << "0x" << std::hex << entry.hash;

        file << entry.id << "\t"
             << atomTypeName(entry.type) << "\t"
             << atomCategoryName(entry.category) << "\t"
             << atomOriginName(entry.origin) << "\t"
             << entry.raw_text_ref.length << "\t"
             << entry.parsed_ref.length << "\t"
             << escapeForTsv(raw) << "\t"
             << escapeForTsv(parsed) << "\t"
             << entry.numeric_value << "\t"
             << entry.flags << "\t"
             << entry.confidence << "\t"
             << entry.created_at << "\t"
             << entry.source_start << "\t"
             << entry.source_end << "\t"
             << hash_stream.str()
             << "\n";
    }

    return file.good();
}

bool AtomTable::loadFromFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    char magic[4] = {0};
    file.read(magic, sizeof(magic));
    if (file.gcount() != sizeof(magic) ||
        magic[0] != 'A' || magic[1] != 'T' || magic[2] != 'M' || magic[3] != 'B') {
        return false;
    }

    uint32_t version = 0;
    uint32_t entry_count = 0;
    uint32_t pool_size = 0;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&entry_count), sizeof(entry_count));
    file.read(reinterpret_cast<char*>(&pool_size), sizeof(pool_size));

    if (!file.good() || version != 1) {
        return false;
    }
    // Sanity check: reject obviously corrupt files (> 10M entries)
    if (entry_count > 10000000) {
        return false;
    }

    std::vector<AtomEntry> entries(entry_count);
    std::vector<char> pool(pool_size);

    if (entry_count > 0) {
        file.read(reinterpret_cast<char*>(entries.data()),
                  entry_count * sizeof(AtomEntry));
    }
    if (pool_size > 0) {
        file.read(reinterpret_cast<char*>(pool.data()),
                  pool_size * sizeof(char));
    }
    if (!file.good()) {
        return false;
    }

    for (uint32_t i = 0; i < entry_count; ++i) {
        const AtomEntry& entry = entries[i];
        if (entry.id < ATOM_TOKEN_BASE || entry.id >= ATOM_TOKEN_MAX) {
            return false;
        }
        const uint32_t expected_idx = entry.id - ATOM_TOKEN_BASE;
        if (expected_idx != i) {
            return false;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        freeGPUData(gpu_data_);

        entries_ = std::move(entries);
        string_pool_ = std::move(pool);
        hash_to_id_.clear();
        type_index_.clear();
        pending_gpu_upload_.clear();
        dedup_hits_ = 0;
        total_queries_ = 0;

        hash_to_id_.reserve(entries_.size());
        pending_gpu_upload_.reserve(entries_.size());
        for (const auto& entry : entries_) {
            hash_to_id_[entry.hash] = entry.id;
            type_index_[entry.type].push_back(entry.id);
            pending_gpu_upload_.push_back(entry.id);
        }

        next_id_ = static_cast<uint32_t>(entries_.size());
        gpu_dirty_ = !entries_.empty();
    }

    return true;
}

bool AtomTable::computeFileHash(const std::string& path, uint64_t& out_hash) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
    constexpr uint64_t kFnvPrime = 1099511628211ULL;
    out_hash = kFnvOffset;

    std::vector<char> buffer(64 * 1024);
    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        std::streamsize count = file.gcount();
        for (std::streamsize i = 0; i < count; ++i) {
            out_hash ^= static_cast<uint64_t>(static_cast<uint8_t>(buffer[i]));
            out_hash *= kFnvPrime;
        }
    }

    return !file.bad();
}

//--------------------------------------------------//
// String Pool Implementation
//--------------------------------------------------//

StringRef AtomTable::internString(const std::string& str) {
    return internString(str.data(), str.size());
}

StringRef AtomTable::internString(std::string_view sv) {
    return internString(sv.data(), sv.size());
}

StringRef AtomTable::internString(const char* data, size_t length) {
    if (length == 0) return StringRef(0, 0);
    
    uint32_t offset = static_cast<uint32_t>(string_pool_.size());
    uint32_t len = static_cast<uint32_t>(length);
    
    // Append to pool
    string_pool_.insert(string_pool_.end(), data, data + length);
    
    return StringRef(offset, len);
}

std::string_view AtomTable::getString(const StringRef& ref) const {
    if (ref.length == 0) return std::string_view();
    return std::string_view(string_pool_.data() + ref.offset, ref.length);
}

size_t AtomTable::getDeduplicationHitRate() const {
    if (total_queries_ == 0) return 0;
    return (dedup_hits_ * 100) / total_queries_;
}

//--------------------------------------------------//
// Deduplication
//--------------------------------------------------//

uint32_t AtomTable::findExisting(uint64_t hash, std::string_view raw_text) {
    total_queries_++;
    
    auto it = hash_to_id_.find(hash);
    if (it == hash_to_id_.end()) {
        return UINT32_MAX;  // Not found
    }
    
    // The stored value is the token ID (ATOM_TOKEN_BASE+), get index in entries_
    uint32_t token_id = it->second;
    uint32_t entry_idx = token_id - ATOM_TOKEN_BASE;
    
    if (entry_idx >= entries_.size()) {
        return UINT32_MAX;
    }
    
    const AtomEntry& existing = entries_[entry_idx];
    std::string_view existing_text = getString(existing.raw_text_ref);
    
    if (existing_text == raw_text) {
        dedup_hits_++;
        return token_id;  // Return the token ID (ATOM_TOKEN_BASE+), not the array index
    }
    
    return UINT32_MAX;  // Hash collision but different content
}

//--------------------------------------------------//
// Helper Functions
//--------------------------------------------------//

namespace {
// Check if atom type needs separate parsed representation
// (Most types don't - raw_text is sufficient)
bool hasEdgeWhitespace(std::string_view text) {
    if (text.empty()) {
        return false;
    }
    const unsigned char first = static_cast<unsigned char>(text.front());
    const unsigned char last = static_cast<unsigned char>(text.back());
    return std::isspace(first) || std::isspace(last);
}

std::string_view trimWhitespace(std::string_view text) {
    size_t start = 0;
    size_t end = text.size();
    while (start < end && std::isspace(static_cast<unsigned char>(text[start]))) {
        ++start;
    }
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(start, end - start);
}

bool needsSerialization(AtomType type) {
    switch (type) {
        // These types benefit from normalized serialization
        case AtomType::ATOM_URL:           // Normalize scheme/host
        case AtomType::ATOM_EMAIL:         // Normalize case
        case AtomType::ATOM_STRING_LITERAL: // Unescape sequences
            return true;
        
        // These types: raw_text == parsed representation
        case AtomType::ATOM_INTEGER:
        case AtomType::ATOM_FLOAT:
        case AtomType::ATOM_HEX:
        case AtomType::ATOM_BINARY:
        case AtomType::ATOM_DATE:
        case AtomType::ATOM_TIME:
        case AtomType::ATOM_IP_ADDRESS:
        case AtomType::ATOM_PATH:
        case AtomType::ATOM_IDENTIFIER:
        default:
            return false;  // Skip serialization - use raw_text
    }
}
} // anonymous namespace

//--------------------------------------------------//
// Registration
//--------------------------------------------------//

uint32_t AtomTable::registerAtom(AtomType type, 
                                  std::string_view raw_text,
                                  size_t source_start,
                                  size_t source_end) {
    // Parse the atom (parseAtom still needs std::string for now)
    std::string raw_text_str(raw_text);
    auto result = parseAtom(type, raw_text_str);
    
    AtomValue parsed;
    if (result.success) {
        parsed = result.value;
    } else {
        // Fallback to generic (convert string_view to string)
        parsed = AtomGeneric{std::string(raw_text)};
    }
    
    return registerAtom(type, raw_text, parsed, source_start, source_end);
}

uint32_t AtomTable::registerAtom(AtomType type,
                                  std::string_view raw_text,
                                  const AtomValue& parsed_value,
                                  size_t source_start,
                                  size_t source_end) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Compute hash for deduplication
    uint64_t hash = computeHash(type, raw_text);
    
    // Check if this atom already exists
    uint32_t existing_id = findExisting(hash, raw_text);
    if (existing_id != UINT32_MAX) {
        return existing_id;  // Return existing ID (deduplication hit!)
    }
    
    // New atom - allocate entry
    // Entry IDs start at ATOM_TOKEN_BASE (side-channel index, not model token vocab)
    AtomEntry entry;
    entry.id = ATOM_TOKEN_BASE + next_id_;
    next_id_++;
    
    entry.type = type;
    entry.hash = hash;
    entry.source_start = static_cast<uint32_t>(source_start);
    entry.source_end = static_cast<uint32_t>(source_end);
    
    // Initialize metadata
    entry.origin = AtomOrigin::USER_INPUT;
    entry.category = getCategoryForType(type);
    entry.confidence = 1.0f;
    entry.created_at = getCurrentTimestamp();
    
    // Intern raw text into string pool
    entry.raw_text_ref = internString(raw_text);
    
    // Pack numeric value (no string copying for GPU)
    packNumericValue(entry, parsed_value);
    
    // Skip serialization for types where raw_text == parsed representation
    if (needsSerialization(type) && !hasEdgeWhitespace(raw_text)) {
        // Use stack buffer to avoid heap allocation
        char buffer[1024];
        size_t len = atomValueSerialize(type, parsed_value, buffer, sizeof(buffer));
        if (len > 0 && std::string_view(buffer, len) != raw_text) {
            entry.parsed_ref = internString(buffer, len);
        }
    }
    
    // Index for fast lookup
    uint32_t new_id = entry.id;
    hash_to_id_[hash] = new_id;
    type_index_[type].push_back(new_id);
    
    // Mark for GPU upload
    pending_gpu_upload_.push_back(new_id);
    gpu_dirty_ = true;
    
    entries_.push_back(std::move(entry));
    return new_id;
}

// Convenience overload: AtomValue first, then raw_text (used by tests)
uint32_t AtomTable::registerAtom(AtomType type,
                                  const AtomValue& parsed_value,
                                  std::string_view raw_text,
                                  size_t source_start,
                                  size_t source_end) {
    // Delegate to the existing implementation with swapped parameters
    return registerAtom(type, raw_text, parsed_value, source_start, source_end);
}

bool AtomTable::tryRegisterSpan(const StructuralSpan& span, uint32_t& out_id) {
    // Zero-copy: pass buffer pointer + length directly to string pool!
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Get string_view from span (no allocation)
    std::string_view raw_text = span.view();
    
    // Compute hash for deduplication
    uint64_t hash = computeHash(span.atom_type, raw_text);
    
    // Check if this atom already exists
    uint32_t existing_id = findExisting(hash, raw_text);
    if (existing_id != UINT32_MAX) {
        out_id = existing_id;  // Return existing ID (deduplication hit!)
        return true;
    }
    
    // New atom - allocate entry
    
    // Parse the atom (still needs std::string temporarily)
    std::string raw_text_str(raw_text);
    auto result = parseAtom(span.atom_type, raw_text_str);
    
    AtomValue parsed;
    if (result.success) {
        parsed = result.value;
    } else {
        parsed = AtomGeneric{raw_text_str};
    }
    
    AtomEntry entry;
    entry.id = ATOM_TOKEN_BASE + next_id_;
    next_id_++;
    entry.type = span.atom_type;
    entry.hash = hash;
    entry.source_start = static_cast<uint32_t>(span.start);
    entry.source_end = static_cast<uint32_t>(span.end);
    
    // Initialize metadata
    entry.origin = AtomOrigin::USER_INPUT;
    entry.category = getCategoryForType(span.atom_type);
    entry.confidence = 1.0f;
    entry.created_at = getCurrentTimestamp();
    
    // ZERO-COPY INTERNING: Pass buffer pointer + length directly!
    entry.raw_text_ref = internString(span.buffer_ptr + span.offset, span.length);
    
    // Pack numeric value
    packNumericValue(entry, parsed);
    
    // Skip serialization for types where raw_text == parsed representation
    // Only serialize if we need a different string representation
    if (needsSerialization(span.atom_type) && !hasEdgeWhitespace(raw_text)) {
        std::string serialized = atomValueToString(span.atom_type, parsed);
        if (!serialized.empty()) {
            std::string_view raw = span.view();
            if (serialized != raw) {
                entry.parsed_ref = internString(serialized);
            }
        }
    }
    
    // Index for fast lookup
    uint32_t new_id = entry.id;
    hash_to_id_[hash] = new_id;
    type_index_[span.atom_type].push_back(new_id);
    
    // Mark for GPU upload
    pending_gpu_upload_.push_back(new_id);
    gpu_dirty_ = true;
    
    entries_.push_back(std::move(entry));
    out_id = new_id;
    return true;
}

uint32_t AtomTable::registerSpan(const StructuralSpan& span) {
    uint32_t out_id = UINT32_MAX;
    if (!tryRegisterSpan(span, out_id)) {
        throw std::runtime_error(
            "AtomTable::registerSpan failed for atom type " +
            std::to_string(static_cast<int>(span.atom_type)));
    }
    return out_id;
}

//--------------------------------------------------//
// Lookup
//--------------------------------------------------//

const AtomEntry* AtomTable::getAtom(uint32_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Token IDs are offset by ATOM_TOKEN_BASE, convert to array index
    if (id < ATOM_TOKEN_BASE) {
        return nullptr;  // Not a valid atom token
    }
    
    uint32_t idx = id - ATOM_TOKEN_BASE;
    if (idx >= entries_.size()) {
        return nullptr;
    }
    
    // Verify the entry actually has this ID (sanity check)
    const AtomEntry& entry = entries_[idx];
    if (entry.id == id) {
        return &entry;
    }
    
    // Fallback: Linear search (shouldn't happen with proper indexing)
    for (const auto& e : entries_) {
        if (e.id == id) {
            return &e;
        }
    }
    return nullptr;
}

std::vector<const AtomEntry*> AtomTable::getAtomsByType(AtomType type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<const AtomEntry*> result;
    
    auto it = type_index_.find(type);
    if (it != type_index_.end()) {
        result.reserve(it->second.size());
        for (uint32_t token_id : it->second) {
            // Convert token ID to array index
            if (token_id >= ATOM_TOKEN_BASE) {
                uint32_t idx = token_id - ATOM_TOKEN_BASE;
                if (idx < entries_.size()) {
                    result.push_back(&entries_[idx]);
                }
            }
        }
    }
    
    return result;
}

bool AtomTable::hasAtom(uint32_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    // Convert token ID to array index and check bounds
    if (id < ATOM_TOKEN_BASE) return false;
    uint32_t idx = id - ATOM_TOKEN_BASE;
    return idx < entries_.size();
}

//--------------------------------------------------//
// Main Parse Entry Point
//--------------------------------------------------//

ParseResult AtomTable::parseAtom(AtomType type, const std::string& text) {
    std::string_view trimmed = trimWhitespace(text);
    const std::string* parse_text = &text;
    std::string trimmed_storage;
    if (trimmed.size() != text.size()) {
        trimmed_storage.assign(trimmed);
        parse_text = &trimmed_storage;
    }

    switch (type) {
        case AtomType::ATOM_INTEGER:
            return parseInteger(*parse_text);
            
        case AtomType::ATOM_FLOAT:
            return parseFloat(*parse_text);
            
        case AtomType::ATOM_HEX:
            return parseHex(*parse_text);
            
        case AtomType::ATOM_BINARY:
            return parseBinary(*parse_text);
            
        case AtomType::ATOM_URL:
            return parseURL(*parse_text);
            
        case AtomType::ATOM_EMAIL:
            return parseEmail(*parse_text);
            
        case AtomType::ATOM_PATH:
            return parsePath(*parse_text);
            
        case AtomType::ATOM_DATE:
            return parseDate(*parse_text);
            
        case AtomType::ATOM_TIME:
            return parseTime(*parse_text);
            
        case AtomType::ATOM_IP_ADDRESS:
            return parseIP(*parse_text);
            
        case AtomType::ATOM_STRING_LITERAL:
            return parseStringLiteral(*parse_text);
            
        case AtomType::ATOM_IDENTIFIER:
            return parseIdentifier(*parse_text);
            
        default:
            // Return as generic
            return ParseResult{true, AtomGeneric{*parse_text}, ""};
    }
}

//--------------------------------------------------//
// Integer Parsing
//--------------------------------------------------//

ParseResult AtomTable::parseInteger(const std::string& text) {
    ParseResult result;
    result.success = false;
    
    if (text.empty()) {
        result.error_message = "Empty input";
        return result;
    }
    
    try {
        AtomInteger atom;
        atom.base = 10;
        atom.has_sign = (text[0] == '+' || text[0] == '-');
        
        size_t pos = 0;
        atom.value = std::stoll(text, &pos, 10);
        
        if (pos != text.size()) {
            result.error_message = "Invalid integer format";
            return result;
        }
        
        result.success = true;
        result.value = atom;
    } catch (const std::exception& e) {
        result.error_message = e.what();
    }
    
    return result;
}

//--------------------------------------------------//
// Float Parsing
//--------------------------------------------------//

ParseResult AtomTable::parseFloat(const std::string& text) {
    ParseResult result;
    result.success = false;
    
    if (text.empty()) {
        result.error_message = "Empty input";
        return result;
    }
    
    try {
        AtomFloat atom;
        size_t pos = 0;
        atom.value = std::stod(text, &pos);
        
        if (pos != text.size()) {
            result.error_message = "Invalid float format";
            return result;
        }

        if (!std::isfinite(atom.value)) {
            result.error_message = "Non-finite float";
            return result;
        }
        
        // Check for exponent
        atom.has_exponent = (text.find('e') != std::string::npos || 
                            text.find('E') != std::string::npos);
        atom.exponent = 0;
        
        if (atom.has_exponent) {
            size_t e_pos = text.find_first_of("eE");
            if (e_pos != std::string::npos && e_pos + 1 < text.size()) {
                atom.exponent = std::stoi(text.substr(e_pos + 1));
            }
        }
        
        result.success = true;
        result.value = atom;
    } catch (const std::exception& e) {
        result.error_message = e.what();
    }
    
    return result;
}

//--------------------------------------------------//
// Hex Parsing
//--------------------------------------------------//

ParseResult AtomTable::parseHex(const std::string& text) {
    ParseResult result;
    result.success = false;
    
    if (text.size() < 3 || text[0] != '0' || 
        (text[1] != 'x' && text[1] != 'X')) {
        result.error_message = "Invalid hex format (expected 0x prefix)";
        return result;
    }
    
    try {
        AtomInteger atom;
        atom.base = 16;
        atom.has_sign = false;
        
        size_t pos = 0;
        atom.value = std::stoll(text, &pos, 16);
        
        if (pos != text.size()) {
            result.error_message = "Invalid hex digits";
            return result;
        }
        
        result.success = true;
        result.value = atom;
    } catch (const std::exception& e) {
        result.error_message = e.what();
    }
    
    return result;
}

//--------------------------------------------------//
// Binary Parsing
//--------------------------------------------------//

ParseResult AtomTable::parseBinary(const std::string& text) {
    ParseResult result;
    result.success = false;
    
    if (text.size() < 3 || text[0] != '0' || 
        (text[1] != 'b' && text[1] != 'B')) {
        result.error_message = "Invalid binary format (expected 0b prefix)";
        return result;
    }
    
    try {
        AtomInteger atom;
        atom.base = 2;
        atom.has_sign = false;
        
        // Parse binary digits manually
        atom.value = 0;
        for (size_t i = 2; i < text.size(); ++i) {
            if (text[i] == '0') {
                atom.value = atom.value * 2;
            } else if (text[i] == '1') {
                atom.value = atom.value * 2 + 1;
            } else {
                result.error_message = "Invalid binary digit";
                return result;
            }
        }
        
        result.success = true;
        result.value = atom;
    } catch (const std::exception& e) {
        result.error_message = e.what();
    }
    
    return result;
}

//--------------------------------------------------//
// URL Parsing
//--------------------------------------------------//

ParseResult AtomTable::parseURL(const std::string& text) {
    ParseResult result;
    result.success = false;
    
    AtomURL atom;
    atom.port = -1;
    
    // Find scheme
    size_t scheme_end = text.find("://");
    if (scheme_end == std::string::npos) {
        result.error_message = "Missing scheme";
        return result;
    }
    
    atom.scheme = text.substr(0, scheme_end);
    std::transform(atom.scheme.begin(), atom.scheme.end(), 
                   atom.scheme.begin(), ::tolower);
    
    size_t pos = scheme_end + 3;
    
    // Find host (and optional port)
    size_t path_start = text.find('/', pos);
    size_t query_start = text.find('?', pos);
    size_t fragment_start = text.find('#', pos);
    
    size_t host_end = text.size();
    if (path_start != std::string::npos) host_end = std::min(host_end, path_start);
    if (query_start != std::string::npos) host_end = std::min(host_end, query_start);
    if (fragment_start != std::string::npos) host_end = std::min(host_end, fragment_start);
    
    std::string host_port = text.substr(pos, host_end - pos);
    
    // Check for port
    size_t colon_pos = host_port.rfind(':');
    if (colon_pos != std::string::npos) {
        atom.host = host_port.substr(0, colon_pos);
        try {
            atom.port = std::stoi(host_port.substr(colon_pos + 1));
        } catch (...) {
            atom.host = host_port;
        }
    } else {
        atom.host = host_port;
    }
    
    pos = host_end;
    
    // Path
    if (path_start != std::string::npos && path_start < text.size()) {
        size_t path_end = text.size();
        if (query_start != std::string::npos) path_end = std::min(path_end, query_start);
        if (fragment_start != std::string::npos) path_end = std::min(path_end, fragment_start);
        atom.path = text.substr(path_start, path_end - path_start);
    }
    
    // Query
    if (query_start != std::string::npos && query_start < text.size()) {
        size_t query_end = text.size();
        if (fragment_start != std::string::npos) query_end = std::min(query_end, fragment_start);
        atom.query = text.substr(query_start + 1, query_end - query_start - 1);
    }
    
    // Fragment
    if (fragment_start != std::string::npos && fragment_start + 1 < text.size()) {
        atom.fragment = text.substr(fragment_start + 1);
    }
    
    result.success = true;
    result.value = atom;
    return result;
}

//--------------------------------------------------//
// Email Parsing
//--------------------------------------------------//

ParseResult AtomTable::parseEmail(const std::string& text) {
    ParseResult result;
    result.success = false;
    
    size_t at_pos = text.find('@');
    if (at_pos == std::string::npos || at_pos == 0 || at_pos == text.size() - 1) {
        result.error_message = "Invalid email format";
        return result;
    }
    
    AtomEmail atom;
    atom.local = text.substr(0, at_pos);
    atom.domain = text.substr(at_pos + 1);
    
    // Basic validation
    if (atom.domain.find('.') == std::string::npos) {
        result.error_message = "Invalid domain (no TLD)";
        return result;
    }
    
    result.success = true;
    result.value = atom;
    return result;
}

//--------------------------------------------------//
// Path Parsing
//--------------------------------------------------//

ParseResult AtomTable::parsePath(const std::string& text) {
    ParseResult result;
    result.success = false;
    
    if (text.empty()) {
        result.error_message = "Empty path";
        return result;
    }
    
    AtomPath atom;
    atom.is_windows = (text.find('\\') != std::string::npos) ||
                      (text.size() > 2 && text[1] == ':');
    atom.is_absolute = (text[0] == '/' || text[0] == '\\') ||
                       (text.size() > 2 && text[1] == ':');
    
    // Split into components
    std::string current;
    char sep = atom.is_windows ? '\\' : '/';
    
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        if (c == '/' || c == '\\') {
            if (!current.empty()) {
                atom.components.push_back(current);
                current.clear();
            }
        } else {
            current += c;
        }
    }
    
    if (!current.empty()) {
        atom.components.push_back(current);
    }
    
    // Extract extension from last component
    if (!atom.components.empty()) {
        const std::string& last = atom.components.back();
        size_t dot_pos = last.rfind('.');
        if (dot_pos != std::string::npos && dot_pos > 0) {
            atom.extension = last.substr(dot_pos + 1);
        }
    }
    
    result.success = true;
    result.value = atom;
    return result;
}

//--------------------------------------------------//
// Date Parsing
//--------------------------------------------------//

ParseResult AtomTable::parseDate(const std::string& text) {
    ParseResult result;
    result.success = false;
    
    AtomDate atom;
    
    // Try ISO format: YYYY-MM-DD
    if (text.size() == 10 && text[4] == '-' && text[7] == '-') {
        try {
            atom.year = std::stoi(text.substr(0, 4));
            atom.month = std::stoi(text.substr(5, 2));
            atom.day = std::stoi(text.substr(8, 2));
            atom.format = AtomDate::ISO;
            
            if (atom.month >= 1 && atom.month <= 12 &&
                atom.day >= 1 && atom.day <= 31) {
                result.success = true;
                result.value = atom;
                return result;
            }
        } catch (...) {}
    }
    
    // Hand-written US format parser: MM/DD/YYYY or M/D/YY (no regex)
    size_t first_slash = text.find('/');
    if (first_slash != std::string::npos) {
        size_t second_slash = text.find('/', first_slash + 1);
        if (second_slash != std::string::npos) {
            try {
                atom.month = std::stoi(text.substr(0, first_slash));
                atom.day = std::stoi(text.substr(first_slash + 1, second_slash - first_slash - 1));
                atom.year = std::stoi(text.substr(second_slash + 1));
                
                // Handle 2-digit year
                if (atom.year < 100) {
                    atom.year += (atom.year < 50) ? 2000 : 1900;
                }
                
                atom.format = AtomDate::US;
                
                if (atom.month >= 1 && atom.month <= 12 &&
                    atom.day >= 1 && atom.day <= 31) {
                    result.success = true;
                    result.value = atom;
                    return result;
                }
            } catch (...) {}
        }
    }
    
    // Hand-written EU format parser: DD.MM.YYYY or D.M.YY (no regex)
    size_t first_dot = text.find('.');
    if (first_dot != std::string::npos) {
        size_t second_dot = text.find('.', first_dot + 1);
        if (second_dot != std::string::npos) {
            try {
                atom.day = std::stoi(text.substr(0, first_dot));
                atom.month = std::stoi(text.substr(first_dot + 1, second_dot - first_dot - 1));
                atom.year = std::stoi(text.substr(second_dot + 1));
                
                // Handle 2-digit year
                if (atom.year < 100) {
                    atom.year += (atom.year < 50) ? 2000 : 1900;
                }
                
                atom.format = AtomDate::EU;
                
                if (atom.month >= 1 && atom.month <= 12 &&
                    atom.day >= 1 && atom.day <= 31) {
                    result.success = true;
                    result.value = atom;
                    return result;
                }
            } catch (...) {}
        }
    }
    
    result.error_message = "Unrecognized date format";
    return result;
}

//--------------------------------------------------//
// Time Parsing (NO REGEX)
//--------------------------------------------------//

ParseResult AtomTable::parseTime(const std::string& text) {
    ParseResult result;
    result.success = false;
    
    AtomTime atom;
    atom.second = 0;
    atom.is_24h = true;
    atom.is_pm = false;
    
    // Check for AM/PM
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), 
                   lower_text.begin(), ::tolower);
    
    bool has_am = lower_text.find("am") != std::string::npos;
    bool has_pm = lower_text.find("pm") != std::string::npos;
    
    if (has_am || has_pm) {
        atom.is_24h = false;
        atom.is_pm = has_pm;
    }
    
    // Hand-written parser: HH:MM or HH:MM:SS (no regex)
    size_t first_colon = text.find(':');
    if (first_colon == std::string::npos) {
        result.error_message = "Missing colon separator";
        return result;
    }
    
    try {
        // Parse hour
        std::string hour_str = text.substr(0, first_colon);
        atom.hour = std::stoi(hour_str);
        
        // Find second colon (if exists)
        size_t second_colon = text.find(':', first_colon + 1);
        
        if (second_colon != std::string::npos) {
            // HH:MM:SS format
            atom.minute = std::stoi(text.substr(first_colon + 1, second_colon - first_colon - 1));
            
            // Extract seconds (stop at space or end)
            size_t sec_end = text.find(' ', second_colon);
            if (sec_end == std::string::npos) sec_end = text.size();
            atom.second = std::stoi(text.substr(second_colon + 1, sec_end - second_colon - 1));
        } else {
            // HH:MM format
            size_t min_end = text.find(' ', first_colon);
            if (min_end == std::string::npos) min_end = text.size();
            atom.minute = std::stoi(text.substr(first_colon + 1, min_end - first_colon - 1));
        }
        
        // Validate
        int max_hour = atom.is_24h ? 23 : 12;
        if (atom.hour >= 0 && atom.hour <= max_hour &&
            atom.minute >= 0 && atom.minute <= 59 &&
            atom.second >= 0 && atom.second <= 59) {
            result.success = true;
            result.value = atom;
            return result;
        } else {
            result.error_message = "Out of range time values";
        }
    } catch (...) {
        result.error_message = "Failed to parse time components";
    }
    
    return result;
}

//--------------------------------------------------//
// IP Address Parsing (NO REGEX)
//--------------------------------------------------//

ParseResult AtomTable::parseIP(const std::string& text) {
    ParseResult result;
    result.success = false;
    
    AtomIP atom;
    atom.is_valid = false;
    
    // Hand-written parser: XXX.XXX.XXX.XXX (no regex)
    size_t pos = 0;
    for (int i = 0; i < 4; ++i) {
        size_t dot_pos = (i < 3) ? text.find('.', pos) : text.size();
        
        if (dot_pos == std::string::npos) {
            result.error_message = "Invalid IP format - missing dots";
            return result;
        }
        
        try {
            std::string octet_str = text.substr(pos, dot_pos - pos);
            if (octet_str.empty() || octet_str.size() > 3) {
                result.error_message = "Invalid octet length";
                return result;
            }
            
            int octet = std::stoi(octet_str);
            if (octet < 0 || octet > 255) {
                result.error_message = "Octet out of range (0-255)";
                return result;
            }
            
            atom.octets[i] = static_cast<uint8_t>(octet);
            pos = dot_pos + 1;
        } catch (...) {
            result.error_message = "Failed to parse octet";
            return result;
        }
    }
    
    atom.is_valid = true;
    result.success = true;
    result.value = atom;
    return result;
}

//--------------------------------------------------//
// String Literal Parsing
//--------------------------------------------------//

ParseResult AtomTable::parseStringLiteral(const std::string& text) {
    ParseResult result;
    result.success = false;
    
    if (text.size() < 2) {
        result.error_message = "String too short";
        return result;
    }
    
    char quote = text[0];
    if ((quote != '"' && quote != '\'') || text.back() != quote) {
        result.error_message = "Invalid string delimiters";
        return result;
    }
    
    AtomString atom;
    atom.quote_char = quote;
    atom.has_escapes = false;
    
    // Unescape the content
    std::string content = text.substr(1, text.size() - 2);
    std::string unescaped;
    unescaped.reserve(content.size());
    
    for (size_t i = 0; i < content.size(); ++i) {
        if (content[i] == '\\' && i + 1 < content.size()) {
            atom.has_escapes = true;
            char next = content[i + 1];
            switch (next) {
                case 'n': unescaped += '\n'; break;
                case 't': unescaped += '\t'; break;
                case 'r': unescaped += '\r'; break;
                case '\\': unescaped += '\\'; break;
                case '"': unescaped += '"'; break;
                case '\'': unescaped += '\''; break;
                case '0': unescaped += '\0'; break;
                default: unescaped += next; break;
            }
            ++i;
        } else {
            unescaped += content[i];
        }
    }
    
    atom.value = unescaped;
    result.success = true;
    result.value = atom;
    return result;
}

//--------------------------------------------------//
// Identifier Parsing
//--------------------------------------------------//

ParseResult AtomTable::parseIdentifier(const std::string& text) {
    ParseResult result;
    result.success = false;
    
    if (text.empty() || (!std::isalpha(text[0]) && text[0] != '_')) {
        result.error_message = "Invalid identifier start";
        return result;
    }
    
    AtomIdentifier atom;
    atom.name = text;
    
    // Detect naming style
    bool has_underscore = text.find('_') != std::string::npos;
    bool has_upper = false;
    bool has_lower = false;
    bool starts_upper = std::isupper(text[0]);
    
    for (char c : text) {
        if (std::isupper(c)) has_upper = true;
        if (std::islower(c)) has_lower = true;
    }
    
    if (has_underscore) {
        if (has_upper && !has_lower) {
            atom.style = AtomIdentifier::SCREAMING_SNAKE;
        } else {
            atom.style = AtomIdentifier::SNAKE_CASE;
        }
    } else if (has_upper && has_lower) {
        atom.style = starts_upper ? AtomIdentifier::PASCAL_CASE : AtomIdentifier::CAMEL_CASE;
    } else {
        atom.style = AtomIdentifier::UNKNOWN;
    }
    
    result.success = true;
    result.value = atom;
    return result;
}

//--------------------------------------------------//
// Serialization
//--------------------------------------------------//

std::string AtomTable::atomToString(const AtomEntry& entry) const {
    // Return parsed string if available, otherwise raw text
    std::string_view parsed_str = getString(entry.parsed_ref);
    if (!parsed_str.empty()) {
        return std::string(parsed_str);
    }
    return std::string(getString(entry.raw_text_ref));
}

// Serialize atom value directly to buffer (ZERO HEAP ALLOCATION!)
size_t AtomTable::atomValueSerialize(AtomType type, const AtomValue& value, char* out, size_t max) {
    if (!out || max == 0) return 0;
    
    size_t written = 0;
    
    std::visit([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        
        if constexpr (std::is_same_v<T, AtomInteger>) {
            if (arg.base == 16) {
                written = snprintf(out, max, "0x%llx", (long long)arg.value);
            } else if (arg.base == 2) {
                // Binary
                written = snprintf(out, max, "0b");
                if (written < max) {
                    int64_t v = arg.value;
                    char temp[65];
                    int pos = 0;
                    do {
                        temp[pos++] = '0' + (v & 1);
                        v >>= 1;
                    } while (v > 0 && pos < 64);
                    for (int i = pos - 1; i >= 0 && written < max; --i) {
                        out[written++] = temp[i];
                    }
                }
            } else {
                written = snprintf(out, max, "%lld", (long long)arg.value);
            }
        }
        else if constexpr (std::is_same_v<T, AtomFloat>) {
            written = snprintf(out, max, "%g", arg.value);
        }
        else if constexpr (std::is_same_v<T, AtomURL>) {
            written = snprintf(out, max, "%s://%s", arg.scheme.c_str(), arg.host.c_str());
            if (arg.port > 0 && written < max) {
                written += snprintf(out + written, max - written, ":%d", arg.port);
            }
            if (!arg.path.empty() && written < max) {
                written += snprintf(out + written, max - written, "%s", arg.path.c_str());
            }
            if (!arg.query.empty() && written < max) {
                written += snprintf(out + written, max - written, "?%s", arg.query.c_str());
            }
            if (!arg.fragment.empty() && written < max) {
                written += snprintf(out + written, max - written, "#%s", arg.fragment.c_str());
            }
        }
        else if constexpr (std::is_same_v<T, AtomEmail>) {
            written = snprintf(out, max, "%s@%s", arg.local.c_str(), arg.domain.c_str());
        }
        else if constexpr (std::is_same_v<T, AtomString>) {
            // Write unescaped value
            size_t len = std::min(arg.value.size(), max - 1);
            memcpy(out, arg.value.data(), len);
            written = len;
        }
        else if constexpr (std::is_same_v<T, AtomIdentifier>) {
            size_t len = std::min(arg.name.size(), max - 1);
            memcpy(out, arg.name.data(), len);
            written = len;
        }
        else if constexpr (std::is_same_v<T, AtomGeneric>) {
            size_t len = std::min(arg.raw_value.size(), max - 1);
            memcpy(out, arg.raw_value.data(), len);
            written = len;
        }
    }, value);
    
    return (written < max) ? written : 0;  // Return 0 if truncated
}

std::string AtomTable::atomValueToString(AtomType type, const AtomValue& value) {
    std::ostringstream oss;
    
    std::visit([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        
        if constexpr (std::is_same_v<T, AtomInteger>) {
            if (arg.base == 16) {
                oss << "0x" << std::hex << arg.value;
            } else if (arg.base == 2) {
                oss << "0b";
                // Convert to binary string
                int64_t v = arg.value;
                std::string bin;
                do {
                    bin = (char)('0' + (v & 1)) + bin;
                    v >>= 1;
                } while (v > 0);
                oss << bin;
            } else {
                oss << arg.value;
            }
        }
        else if constexpr (std::is_same_v<T, AtomFloat>) {
            oss << arg.value;
        }
        else if constexpr (std::is_same_v<T, AtomURL>) {
            oss << arg.scheme << "://" << arg.host;
            if (arg.port > 0) oss << ":" << arg.port;
            oss << arg.path;
            if (!arg.query.empty()) oss << "?" << arg.query;
            if (!arg.fragment.empty()) oss << "#" << arg.fragment;
        }
        else if constexpr (std::is_same_v<T, AtomEmail>) {
            oss << arg.local << "@" << arg.domain;
        }
        else if constexpr (std::is_same_v<T, AtomPath>) {
            char sep = arg.is_windows ? '\\' : '/';
            if (arg.is_absolute) oss << sep;
            for (size_t i = 0; i < arg.components.size(); ++i) {
                if (i > 0) oss << sep;
                oss << arg.components[i];
            }
        }
        else if constexpr (std::is_same_v<T, AtomDate>) {
            if (arg.format == AtomDate::ISO) {
                oss << arg.year << "-" 
                    << std::setw(2) << std::setfill('0') << arg.month << "-"
                    << std::setw(2) << std::setfill('0') << arg.day;
            } else {
                oss << arg.month << "/" << arg.day << "/" << arg.year;
            }
        }
        else if constexpr (std::is_same_v<T, AtomTime>) {
            oss << std::setw(2) << std::setfill('0') << arg.hour << ":"
                << std::setw(2) << std::setfill('0') << arg.minute;
            if (arg.second > 0) {
                oss << ":" << std::setw(2) << std::setfill('0') << arg.second;
            }
            if (!arg.is_24h) {
                oss << (arg.is_pm ? "pm" : "am");
            }
        }
        else if constexpr (std::is_same_v<T, AtomIP>) {
            oss << (int)arg.octets[0] << "." << (int)arg.octets[1] << "."
                << (int)arg.octets[2] << "." << (int)arg.octets[3];
        }
        else if constexpr (std::is_same_v<T, AtomString>) {
            oss << arg.quote_char << arg.value << arg.quote_char;
        }
        else if constexpr (std::is_same_v<T, AtomIdentifier>) {
            oss << arg.name;
        }
        else if constexpr (std::is_same_v<T, AtomGeneric>) {
            oss << arg.raw_value;
        }
    }, value);
    
    return oss.str();
}

double AtomTable::getNumericValue(const AtomEntry& entry) {
    // Numeric value is already packed in the entry for GPU
    if (isNumericAtom(entry.type)) {
        return static_cast<double>(entry.numeric_value);
    }
    return 0.0;
}

bool AtomTable::hasNumericValue(AtomType type) {
    return isNumericAtom(type);
}

//--------------------------------------------------//
// Metadata Helpers
//--------------------------------------------------//

AtomCategory AtomTable::getCategoryForType(AtomType type) {
    switch (type) {
        case AtomType::ATOM_INTEGER:
        case AtomType::ATOM_FLOAT:
        case AtomType::ATOM_HEX:
        case AtomType::ATOM_BINARY:
            return AtomCategory::NUMERIC;
            
        case AtomType::ATOM_DATE:
        case AtomType::ATOM_TIME:
            return AtomCategory::TEMPORAL;
            
        case AtomType::ATOM_URL:
        case AtomType::ATOM_EMAIL:
        case AtomType::ATOM_PATH:
        case AtomType::ATOM_IP_ADDRESS:
            return AtomCategory::STRUCTURAL;
            
        case AtomType::ATOM_STRING_LITERAL:
            return AtomCategory::STRING;
            
        case AtomType::ATOM_IDENTIFIER:
            return AtomCategory::IDENTIFIER;
            
        default:
            return AtomCategory::GENERIC;
    }
}

uint64_t AtomTable::computeHash(const AtomEntry& entry) const {
    std::string_view raw_text = getString(entry.raw_text_ref);
    return computeHash(entry.type, raw_text);
}

uint64_t AtomTable::computeHash(AtomType type, std::string_view raw_text) {
    // Simple FNV-1a hash
    uint64_t hash = 14695981039346656037ULL;
    
    // Hash the type
    hash ^= static_cast<uint64_t>(type);
    hash *= 1099511628211ULL;
    
    // Hash the raw text
    for (char c : raw_text) {
        hash ^= static_cast<uint64_t>(static_cast<uint8_t>(c));
        hash *= 1099511628211ULL;
    }
    
    return hash;
}

uint64_t AtomTable::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration);
    return static_cast<uint64_t>(micros.count());
}

void AtomTable::setOrigin(uint32_t id, AtomOrigin origin) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (id < ATOM_TOKEN_BASE) return;
    uint32_t idx = id - ATOM_TOKEN_BASE;
    if (idx < entries_.size()) {
        entries_[idx].origin = origin;
    }
}

void AtomTable::setConfidence(uint32_t id, float confidence) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (id < ATOM_TOKEN_BASE) return;
    uint32_t idx = id - ATOM_TOKEN_BASE;
    if (idx < entries_.size()) {
        entries_[idx].confidence = std::clamp(confidence, 0.0f, 1.0f);
    }
}

void AtomTable::setCategory(uint32_t id, AtomCategory category) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (id < ATOM_TOKEN_BASE) return;
    uint32_t idx = id - ATOM_TOKEN_BASE;
    if (idx < entries_.size()) {
        entries_[idx].category = category;
    }
}

//--------------------------------------------------//
// GPU Packing (Numeric Values Only)
//--------------------------------------------------//

void AtomTable::packNumericValue(AtomEntry& entry, const AtomValue& parsed) {
    entry.numeric_value = 0.0f;
    entry.flags = 0;
    
    std::visit([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        
        if constexpr (std::is_same_v<T, AtomInteger>) {
            entry.numeric_value = static_cast<float>(arg.value);
            entry.flags = (arg.base << 8) | (arg.has_sign ? 1 : 0);
        }
        else if constexpr (std::is_same_v<T, AtomFloat>) {
            entry.numeric_value = static_cast<float>(arg.value);
            entry.flags = (arg.has_exponent ? 1 : 0) | (arg.exponent << 8);
        }
        else if constexpr (std::is_same_v<T, AtomURL>) {
            entry.flags = arg.port > 0 ? arg.port : 0;
        }
        else if constexpr (std::is_same_v<T, AtomPath>) {
            entry.flags = (arg.is_absolute ? 1 : 0) | (arg.is_windows ? 2 : 0);
        }
        else if constexpr (std::is_same_v<T, AtomDate>) {
            entry.numeric_value = static_cast<float>(
                arg.year * 10000 + arg.month * 100 + arg.day);
            entry.flags = static_cast<uint32_t>(arg.format);
        }
        else if constexpr (std::is_same_v<T, AtomTime>) {
            entry.numeric_value = static_cast<float>(
                arg.hour * 10000 + arg.minute * 100 + arg.second);
            entry.flags = (arg.is_24h ? 0 : 1) | (arg.is_pm ? 2 : 0);
        }
        else if constexpr (std::is_same_v<T, AtomIP>) {
            uint32_t ip_val = (arg.octets[0] << 24) | (arg.octets[1] << 16) |
                             (arg.octets[2] << 8) | arg.octets[3];
            entry.numeric_value = static_cast<float>(ip_val);
            entry.flags = arg.is_valid ? 1 : 0;
        }
        else if constexpr (std::is_same_v<T, AtomString>) {
            entry.flags = (arg.quote_char == '"' ? 1 : 0) | (arg.has_escapes ? 2 : 0);
        }
        else if constexpr (std::is_same_v<T, AtomIdentifier>) {
            entry.flags = static_cast<uint32_t>(arg.style);
        }
    }, parsed);
}

//--------------------------------------------------//
// Batch GPU Upload - Only uploads pending atoms
//--------------------------------------------------//

bool AtomTable::uploadToGPU(AtomTable::GPUAtomData& out_data, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Only upload if there are pending changes
    if (!gpu_dirty_ || pending_gpu_upload_.empty()) {
        out_data.num_atoms = entries_.size();
        return true;
    }
    
    size_t num_atoms = entries_.size();
    if (num_atoms == 0) {
        out_data.num_atoms = 0;
        gpu_dirty_ = false;
        return true;
    }
    
    cudaError_t err;
    
    // Allocate device memory (numeric values, types, flags only - no aux buffer)
    err = cudaMalloc(&out_data.d_numeric_values, num_atoms * sizeof(float));
    if (err != cudaSuccess) return false;
    
    err = cudaMalloc(&out_data.d_flags, num_atoms * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(out_data.d_numeric_values);
        return false;
    }
    
    err = cudaMalloc(&out_data.d_types, num_atoms * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(out_data.d_numeric_values);
        cudaFree(out_data.d_flags);
        return false;
    }
    
    // Pack data into host arrays (cache-aligned for transfer efficiency)
    std::vector<float> h_numeric(num_atoms);
    std::vector<uint32_t> h_flags(num_atoms);
    std::vector<uint32_t> h_types(num_atoms);
    
    for (size_t i = 0; i < num_atoms; ++i) {
        const AtomEntry& entry = entries_[i];
        h_numeric[i] = entry.numeric_value;
        h_flags[i] = entry.flags;
        h_types[i] = static_cast<uint32_t>(entry.type);
    }
    
    // Batch upload to GPU (async for performance)
    cudaMemcpyAsync(out_data.d_numeric_values, h_numeric.data(),
                    num_atoms * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(out_data.d_flags, h_flags.data(),
                    num_atoms * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(out_data.d_types, h_types.data(),
                    num_atoms * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    
    out_data.num_atoms = num_atoms;
    
    // Clear pending upload queue and mark clean
    pending_gpu_upload_.clear();
    gpu_dirty_ = false;
    
    return cudaGetLastError() == cudaSuccess;
}

// Convenience: Upload to internal GPU buffer
bool AtomTable::uploadToGPU(cudaStream_t stream) {
    // Free existing GPU data if any
    freeGPUData(gpu_data_);
    
    // Upload to internal buffer
    return uploadToGPU(gpu_data_, stream);
}

void AtomTable::freeGPUData(AtomTable::GPUAtomData& data) {
    if (data.d_numeric_values) cudaFree(data.d_numeric_values);
    if (data.d_flags) cudaFree(data.d_flags);
    if (data.d_types) cudaFree(data.d_types);
    
    data = AtomTable::GPUAtomData{};
}

} // namespace Tokenizer
} // namespace GRIM

