#pragma once
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <memory>
#include <optional>
#include <mutex>
#include <cstdint>
#include <ctime>

// Forward declare FlatBuffers types (will be generated)
namespace GRIM { namespace Memory {
    struct MemoryRecord;
    struct MemoryStore;
    struct IndexEntry;
    struct TagIndex;
}}

namespace GRIM {

// ============================================================================
// ENUMS (Unified from memory.hpp + memory_manager.hpp)
// ============================================================================

enum class CommType : uint8_t { 
    COMMAND = 0, 
    QUESTION = 1, 
    BANTER = 2, 
    UNKNOWN = 3 
};

enum class SourceType : uint8_t {
    USER_VOICE = 0,
    USER_TEXT = 1,
  SYSTEM_HW = 2,
    SYSTEM_SW = 3,
    NETWORK_API = 4,
    GRIM_INTERNAL = 5,
    BASE = 6,
    FIELD = 7,
    FUSED = 8
};

enum class TypeTag : uint8_t {
    FACT = 0,
    EVENT = 1,
    COMMAND = 2,
    STATUS = 3,
    SUMMARY = 4,
    LEARNED_COMMAND = 5,
    UNKNOWN_COMMAND = 6,
    PREDICTION = 7
};

enum class MemoryIntent : uint8_t {
    INFORM = 0,
    ASK = 1,
    SET_PREF = 2,
    CORRECT = 3,
    STATUS_UPDATE = 4,
    QUERY = 5
};

enum class ContextType : uint8_t {
    CONVERSATION = 0,
    WAKE = 1,
    DEV_MODE = 2,
    MONITOR = 3,
    SYSTEM_BOOT = 4,
    COMMAND_LEARNING = 5
};

enum class Modality : uint8_t {
    TEXT = 0,
    AUDIO = 1,
    VISION = 2,
    SENSOR = 3
};

// ============================================================================
// LEGACY COMPATIBILITY (Old enum names mapped to new system)
// ============================================================================

// Map old SourceTag to new SourceType
using SourceTag = SourceType;
namespace SourceTagCompat {
    constexpr SourceType UserVoice = SourceType::USER_VOICE;
    constexpr SourceType UserText = SourceType::USER_TEXT;
    constexpr SourceType SystemHW = SourceType::SYSTEM_HW;
    constexpr SourceType SystemSW = SourceType::SYSTEM_SW;
    constexpr SourceType NetworkAPI = SourceType::NETWORK_API;
    constexpr SourceType GrimInternal = SourceType::GRIM_INTERNAL;
}

// Map old IntentTag to new MemoryIntent
using IntentTag = MemoryIntent;
namespace IntentTagCompat {
    constexpr MemoryIntent Inform = MemoryIntent::INFORM;
    constexpr MemoryIntent Ask = MemoryIntent::ASK;
    constexpr MemoryIntent SetPref = MemoryIntent::SET_PREF;
    constexpr MemoryIntent Correct = MemoryIntent::CORRECT;
    constexpr MemoryIntent StatusUpdate = MemoryIntent::STATUS_UPDATE;
  constexpr MemoryIntent Query = MemoryIntent::QUERY;
}

// Map old ContextTag to new ContextType
using ContextTag = ContextType;
namespace ContextTagCompat {
    constexpr ContextType Conversation = ContextType::CONVERSATION;
    constexpr ContextType Wake = ContextType::WAKE;
constexpr ContextType DevMode = ContextType::DEV_MODE;
    constexpr ContextType Monitor = ContextType::MONITOR;
    constexpr ContextType SystemBoot = ContextType::SYSTEM_BOOT;
    constexpr ContextType CommandLearning = ContextType::COMMAND_LEARNING;
}

// ============================================================================
// STRING CONVERSION MAPS
// ============================================================================

static const std::unordered_map<SourceType, std::string> SourceNames = {
    {SourceType::USER_VOICE, "user.voice"},
    {SourceType::USER_TEXT, "user.text"},
    {SourceType::SYSTEM_HW, "system.hw"},
    {SourceType::SYSTEM_SW, "system.sw"},
    {SourceType::NETWORK_API, "network.api"},
    {SourceType::GRIM_INTERNAL, "grim.internal"},
    {SourceType::BASE, "base"},
    {SourceType::FIELD, "field"},
    {SourceType::FUSED, "fused"}
};

static const std::unordered_map<TypeTag, std::string> TypeNames = {
    {TypeTag::FACT, "fact"},
    {TypeTag::EVENT, "event"},
    {TypeTag::COMMAND, "command"},
    {TypeTag::STATUS, "status"},
 {TypeTag::SUMMARY, "summary"},
    {TypeTag::LEARNED_COMMAND, "learnedcommand"},
    {TypeTag::UNKNOWN_COMMAND, "unknowncommand"},
    {TypeTag::PREDICTION, "prediction"}
};

static const std::unordered_map<MemoryIntent, std::string> IntentNames = {
    {MemoryIntent::INFORM, "inform"},
    {MemoryIntent::ASK, "ask"},
    {MemoryIntent::SET_PREF, "set_pref"},
    {MemoryIntent::CORRECT, "correct"},
    {MemoryIntent::STATUS_UPDATE, "status_update"},
    {MemoryIntent::QUERY, "query"}
};

static const std::unordered_map<ContextType, std::string> ContextNames = {
    {ContextType::CONVERSATION, "conversation"},
{ContextType::WAKE, "wake"},
    {ContextType::DEV_MODE, "dev_mode"},
    {ContextType::MONITOR, "monitor"},
    {ContextType::SYSTEM_BOOT, "system_boot"},
    {ContextType::COMMAND_LEARNING, "command_learning"}
};

static const std::unordered_map<CommType, std::string> CommTypeNames = {
    {CommType::COMMAND, "command"},
    {CommType::QUESTION, "question"},
    {CommType::BANTER, "banter"},
    {CommType::UNKNOWN, "unknown"}
};

static const std::unordered_map<Modality, std::string> ModalityNames = {
    {Modality::TEXT, "text"},
    {Modality::AUDIO, "audio"},
    {Modality::VISION, "vision"},
    {Modality::SENSOR, "sensor"}
};

// Helper to convert enum to string
template<typename EnumType>
inline std::string toString(EnumType value, const std::unordered_map<EnumType, std::string>& map) {
    auto it = map.find(value);
    return (it != map.end()) ? it->second : "unknown";
}

// Helper to convert string to enum
template<typename EnumType>
inline EnumType fromString(const std::string& s, const std::unordered_map<EnumType, std::string>& map, EnumType fallback) {
    for (const auto& [k, v] : map) {
      if (v == s) return k;
    }
    return fallback;
}

// ============================================================================
// UNIFIED MEMORY OBJECT (replaces old MemoryObject)
// ============================================================================

class UnifiedMemoryObject {
public:
    // Core identity
    uint64_t id = 0;
    uint64_t timestamp = 0;
    
    // Classification
    SourceType source = SourceType::GRIM_INTERNAL;
    TypeTag type = TypeTag::FACT;
    MemoryIntent intent = MemoryIntent::INFORM;
    ContextType context = ContextType::CONVERSATION;
    CommType comm_type = CommType::UNKNOWN;
    Modality modality = Modality::TEXT;
    
    // Content
    std::string raw;
    std::string normalized;
    std::string intent_name;
    
    // Metadata
    std::vector<std::string> tags;
    float confidence = 1.0f;
    float importance = 1.0f;
    float recency_weight = 1.0f;
    
    // Semantic search (768-dimensional embedding)
    std::array<float, 768> embedding{};
    
 // Relations
    uint64_t parent_id = 0;
    std::vector<uint64_t> related_ids;
    
    // Constructors
    UnifiedMemoryObject() = default;
    
    UnifiedMemoryObject(SourceType src, TypeTag t, MemoryIntent i, ContextType c,
     const std::string& rawInput, float conf = 1.0f)
        : source(src), type(t), intent(i), context(c),
          confidence(conf), raw(rawInput), normalized(rawInput) {
 id = generateID();
      timestamp = std::time(nullptr);
  }
    
    // Serialization
    std::string toJSON() const;
    static UnifiedMemoryObject fromJSON(const std::string& jsonStr);
    
    // FlatBuffers serialization (zero-copy)
std::vector<uint8_t> toFlatBuffer() const;
    static UnifiedMemoryObject fromFlatBuffer(const uint8_t* data, size_t size);
    
    // Utilities
    static uint64_t generateID();
    static std::string generateUUID();  // Legacy compatibility
};

// Backward compatibility alias
using MemoryObject = UnifiedMemoryObject;

// ============================================================================
// QUERY & RETRIEVAL
// ============================================================================

struct QueryResult {
    uint64_t id = 0;
    float score = 0.0f;
    UnifiedMemoryObject memory;
};

struct RetrievalQuery {
  CommType type = CommType::UNKNOWN;
    std::string text;
    std::vector<float> embedding;
std::string intent;
    std::vector<std::string> tags;
  
    // Filters
    std::optional<SourceType> source_filter;
    std::optional<TypeTag> type_filter;
    std::optional<uint64_t> min_timestamp;
    std::optional<uint64_t> max_timestamp;
    float min_confidence = 0.0f;
    
    int max_results = 10;
};

// ============================================================================
// UNIFIED MEMORY STORAGE (replaces MemoryStorage)
// ============================================================================

class UnifiedMemoryStorage {
public:
 UnifiedMemoryStorage();
    ~UnifiedMemoryStorage();
    
    // Lifecycle
    void initialize(const std::string& storagePath);
 void shutdown();
    void flush();
    
    // Storage operations
    void storeShortTerm(const UnifiedMemoryObject& obj);
    void storeLongTerm(const UnifiedMemoryObject& obj);
    
    // Retrieval
    std::optional<UnifiedMemoryObject> getById(uint64_t id);
    std::vector<UnifiedMemoryObject> search(const std::string& query, int maxResults = 10);
    std::vector<QueryResult> semanticSearch(const RetrievalQuery& query);
    
    // Tag-based indexing
    std::vector<UnifiedMemoryObject> getByTag(const std::string& tag);
    std::vector<UnifiedMemoryObject> getByTags(const std::vector<std::string>& tags, bool matchAll = false);
    
    // Type-specific queries
    std::optional<UnifiedMemoryObject> findLearnedCommand(const std::string& phrase);
    std::vector<UnifiedMemoryObject> getAllLearnedCommands();
    std::vector<UnifiedMemoryObject> getByType(TypeTag type);
    std::vector<UnifiedMemoryObject> getBySource(SourceType source);
    
    // Specialized storage
    void storeLearnedCommand(const std::string& phrase, const std::string& action, float confidence = 1.0f);
 
    // Maintenance
    void decay(float rate);
    void rebuildIndex();
    void compactStorage();
    
    // Statistics
 struct Stats {
uint64_t total_records = 0;
uint64_t short_term_count = 0;
      uint64_t long_term_count = 0;
        uint64_t indexed_tags = 0;
 uint64_t storage_size_bytes = 0;
        float avg_confidence = 0.0f;
 };
 Stats getStats() const;
    
    // Legacy compatibility
    static bool recentlyModified(const std::string& key, int seconds);
    static std::time_t getLastModified(const std::string& key);
    
private:
    void loadFromDisk();
    void saveToDisk();
    void saveToFlatBuffer();
    void loadFromFlatBuffer();
    void updateIndex(const UnifiedMemoryObject& obj);
    void updateTagIndex(const UnifiedMemoryObject& obj);
    
    static constexpr size_t SHORT_TERM_MAX = 100;  // Increased from 50
    
    // In-memory storage
    std::vector<UnifiedMemoryObject> shortTerm;
    std::unordered_map<uint64_t, UnifiedMemoryObject> longTerm;
    
 // Indexes for fast retrieval
    std::unordered_map<std::string, std::vector<uint64_t>> tagIndex;     // tag -> memory IDs
    std::unordered_map<TypeTag, std::vector<uint64_t>> typeIndex;        // type -> memory IDs
    std::unordered_map<SourceType, std::vector<uint64_t>> sourceIndex;   // source -> memory IDs
    
    // FlatBuffer storage (zero-copy)
    std::vector<uint8_t> flatBufferData;
    
    // File paths
    std::string storagePath;
    std::string jsonPath;  // Legacy JSON format
    std::string fbPath;    // FlatBuffer format
  std::string indexPath; // Separate index file
    
    // Thread safety
    mutable std::mutex mtx;
};

// Backward compatibility alias
using MemoryStorage = UnifiedMemoryStorage;

} // namespace GRIM
