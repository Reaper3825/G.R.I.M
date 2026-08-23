#include "unified_memory.hpp"
#include "unified_memory_generated.h"  // FlatBuffers generated header
#include "atomic_writer.hpp"
#include "logger.hpp"
#include <fstream>
#include <filesystem>
#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <nlohmann/json.hpp>

namespace GRIM {

// ============================================================================
// ID Generation
// ============================================================================

uint64_t UnifiedMemoryObject::generateID() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;
    return dis(gen);
}

std::string UnifiedMemoryObject::generateUUID() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFFFF);

    auto to_hex = [](uint32_t val, int width) {
   std::stringstream ss;
        ss << std::hex << std::setw(width) << std::setfill('0') << val;
        return ss.str();
    };

    return to_hex(dis(gen), 8) + "-" +
    to_hex(dis(gen) >> 16, 4) + "-" +
           to_hex(dis(gen) >> 16, 4) + "-" +
        to_hex(dis(gen) >> 16, 4) + "-" +
           to_hex(dis(gen), 8) + to_hex(dis(gen) >> 16, 4);
}

// ============================================================================
// JSON Serialization (Backward Compatibility)
// ============================================================================

std::string UnifiedMemoryObject::toJSON() const {
    nlohmann::json j;
    
    j["id"] = id;
    j["timestamp"] = timestamp;
    j["domain"] = toString(domain, DomainNames);
    j["type"] = toString(type, TypeNames);
    j["context"] = toString(context, ContextNames);
    j["comm_type"] = toString(comm_type, CommTypeNames);
    j["modality"] = toString(modality, ModalityNames);
    j["confidence"] = confidence;
    j["importance"] = importance;
    j["recency_weight"] = recency_weight;
    j["raw"] = raw;
    j["normalized"] = normalized;
    j["tags"] = tags;
    j["parent_id"] = parent_id;
    j["related_ids"] = related_ids;
    
    return j.dump(2);
}

UnifiedMemoryObject UnifiedMemoryObject::fromJSON(const std::string& jsonStr) {
    nlohmann::json j = nlohmann::json::parse(jsonStr);
    
    UnifiedMemoryObject obj;
    obj.id = j.value("id", generateID());
    obj.timestamp = j.value("timestamp", static_cast<uint64_t>(std::time(nullptr)));
    
 obj.domain = fromString(j.value("domain", "base"), DomainNames, MemoryDomain::BASE);
    obj.type = fromString(j.value("type", "string"), TypeNames, TypeTag::STRING);
    obj.context = fromString(j.value("context", "conversation"), ContextNames, ContextType::CONVERSATION);
    obj.comm_type = fromString(j.value("comm_type", "unknown"), CommTypeNames, CommType::UNKNOWN);
    obj.modality = fromString(j.value("modality", "text"), ModalityNames, Modality::TEXT);
  
    obj.confidence = j.value("confidence", 1.0f);
 obj.importance = j.value("importance", 1.0f);
    obj.recency_weight = j.value("recency_weight", 1.0f);
    obj.raw = j.value("raw", "");
    obj.normalized = j.value("normalized", "");
    obj.tags = j.value("tags", std::vector<std::string>{});
    obj.parent_id = j.value("parent_id", 0ULL);
    obj.related_ids = j.value("related_ids", std::vector<uint64_t>{});
    
    return obj;
}

// ============================================================================
// FlatBuffers Serialization (Zero-Copy)
// ============================================================================

std::vector<uint8_t> UnifiedMemoryObject::toFlatBuffer() const {
    flatbuffers::FlatBufferBuilder builder(1024);
    
    // Convert strings
    auto raw_fb = builder.CreateString(raw);
    auto normalized_fb = builder.CreateString(normalized);
    
    // Convert tags
    std::vector<flatbuffers::Offset<flatbuffers::String>> tag_offsets;
    for (const auto& tag : tags) {
        tag_offsets.push_back(builder.CreateString(tag));
    }
    auto tags_fb = builder.CreateVector(tag_offsets);
    
    // Convert embedding
    auto embedding_fb = builder.CreateVector(embedding.data(), embedding.size());
    
    // Convert related IDs
    auto related_ids_fb = builder.CreateVector(related_ids);
    
  // Build record
    auto record = GRIM::Memory::CreateMemoryRecord(builder,
 id,
        timestamp,
        static_cast<GRIM::Memory::MemoryDomain>(domain),
        static_cast<GRIM::Memory::TypeTag>(type),
      static_cast<GRIM::Memory::ContextType>(context),
        static_cast<GRIM::Memory::CommType>(comm_type),
        static_cast<GRIM::Memory::Modality>(modality),
      raw_fb,
        normalized_fb,
        tags_fb,
        confidence,
      importance,
     recency_weight,
        embedding_fb,
        parent_id,
        related_ids_fb
    );
    
    builder.Finish(record);
    
    return std::vector<uint8_t>(builder.GetBufferPointer(), 
        builder.GetBufferPointer() + builder.GetSize());
}

UnifiedMemoryObject UnifiedMemoryObject::fromFlatBuffer(const uint8_t* data, size_t size) {
    auto store = GRIM::Memory::GetMemoryStore(data);

    UnifiedMemoryObject obj;

    // Get the first record (for single-record deserialization)
    if (!store || !store->records() || store->records()->size() == 0) {
        return obj;  // Empty object
    }

    const auto* record = store->records()->Get(0);

    obj.id = record->id();
    obj.timestamp = record->timestamp();

    obj.domain = static_cast<MemoryDomain>(record->domain());
    obj.type = static_cast<TypeTag>(record->type());
    obj.context = static_cast<ContextType>(record->context());
    obj.comm_type = static_cast<CommType>(record->comm_type());
    obj.modality = static_cast<Modality>(record->modality());

    obj.raw = record->raw() ? record->raw()->str() : "";
    obj.normalized = record->normalized() ? record->normalized()->str() : "";

    if (record->tags()) {
        for (const auto* tag : *record->tags()) {
            obj.tags.push_back(tag->str());
        }
    }

    obj.confidence = record->confidence();
    obj.importance = record->importance();
    obj.recency_weight = record->recency_weight();

    if (record->embedding()) {
        size_t count = std::min(static_cast<size_t>(record->embedding()->size()), obj.embedding.size());
        std::copy_n(record->embedding()->begin(), count, obj.embedding.begin());
    }

    obj.parent_id = record->parent_id();

    if (record->related_ids()) {
        obj.related_ids.assign(record->related_ids()->begin(), record->related_ids()->end());
    }

    return obj;
}

// ============================================================================
// UnifiedMemoryStorage Implementation
// ============================================================================

UnifiedMemoryStorage::UnifiedMemoryStorage() {}

UnifiedMemoryStorage::~UnifiedMemoryStorage() {
    shutdown();
}

void UnifiedMemoryStorage::initialize(const std::string& storagePath) {
    std::lock_guard<std::mutex> lock(mtx);
    
    this->storagePath = storagePath;
    
    // Determine file paths
    std::filesystem::path basePath(storagePath);
    basePath.replace_extension();  // Remove extension
    
 jsonPath = basePath.string() + ".json";
    fbPath = basePath.string() + ".fb";
    indexPath = basePath.string() + ".idx";
    
    // Try to load from FlatBuffer first (faster)
    if (std::filesystem::exists(fbPath)) {
  LOG_DEBUG("UnifiedMemory", "Loading from FlatBuffer: " + fbPath);
        loadFromFlatBuffer();
    }
    // Fall back to JSON (backward compatibility)
    else if (std::filesystem::exists(jsonPath)) {
        LOG_DEBUG("UnifiedMemory", "Loading from JSON: " + jsonPath);
      loadFromDisk();
    }
    else {
        LOG_DEBUG("UnifiedMemory", "No existing memory file, starting fresh");
    }

    initialized_ = true;
    shutdown_complete_ = false;
}

void UnifiedMemoryStorage::shutdown() {
    std::lock_guard<std::mutex> lock(mtx);
    if (!initialized_ || shutdown_complete_) return;

    saveToDisk();
    saveToFlatBuffer();
    shutdown_complete_ = true;
}

void UnifiedMemoryStorage::flush() {
    std::lock_guard<std::mutex> lock(mtx);
    if (!initialized_ || shutdown_complete_) return;
    saveToDisk();
    saveToFlatBuffer();
}

void UnifiedMemoryStorage::loadFromDisk() {
    // Load from JSON (legacy format)
    std::ifstream ifs(jsonPath);
    if (!ifs.is_open()) return;
    
  nlohmann::json j;
    ifs >> j;
    
    if (!j.contains("memories")) return;

    for (auto& mem : j["memories"]) {
        UnifiedMemoryObject obj = UnifiedMemoryObject::fromJSON(mem.dump());
        longTerm[obj.id] = obj;
        updateIndex(obj);
        updateTagIndex(obj);
    }
    
    LOG_DEBUG("UnifiedMemory", "Loaded " + std::to_string(longTerm.size()) + " memories from JSON");
}

void UnifiedMemoryStorage::saveToDisk() {
 // Save to JSON (backward compatibility) — atomic write
    if (jsonPath.empty()) return;
    
    nlohmann::json j;
    j["memories"] = nlohmann::json::array();
    
    for (const auto& [id, obj] : longTerm) {
        j["memories"].push_back(nlohmann::json::parse(obj.toJSON()));
  }
    
    try {
        AtomicWriter::writeString(jsonPath, j.dump(2));
        LOG_DEBUG("UnifiedMemory", "Saved " + std::to_string(longTerm.size()) + " memories to JSON (atomic)");
    } catch (const std::exception& e) {
        LOG_ERROR("UnifiedMemory", std::string("Atomic JSON save failed: ") + e.what());
    }
}

void UnifiedMemoryStorage::saveToFlatBuffer() {
 if (fbPath.empty()) return;
    
    flatbuffers::FlatBufferBuilder builder(64 * 1024);
    
    // Build metadata
    auto metadata = GRIM::Memory::CreateMemoryMetadata(builder,
        1,  // schema_version
  std::time(nullptr),  // build_time
        longTerm.size(),  // record_count
   0  // total_size_bytes (calculated later)
 );
  
    // Build records
    std::vector<flatbuffers::Offset<GRIM::Memory::MemoryRecord>> record_offsets;
    for (const auto& [id, obj] : longTerm) {
     auto raw_fb = builder.CreateString(obj.raw);
   auto normalized_fb = builder.CreateString(obj.normalized);
      
      std::vector<flatbuffers::Offset<flatbuffers::String>> tag_offsets;
 for (const auto& tag : obj.tags) {
  tag_offsets.push_back(builder.CreateString(tag));
        }
        auto tags_fb = builder.CreateVector(tag_offsets);
      
        auto embedding_fb = builder.CreateVector(obj.embedding.data(), obj.embedding.size());
    auto related_ids_fb = builder.CreateVector(obj.related_ids);
        
 auto record = GRIM::Memory::CreateMemoryRecord(builder,
       obj.id, obj.timestamp,
            static_cast<GRIM::Memory::MemoryDomain>(obj.domain),
        static_cast<GRIM::Memory::TypeTag>(obj.type),
   static_cast<GRIM::Memory::ContextType>(obj.context),
 static_cast<GRIM::Memory::CommType>(obj.comm_type),
      static_cast<GRIM::Memory::Modality>(obj.modality),
          raw_fb, normalized_fb,
   tags_fb, obj.confidence, obj.importance, obj.recency_weight,
            embedding_fb, obj.parent_id, related_ids_fb
        );
        
        record_offsets.push_back(record);
    }
    
    auto records_fb = builder.CreateVector(record_offsets);
    
    // Build store
    auto store = GRIM::Memory::CreateMemoryStore(builder,
        metadata,
    records_fb
    );
    
    builder.Finish(store);
    
    // Write to file — atomic write
    try {
        AtomicWriter::write(fbPath, builder.GetBufferPointer(), builder.GetSize());
        LOG_DEBUG("UnifiedMemory", "Saved " + std::to_string(longTerm.size()) + " memories to FlatBuffer (atomic)");
    } catch (const std::exception& e) {
        LOG_ERROR("UnifiedMemory", std::string("Atomic FlatBuffer save failed: ") + e.what());
    }
}

void UnifiedMemoryStorage::loadFromFlatBuffer() {
    std::ifstream ifs(fbPath, std::ios::binary);
    if (!ifs.is_open()) return;
  
    // Read entire file
    ifs.seekg(0, std::ios::end);
    size_t fileSize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    
    flatBufferData.resize(fileSize);
    ifs.read(reinterpret_cast<char*>(flatBufferData.data()), fileSize);
    
    // Parse FlatBuffer
    auto store = GRIM::Memory::GetMemoryStore(flatBufferData.data());
    
    if (store->records()) {
        for (const auto* record : *store->records()) {
   UnifiedMemoryObject obj;
       obj.id = record->id();
     obj.timestamp = record->timestamp();
        obj.domain = static_cast<MemoryDomain>(record->domain());
            obj.type = static_cast<TypeTag>(record->type());
 obj.context = static_cast<ContextType>(record->context());
         obj.comm_type = static_cast<CommType>(record->comm_type());
            obj.modality = static_cast<Modality>(record->modality());
            obj.raw = record->raw() ? record->raw()->str() : "";
            obj.normalized = record->normalized() ? record->normalized()->str() : "";
   
            if (record->tags()) {
                for (const auto* tag : *record->tags()) {
          obj.tags.push_back(tag->str());
           }
  }
            
            obj.confidence = record->confidence();
    obj.importance = record->importance();
    obj.recency_weight = record->recency_weight();
  obj.parent_id = record->parent_id();
   
   if (record->related_ids()) {
      obj.related_ids.assign(record->related_ids()->begin(), record->related_ids()->end());
            }
   
      longTerm[obj.id] = obj;
  updateIndex(obj);
        updateTagIndex(obj);
        }
    }
    
    LOG_DEBUG("UnifiedMemory", "Loaded " + std::to_string(longTerm.size()) + " memories from FlatBuffer");
}

void UnifiedMemoryStorage::storeShortTerm(const UnifiedMemoryObject& obj) {
    std::lock_guard<std::mutex> lock(mtx);
    
    if (shortTerm.size() >= SHORT_TERM_MAX) {
    shortTerm.erase(shortTerm.begin());
    }
    
    shortTerm.push_back(obj);
}

void UnifiedMemoryStorage::storeLongTerm(const UnifiedMemoryObject& obj) {
    std::lock_guard<std::mutex> lock(mtx);
    
    longTerm[obj.id] = obj;
    updateIndex(obj);
    updateTagIndex(obj);
    
    // Auto-save (can be optimized to batch)
    saveToDisk();
    saveToFlatBuffer();
}

void UnifiedMemoryStorage::updateIndex(const UnifiedMemoryObject& obj) {
  // Update type index
    typeIndex[obj.type].push_back(obj.id);
    
    // Update domain index
    domainIndex[obj.domain].push_back(obj.id);
}

void UnifiedMemoryStorage::updateTagIndex(const UnifiedMemoryObject& obj) {
    for (const auto& tag : obj.tags) {
        tagIndex[tag].push_back(obj.id);
    }
}

std::optional<UnifiedMemoryObject> UnifiedMemoryStorage::getById(uint64_t id) {
std::lock_guard<std::mutex> lock(mtx);
    
    auto it = longTerm.find(id);
    if (it != longTerm.end()) {
        return it->second;
    }
    
    for (const auto& obj : shortTerm) {
  if (obj.id == id) {
    return obj;
     }
    }
    
    return std::nullopt;
}

std::vector<UnifiedMemoryObject> UnifiedMemoryStorage::search(const std::string& query, int maxResults) {
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<UnifiedMemoryObject> results;
    
    auto match = [&](const UnifiedMemoryObject& obj) {
   return obj.raw.find(query) != std::string::npos ||
             obj.normalized.find(query) != std::string::npos;
    };
    
    // Search short-term first (more recent)
    for (auto it = shortTerm.rbegin(); it != shortTerm.rend() && results.size() < maxResults; ++it) {
        if (match(*it)) {
       results.push_back(*it);
        }
    }
    
    // Then long-term
    for (const auto& [id, obj] : longTerm) {
        if (match(obj)) {
results.push_back(obj);
      if (results.size() >= maxResults) break;
        }
    }
    
return results;
}

std::vector<UnifiedMemoryObject> UnifiedMemoryStorage::getByTag(const std::string& tag) {
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<UnifiedMemoryObject> results;
    
    auto it = tagIndex.find(tag);
    if (it != tagIndex.end()) {
     for (uint64_t id : it->second) {
         auto memIt = longTerm.find(id);
      if (memIt != longTerm.end()) {
                results.push_back(memIt->second);
   }
        }
    }
    
    return results;
}

std::vector<UnifiedMemoryObject> UnifiedMemoryStorage::getByTags(const std::vector<std::string>& tags, bool matchAll) {
    if (tags.empty()) return {};
    
    std::lock_guard<std::mutex> lock(mtx);
    
if (matchAll) {
      // AND logic: memory must have ALL tags
     std::set<uint64_t> result_ids;
        bool first = true;
        
        for (const auto& tag : tags) {
            auto it = tagIndex.find(tag);
         if (it == tagIndex.end()) {
        return {};  // Tag not found = no matches
      }
  
          std::set<uint64_t> tag_ids(it->second.begin(), it->second.end());
            
            if (first) {
              result_ids = tag_ids;
    first = false;
          } else {
      std::set<uint64_t> intersection;
    std::set_intersection(result_ids.begin(), result_ids.end(),
         tag_ids.begin(), tag_ids.end(),
std::inserter(intersection, intersection.begin()));
     result_ids = intersection;
            }
        }
        
        std::vector<UnifiedMemoryObject> results;
        for (uint64_t id : result_ids) {
         auto memIt = longTerm.find(id);
            if (memIt != longTerm.end()) {
     results.push_back(memIt->second);
          }
        }
        return results;
    } else {
        // OR logic: memory can have ANY tag
        std::set<uint64_t> result_ids;
        
    for (const auto& tag : tags) {
     auto it = tagIndex.find(tag);
     if (it != tagIndex.end()) {
 result_ids.insert(it->second.begin(), it->second.end());
    }
    }
        
        std::vector<UnifiedMemoryObject> results;
  for (uint64_t id : result_ids) {
            auto memIt = longTerm.find(id);
   if (memIt != longTerm.end()) {
    results.push_back(memIt->second);
   }
     }
        return results;
    }
}

std::vector<UnifiedMemoryObject> UnifiedMemoryStorage::getByType(TypeTag type) {
std::lock_guard<std::mutex> lock(mtx);
    std::vector<UnifiedMemoryObject> results;
    
    auto it = typeIndex.find(type);
    if (it != typeIndex.end()) {
        for (uint64_t id : it->second) {
            auto memIt = longTerm.find(id);
          if (memIt != longTerm.end()) {
      results.push_back(memIt->second);
   }
        }
    }
    
    return results;
}

std::vector<UnifiedMemoryObject> UnifiedMemoryStorage::getByDomain(MemoryDomain domain) {
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<UnifiedMemoryObject> results;
    
    auto it = domainIndex.find(domain);
    if (it != domainIndex.end()) {
 for (uint64_t id : it->second) {
            auto memIt = longTerm.find(id);
  if (memIt != longTerm.end()) {
  results.push_back(memIt->second);
            }
        }
    }
    
    return results;
}

std::optional<UnifiedMemoryObject> UnifiedMemoryStorage::findLearnedCommand(const std::string& phrase) {
    std::lock_guard<std::mutex> lock(mtx);
    
    auto match = [&](const UnifiedMemoryObject& obj) {
     if (std::find(obj.tags.begin(), obj.tags.end(), "learned") == obj.tags.end() ||
         std::find(obj.tags.begin(), obj.tags.end(), "command") == obj.tags.end()) return false;
        return obj.raw == phrase || obj.normalized == phrase || obj.raw.find(phrase) != std::string::npos;
    };
    
    for (const auto& [id, obj] : longTerm) {
        if (match(obj)) return obj;
    }
    
    for (auto it = shortTerm.rbegin(); it != shortTerm.rend(); ++it) {
     if (match(*it)) return *it;
    }
    
    return std::nullopt;
}

std::vector<UnifiedMemoryObject> UnifiedMemoryStorage::getAllLearnedCommands() {
    return getByTags({"learned", "command"}, true);
}

void UnifiedMemoryStorage::storeLearnedCommand(const std::string& phrase, const std::string& action, float confidence) {
    UnifiedMemoryObject obj;
    obj.id = UnifiedMemoryObject::generateID();
  obj.timestamp = std::time(nullptr);
    obj.domain = MemoryDomain::USER;
    obj.type = TypeTag::STRING;
    obj.context = ContextType::CONVERSATION;
    obj.raw = phrase;
    obj.normalized = action;
  obj.confidence = confidence;
    obj.tags = {"learned", "command"};
    
    storeLongTerm(obj);
}

void UnifiedMemoryStorage::decay(float rate) {
    std::lock_guard<std::mutex> lock(mtx);
    
    uint64_t now = std::time(nullptr);
    
    for (auto& [id, obj] : longTerm) {
        float ageDays = static_cast<float>((now - obj.timestamp) / 86400.0);
        obj.confidence -= rate * ageDays;
        if (obj.confidence < 0.2f) {
       obj.confidence = 0.2f;
        }
    }
}

void UnifiedMemoryStorage::rebuildIndex() {
    std::lock_guard<std::mutex> lock(mtx);
    
    tagIndex.clear();
    typeIndex.clear();
    domainIndex.clear();
    
    for (const auto& [id, obj] : longTerm) {
        updateIndex(obj);
        updateTagIndex(obj);
    }
    
    LOG_DEBUG("UnifiedMemory", "Rebuilt indexes");
}

UnifiedMemoryStorage::Stats UnifiedMemoryStorage::getStats() const {
    std::lock_guard<std::mutex> lock(mtx);
    
    Stats stats;
  stats.short_term_count = shortTerm.size();
    stats.long_term_count = longTerm.size();
    stats.total_records = stats.short_term_count + stats.long_term_count;
    stats.indexed_tags = tagIndex.size();
    
    float total_confidence = 0.0f;
    for (const auto& [id, obj] : longTerm) {
        total_confidence += obj.confidence;
    }
    
    if (stats.long_term_count > 0) {
        stats.avg_confidence = total_confidence / stats.long_term_count;
    }
    
    return stats;
}

std::vector<MemoryRetrievalHit> UnifiedMemoryStorage::semanticSearch(const RetrievalQuery& query) {
    // TODO: Implement semantic search with embeddings
    // For now, fallback to text search
    auto textResults = search(query.text, query.max_results);
    
    std::vector<MemoryRetrievalHit> results;
    for (const auto& obj : textResults) {
        MemoryRetrievalHit result;
        result.score = obj.confidence;  // Placeholder
        result.memory = obj;
        results.push_back(result);
    }
    
    return results;
}

// Legacy compatibility stubs
bool UnifiedMemoryStorage::recentlyModified(const std::string& key, int seconds) {
    return false;  // TODO: Implement if needed
}

std::time_t UnifiedMemoryStorage::getLastModified(const std::string& key) {
    return 0;  // TODO: Implement if needed
}

void UnifiedMemoryStorage::compactStorage() {
    std::lock_guard<std::mutex> lock(mtx);
    
    // Remove duplicates and low-confidence memories
    std::vector<uint64_t> toRemove;
    
    for (const auto& [id, obj] : longTerm) {
if (obj.confidence < 0.1f) {
       toRemove.push_back(id);
   }
    }
    
    for (uint64_t id : toRemove) {
        longTerm.erase(id);
    }
    
    if (!toRemove.empty()) {
        rebuildIndex();
        LOG_DEBUG("UnifiedMemory", "Compacted storage, removed " + std::to_string(toRemove.size()) + " low-confidence memories");
    }
}

} // namespace GRIM
