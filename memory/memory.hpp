#pragma once
#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <atomic>

namespace GRIM {

// ---------- ENUMS ---------- //
enum class CommType      : uint8_t { COMMAND, QUESTION, BANTER, UNKNOWN };
enum class MemoryDomain  : uint8_t { BASE, USER, FIELD, FUSED };
enum class ImportanceType: uint8_t { LOW, MEDIUM, HIGH };
enum class Modality      : uint8_t { TEXT, AUDIO, VISION, SENSOR };

// ---------- METADATA ---------- //
struct MemoryMeta {
    uint32_t schema_version = 1;
    uint64_t build_time = 0;
    uint64_t record_count = 0;
    uint32_t checksum = 0;
    uint64_t parent_version = 0;
};

// ---------- MEMORY RECORD ---------- //
struct MemoryEvent {
    uint64_t id = 0;
    uint64_t timestamp = 0;
    CommType comm_type = CommType::UNKNOWN;
    std::string intent;
    std::vector<std::string> tags;
    float confidence = 1.0f;
    float importance = 1.0f;
    float recency_weight = 1.0f;
    uint64_t offset = 0;
    Modality modality = Modality::TEXT;
    std::array<float, 768> embedding{};
};

// ---------- INDEX ---------- //
struct IndexHeader {
    uint32_t magic = 0x47494D58; // 'GIMX'
    uint16_t version = 1;
    uint16_t dim = 768;
    uint64_t fb_version = 0;
    uint64_t count = 0;
};

struct IndexEntry {
    uint64_t id = 0;
    uint64_t offset = 0;
    std::array<float, 768> embedding{};
};

// ---------- CONTEXT ---------- //
struct ContextRecord {
    MemoryDomain domain = MemoryDomain::BASE;
    uint64_t id = 0;
    float relevance = 0.0f;
    const MemoryEvent* ptr = nullptr;
};

struct Context {
    std::vector<ContextRecord> entries;
    void deduplicate();
    void pruneLowConfidence(float threshold = 0.3f);
    void truncateToTokenBudget(size_t token_limit);
};

// ---------- RETRIEVAL ---------- //
struct QueryResult {
    uint64_t id = 0;
    uint64_t offset = 0;
    float score = 0.0f;
};

struct RetrievalQuery {
    CommType type = CommType::UNKNOWN;
    std::string text;
    std::vector<float> embedding;
    std::string intent;
    std::vector<std::string> tags;
};

// ---------- LEARNING ---------- //
struct ObservationLog {
    uint64_t id = 0;
    std::string input;
    std::string output;
    CommType comm_type = CommType::UNKNOWN;
    float reward = 0.0f;
    uint64_t timestamp = 0;
};

struct DeltaPackage {
    uint64_t parent_version = 0;
    std::vector<MemoryEvent> new_events;
};

// ---------- RESPONSE ---------- //
enum class Tone { Neutral, Informative, Playful };

struct ToneProfile {
    Tone tone = Tone::Neutral;
    float pitch_shift = 0.0f;
    float speech_rate = 1.0f;
    std::string style_token;
};

struct Response {
    std::string text;
    CommType type = CommType::UNKNOWN;
    float confidence = 1.0f;
    Tone tone = Tone::Neutral;
};

// ---------- FUSION-ERA PLACEHOLDERS ---------- //
// (to be implemented much later)
struct LatentMemory {
    // placeholder for tensor implementation
    void* tensor_ptr = nullptr;
    uint32_t dim = 768;
    uint64_t count = 0;
};

struct FusionState {
    LatentMemory memory;
    void* sensory_inputs = nullptr;
    void* user_profile_vector = nullptr;
    void* system_state_vector = nullptr;
};

struct FusionSnapshotHeader {
    uint32_t magic = 0x47524D42; // 'GRMB'
    uint16_t version = 1;
    uint64_t parameter_count = 0;
    uint64_t memory_count = 0;
    uint64_t build_time = 0;
};

} // namespace GRIM
