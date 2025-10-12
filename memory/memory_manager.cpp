#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <iomanip>
#include "memory_manager.hpp"

using json = nlohmann::json;

namespace GRIM {
    


// =====================================
// UUID GENERATOR
// =====================================

std::string MemoryObject::generateUUID() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFFFF);

    auto to_hex = [](uint32_t val, int width) {
        std::stringstream ss;
        ss << std::hex << std::setw(width) << std::setfill('0') << val;
        return ss.str();
    };

    // 8-4-4-4-12 layout
    return to_hex(dis(gen), 8) + "-" +
           to_hex(dis(gen) >> 16, 4) + "-" +
           to_hex(dis(gen) >> 16, 4) + "-" +
           to_hex(dis(gen) >> 16, 4) + "-" +
           to_hex(dis(gen), 8) + to_hex(dis(gen) >> 16, 4);
}

// =====================================
// SERIALIZATION TO JSON
// =====================================

std::string MemoryObject::toJSON() const {
    json j;

    j["id"] = id;
    j["timestamp"] = static_cast<long long>(timestamp);
    j["source"] = toString(source, SourceNames);
    j["type"] = toString(type, TypeNames);
    j["intent"] = toString(intent, IntentNames);
    j["context"] = toString(context, ContextNames);
    j["confidence"] = confidence;
    j["raw"] = raw;
    j["normalized"] = normalized;
    j["tags"] = tags;

    return j.dump(4); // pretty print
}

// =====================================
// DESERIALIZATION FROM JSON
// =====================================

MemoryObject MemoryObject::fromJSON(const std::string& jsonStr) {
    json j = json::parse(jsonStr);
    MemoryObject obj;

    obj.id = j.value("id", generateUUID());
    obj.timestamp = j.value("timestamp", std::time(nullptr));

    obj.source = fromString(j.value("source", "grim.internal"), SourceNames, SourceTag::GrimInternal);
    obj.type = fromString(j.value("type", "fact"), TypeNames, TypeTag::Fact);
    obj.intent = fromString(j.value("intent", "inform"), IntentNames, IntentTag::Inform);
    obj.context = fromString(j.value("context", "conversation"), ContextNames, ContextTag::Conversation);

    obj.confidence = j.value("confidence", 1.0f);
    obj.raw = j.value("raw", "");
    obj.normalized = j.value("normalized", "");
    obj.tags = j.value("tags", std::vector<std::string>{});

    return obj;
}


} // namespace GRIM
