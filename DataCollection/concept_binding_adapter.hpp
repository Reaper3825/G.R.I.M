//======================================================//
//  Concept binding adapters.
//
//  The UI and deterministic generator depend only on this
//  interface. Ollama is the first implementation; native
//  GRIM and memory-aware adapters can implement the same
//  contract later.
//======================================================//

#pragma once

#include "concept_generator.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace GRIM::MMO {
class OllamaBackend;
}

namespace GRIM::DataCollection {

struct GeneratorMemoryContext {
    std::string              serializedContext;
    std::vector<std::string> memoryIds;
};

struct GeneratorAdapterOptions {
    uint64_t seed = 0;
    float    temperature = 0.2f;
    float    topP = 0.9f;
    int      topK = 40;
    int      maxTokens = 2048;
    int      timeoutMs = 60000;
};

class IConceptBindingAdapter {
public:
    virtual ~IConceptBindingAdapter() = default;

    virtual GeneratorBindingSet fill(
        const GeneratorDocument& document,
        const GeneratorFrameDefinition& recipe,
        const GeneratorMemoryContext& memory,
        const GeneratorAdapterOptions& options) = 0;

    virtual bool isAvailable() const = 0;
    virtual std::string adapterId() const = 0;
};

struct OllamaConceptBindingConfig {
    std::string url = "http://127.0.0.1:11434";
    std::string model;
    int         maxInputChars = 12000;
};

class OllamaConceptBindingAdapter final : public IConceptBindingAdapter {
public:
    explicit OllamaConceptBindingAdapter(OllamaConceptBindingConfig config);
    ~OllamaConceptBindingAdapter() override;

    GeneratorBindingSet fill(
        const GeneratorDocument& document,
        const GeneratorFrameDefinition& recipe,
        const GeneratorMemoryContext& memory,
        const GeneratorAdapterOptions& options) override;

    bool isAvailable() const override;
    std::string adapterId() const override;

    static GeneratorBindingSet parseBindingResponse(
        const std::string& documentId,
        const std::string& adapterId,
        const std::string& response);

private:
    std::string buildSystemPrompt(const GeneratorFrameDefinition& recipe) const;
    std::string buildUserPrompt(const GeneratorDocument& document,
                                const GeneratorMemoryContext& memory) const;

    OllamaConceptBindingConfig                config_;
    std::unique_ptr<GRIM::MMO::OllamaBackend> backend_;
};

} // namespace GRIM::DataCollection
