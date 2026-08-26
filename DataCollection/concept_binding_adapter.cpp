#include "concept_binding_adapter.hpp"

#include "MMO/Backends/OllamaBackend.hpp"

#include <nlohmann/json.hpp>

#include <sstream>
#include <stdexcept>
#include <utility>

namespace GRIM::DataCollection {
namespace {

using json = nlohmann::json;

} // namespace

OllamaConceptBindingAdapter::OllamaConceptBindingAdapter(
    OllamaConceptBindingConfig config)
    : config_(std::move(config)) {
    if (config_.url.empty()) {
        throw std::runtime_error("Ollama concept adapter requires a URL");
    }
    if (config_.model.empty()) {
        throw std::runtime_error("Ollama concept adapter requires a model");
    }
    backend_ = std::make_unique<GRIM::MMO::OllamaBackend>(
        config_.url, "datahub-concept-generator", config_.model);
}

OllamaConceptBindingAdapter::~OllamaConceptBindingAdapter() = default;

std::string OllamaConceptBindingAdapter::adapterId() const {
    return "ollama:" + config_.model;
}

bool OllamaConceptBindingAdapter::isAvailable() const {
    return backend_ && backend_->isAvailable();
}

std::string OllamaConceptBindingAdapter::buildSystemPrompt(
    const GeneratorFrameDefinition& recipe) const {
    json slots = json::array();
    for (const auto& slot : recipe.slots) {
        slots.push_back({
            {"id", slot.id},
            {"type", ConceptGenerator::slotTypeName(slot.type)},
            {"description", slot.description},
            {"required", slot.required},
            {"grounded", slot.grounded},
            {"min_items", slot.minItems},
            {"max_items", slot.maxItems}
        });
    }

    std::ostringstream prompt;
    prompt
        << "You fill a typed ConceptBlock recipe from one source document.\n"
        << "Read and understand the document, then generate the semantic content "
           "for every required slot. Do not merely copy the entire document.\n"
        << "Return ONLY one strict JSON object. Do not use markdown fences.\n"
        << "Do not emit state_0. Do not add facts unsupported by the document.\n"
        << "Every evidence.quote and non-derived atom.source_quote must be an exact "
           "substring of the source document.\n"
        << "When text refers to an atom, write {{atom:ATOM_ID}} instead of repeating "
           "the raw value. The deterministic renderer adds atom tags and units.\n\n"
        << "Recipe slots:\n" << slots.dump(2) << "\n\n"
        << "Required output shape:\n"
        << R"({
  "question": "...",
  "target_state": "...",
  "reasoning_steps": ["..."],
  "answer": "...",
  "evidence": [
    {"criterion": "...", "quote": "exact document quote"}
  ],
  "atoms": [
    {
      "id": "stable_short_id",
      "type": "integer|float|boolean|string",
      "value": 0,
      "unit": "optional unit",
      "source_quote": "exact document quote",
      "derived": false
    }
  ]
})";
    return prompt.str();
}

std::string OllamaConceptBindingAdapter::buildUserPrompt(
    const GeneratorDocument& document,
    const GeneratorMemoryContext& memory) const {
    std::string text = document.text;
    if (config_.maxInputChars > 0
        && text.size() > static_cast<size_t>(config_.maxInputChars)) {
        text.resize(static_cast<size_t>(config_.maxInputChars));
    }

    std::ostringstream prompt;
    prompt << "DOCUMENT ID: " << document.id << "\n";
    if (!document.title.empty()) prompt << "TITLE: " << document.title << "\n";
    prompt << "\nSOURCE DOCUMENT:\n" << text;
    if (!memory.serializedContext.empty()) {
        prompt << "\n\nOPTIONAL MEMORY CONTEXT:\n"
               << memory.serializedContext
               << "\nMemory is supplemental context. Do not present a memory claim as "
                  "document evidence unless it also occurs in the source document.";
    }
    return prompt.str();
}

GeneratorBindingSet OllamaConceptBindingAdapter::fill(
    const GeneratorDocument& document,
    const GeneratorFrameDefinition& recipe,
    const GeneratorMemoryContext& memory,
    const GeneratorAdapterOptions& options) {
    GeneratorBindingSet failed;
    failed.documentId = document.id;
    failed.adapterId = adapterId();
    if (!backend_) {
        failed.error = "Ollama backend is not initialized";
        return failed;
    }
    if (document.text.empty()) {
        failed.error = "Document text is empty";
        return failed;
    }

    GRIM::MMO::GenerationOptions generation;
    generation.max_tokens = options.maxTokens;
    generation.temperature = options.temperature;
    generation.top_p = options.topP;
    generation.top_k = options.topK;
    generation.timeout_ms = options.timeoutMs;
    generation.seed = static_cast<int64_t>(options.seed & 0x7FFFFFFFFFFFFFFFULL);

    const std::vector<GRIM::MMO::HistoryEntry> history = {
        {"system", buildSystemPrompt(recipe)}
    };
    const auto result = backend_->generateWithHistory(
        buildUserPrompt(document, memory), history, generation);
    if (!result.success) {
        failed.error = result.error.empty() ? "Ollama returned no binding result" : result.error;
        return failed;
    }
    return parseBindingResponse(document.id, adapterId(), result.text);
}

} // namespace GRIM::DataCollection
