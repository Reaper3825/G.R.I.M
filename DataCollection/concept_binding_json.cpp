#include "concept_binding_adapter.hpp"

#include <nlohmann/json.hpp>

#include <utility>

namespace GRIM::DataCollection {
namespace {

using json = nlohmann::json;

GeneratorValueType parseValueType(const std::string& name) {
    if (name == "integer" || name == "int") return GeneratorValueType::Integer;
    if (name == "float" || name == "number") return GeneratorValueType::Float;
    if (name == "boolean" || name == "bool") return GeneratorValueType::Boolean;
    if (name == "string") return GeneratorValueType::String;
    return GeneratorValueType::Text;
}

std::string scalarToString(const json& value) {
    if (value.is_string()) return value.get<std::string>();
    if (value.is_boolean()) return value.get<bool>() ? "true" : "false";
    if (value.is_number()) return value.dump();
    return {};
}

json parseObjectResponse(const std::string& response) {
    json parsed = json::parse(response, nullptr, false);
    if (parsed.is_object()) return parsed;

    const size_t first = response.find('{');
    const size_t last = response.rfind('}');
    if (first == std::string::npos || last == std::string::npos || last <= first) {
        return json{};
    }
    parsed = json::parse(response.substr(first, last - first + 1), nullptr, false);
    return parsed.is_object() ? parsed : json{};
}

} // namespace

GeneratorBindingSet OllamaConceptBindingAdapter::parseBindingResponse(
    const std::string& documentId,
    const std::string& adapterIdValue,
    const std::string& response) {
    GeneratorBindingSet bindings;
    bindings.documentId = documentId;
    bindings.adapterId = adapterIdValue;
    bindings.rawResponse = response;

    const json root = parseObjectResponse(response);
    if (!root.is_object()) {
        bindings.error = "Model response is not a JSON object";
        return bindings;
    }

    bindings.question = root.value("question", std::string{});
    bindings.targetState = root.value("target_state", std::string{});
    bindings.answer = root.value("answer", std::string{});

    const char* reasoningKey = root.contains("reasoning_steps")
        ? "reasoning_steps" : "intermediates";
    if (root.contains(reasoningKey) && root[reasoningKey].is_array()) {
        for (const auto& step : root[reasoningKey]) {
            if (step.is_string()) bindings.reasoningSteps.push_back(step.get<std::string>());
        }
    }

    if (root.contains("evidence") && root["evidence"].is_array()) {
        for (const auto& item : root["evidence"]) {
            if (!item.is_object()) continue;
            bindings.evidence.push_back({
                item.value("criterion", std::string{}),
                item.value("quote", item.value("evidence", std::string{}))
            });
        }
    }

    if (root.contains("atoms") && root["atoms"].is_array()) {
        for (size_t index = 0; index < root["atoms"].size(); ++index) {
            const auto& item = root["atoms"][index];
            if (!item.is_object() || !item.contains("value")) continue;
            GeneratorAtomBinding atom;
            atom.id = item.value("id", "atom_" + std::to_string(index));
            atom.type = parseValueType(item.value("type", std::string("text")));
            atom.value = scalarToString(item["value"]);
            atom.unit = item.value("unit", std::string{});
            atom.sourceQuote = item.value("source_quote", std::string{});
            atom.derived = item.value("derived", false);
            bindings.atoms.push_back(std::move(atom));
        }
    }
    return bindings;
}

} // namespace GRIM::DataCollection
