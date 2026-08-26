#include "concept_generator.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_set>

namespace GRIM::DataCollection {
namespace {

using json = nlohmann::json;

GeneratorValueType parseValueType(const std::string& name) {
    if (name == "integer" || name == "int") return GeneratorValueType::Integer;
    if (name == "float" || name == "number") return GeneratorValueType::Float;
    if (name == "boolean" || name == "bool") return GeneratorValueType::Boolean;
    if (name == "string") return GeneratorValueType::String;
    if (name == "mixed") return GeneratorValueType::Mixed;
    return GeneratorValueType::Text;
}

GeneratorSlotType parseSlotType(const std::string& name) {
    if (name == "text_array") return GeneratorSlotType::TextArray;
    if (name == "evidence_array") return GeneratorSlotType::EvidenceArray;
    if (name == "typed_atom_array") return GeneratorSlotType::TypedAtomArray;
    return GeneratorSlotType::Text;
}

GeneratorSegmentKind parseSegmentKind(const std::string& name) {
    if (name == "lexicon") return GeneratorSegmentKind::Lexicon;
    if (name == "binding") return GeneratorSegmentKind::Binding;
    return GeneratorSegmentKind::Literal;
}

uint64_t fnv1a(const std::string& text) {
    uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char c : text) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t splitmix64(uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

std::string scalarToString(const json& value) {
    if (value.is_string()) return value.get<std::string>();
    if (value.is_boolean()) return value.get<bool>() ? "true" : "false";
    if (value.is_number()) return value.dump();
    return value.dump();
}

std::string defaultLabel(const std::string& id) {
    std::string label = id;
    bool capitalize = true;
    for (char& c : label) {
        if (c == '_' || c == '-') {
            c = ' ';
            capitalize = true;
        } else if (capitalize) {
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            capitalize = false;
        }
    }
    return label;
}

std::string sentenceCase(std::string text) {
    bool capitalizeNext = true;
    bool inTag = false;
    for (size_t index = 0; index < text.size(); ++index) {
        char& c = text[index];
        if (c == '<') { inTag = true; continue; }
        if (c == '>') { inTag = false; continue; }
        if (inTag) continue;

        if (capitalizeNext && std::isalpha(static_cast<unsigned char>(c))) {
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            capitalizeNext = false;
        }
        if (c == '\n') {
            capitalizeNext = true;
        } else if ((c == '.' || c == '!' || c == '?')
                   && (index + 1 == text.size()
                       || std::isspace(static_cast<unsigned char>(text[index + 1])))) {
            capitalizeNext = true;
        }
    }
    return text;
}

std::string bindingText(const GeneratorBindingSet& bindings, const std::string& key) {
    if (key == "question") return bindings.question;
    if (key == "target_state") return bindings.targetState;
    if (key == "answer") return bindings.answer;
    return {};
}

void replaceAll(std::string& text, const std::string& needle, const std::string& replacement) {
    if (needle.empty()) return;
    size_t position = 0;
    while ((position = text.find(needle, position)) != std::string::npos) {
        text.replace(position, needle.size(), replacement);
        position += replacement.size();
    }
}

std::string applyAtomBindings(std::string text,
                              const std::vector<GeneratorAtomBinding>& atoms) {
    for (size_t index = 0; index < atoms.size(); ++index) {
        const auto& atom = atoms[index];
        const std::string rendered = ConceptGenerator::renderAtom(atom);
        if (!atom.id.empty()) {
            replaceAll(text, "{{atom:" + atom.id + "}}", rendered);
        }
        replaceAll(text, "{{atom:" + std::to_string(index) + "}}", rendered);
    }
    return text;
}

bool isInteger(const std::string& value) {
    try {
        size_t used = 0;
        const auto parsed = std::stoll(value, &used);
        (void)parsed;
        return !value.empty() && used == value.size();
    } catch (...) {
        return false;
    }
}

bool isFloat(const std::string& value) {
    try {
        size_t used = 0;
        const double parsed = std::stod(value, &used);
        return !value.empty() && used == value.size() && std::isfinite(parsed);
    } catch (...) {
        return false;
    }
}

} // namespace

const char* ConceptGenerator::valueTypeName(GeneratorValueType type) {
    switch (type) {
        case GeneratorValueType::Integer: return "integer";
        case GeneratorValueType::Float:   return "float";
        case GeneratorValueType::Boolean: return "boolean";
        case GeneratorValueType::String:  return "string";
        case GeneratorValueType::Mixed:   return "mixed";
        case GeneratorValueType::Text:
        default:                          return "text";
    }
}

const char* ConceptGenerator::slotTypeName(GeneratorSlotType type) {
    switch (type) {
        case GeneratorSlotType::TextArray:      return "text_array";
        case GeneratorSlotType::EvidenceArray:  return "evidence_array";
        case GeneratorSlotType::TypedAtomArray: return "typed_atom_array";
        case GeneratorSlotType::Text:
        default:                                return "text";
    }
}

std::string ConceptGenerator::renderAtom(const GeneratorAtomBinding& atom) {
    const char* tag = nullptr;
    switch (atom.type) {
        case GeneratorValueType::Integer: tag = "INT"; break;
        case GeneratorValueType::Float:   tag = "FLOAT"; break;
        case GeneratorValueType::Boolean: tag = "BOOL"; break;
        case GeneratorValueType::String:  tag = "STRING"; break;
        case GeneratorValueType::Text:
        case GeneratorValueType::Mixed:
        default:                          break;
    }

    std::string rendered = atom.value;
    if (tag) rendered = std::string("<") + tag + ">" + atom.value + "</" + tag + ">";
    if (!atom.unit.empty()) rendered += " " + atom.unit;
    return rendered;
}

bool ConceptGenerator::loadCatalog(const std::filesystem::path& path) {
    groups_.clear();
    frames_.clear();
    schemaVersion_ = 0;
    lastError_.clear();

    try {
        std::ifstream input(path);
        if (!input.is_open()) {
            lastError_ = "Could not open lexicon catalog: " + path.string();
            return false;
        }

        json root;
        input >> root;
        if (!root.is_object()) {
            lastError_ = "Lexicon catalog root must be an object";
            return false;
        }
        schemaVersion_ = root.value("schema_version", 1U);

        if (root.contains("groups") && root["groups"].is_array()) {
            for (const auto& groupJson : root["groups"]) {
                if (!groupJson.is_object()) continue;
                GeneratorLexiconGroup group;
                group.id = groupJson.value("id", std::string{});
                if (group.id.empty()) continue;
                group.label = groupJson.value("label", defaultLabel(group.id));
                group.type = parseValueType(groupJson.value("type", std::string("text")));
                group.required = groupJson.value("required", false);
                if (groupJson.contains("values") && groupJson["values"].is_array()) {
                    for (const auto& entryJson : groupJson["values"]) {
                        GeneratorLexiconEntry entry;
                        entry.type = group.type;
                        if (entryJson.is_string()) {
                            entry.text = entryJson.get<std::string>();
                        } else if (entryJson.is_object()) {
                            entry.text = entryJson.value("text", std::string{});
                            entry.type = parseValueType(
                                entryJson.value("type", std::string(valueTypeName(group.type))));
                        }
                        if (!entry.text.empty()) group.entries.push_back(std::move(entry));
                    }
                }
                groups_.push_back(std::move(group));
            }
        }

        if (root.contains("frames") && root["frames"].is_array()) {
            for (const auto& frameJson : root["frames"]) {
                if (!frameJson.is_object()) continue;
                GeneratorFrameDefinition frame;
                frame.id = frameJson.value("id", std::string{});
                if (frame.id.empty()) continue;
                frame.label = frameJson.value("label", defaultLabel(frame.id));
                frame.formatType = frameJson.value("format_type", std::string("chain_of_thought"));

                if (frameJson.contains("slots") && frameJson["slots"].is_array()) {
                    for (const auto& slotJson : frameJson["slots"]) {
                        if (!slotJson.is_object()) continue;
                        GeneratorSlotDefinition slot;
                        slot.id = slotJson.value("id", std::string{});
                        if (slot.id.empty()) continue;
                        slot.type = parseSlotType(slotJson.value("type", std::string("text")));
                        slot.description = slotJson.value("description", std::string{});
                        slot.required = slotJson.value("required", false);
                        slot.grounded = slotJson.value("grounded", false);
                        slot.minItems = slotJson.value("min_items", size_t(0));
                        slot.maxItems = slotJson.value("max_items", size_t(0));
                        frame.slots.push_back(std::move(slot));
                    }
                }

                if (frameJson.contains("render") && frameJson["render"].is_object()) {
                    for (auto fieldIt = frameJson["render"].begin();
                         fieldIt != frameJson["render"].end(); ++fieldIt) {
                        if (!fieldIt.value().is_array()) continue;
                        GeneratorFieldTemplate field;
                        field.target = fieldIt.key();
                        for (const auto& segmentJson : fieldIt.value()) {
                            if (!segmentJson.is_object()) continue;
                            GeneratorFrameSegment segment;
                            segment.kind = parseSegmentKind(
                                segmentJson.value("kind", std::string("literal")));
                            segment.key = segmentJson.value("key", std::string{});
                            segment.text = segmentJson.value("text", std::string{});
                            segment.fallback = segmentJson.value("fallback", std::string{});
                            segment.prefix = segmentJson.value("prefix", std::string{});
                            segment.suffix = segmentJson.value("suffix", std::string{});
                            segment.optional = segmentJson.value("optional", false);
                            field.segments.push_back(std::move(segment));
                        }
                        frame.fields.push_back(std::move(field));
                    }
                }
                frames_.push_back(std::move(frame));
            }
        }

        if (groups_.empty()) {
            lastError_ = "Lexicon catalog contains no usable groups";
            return false;
        }
        if (frames_.empty()) {
            lastError_ = "Lexicon catalog contains no generator frames";
            return false;
        }
        return true;
    } catch (const std::exception& error) {
        lastError_ = "Invalid lexicon catalog: " + std::string(error.what());
        groups_.clear();
        frames_.clear();
        return false;
    }
}

bool ConceptGenerator::loadDataSource(const std::filesystem::path& path) {
    documents_.clear();
    lastError_.clear();
    try {
        std::ifstream input(path);
        if (!input.is_open()) {
            lastError_ = "Could not open generator data source: " + path.string();
            return false;
        }

        json root;
        input >> root;
        const json* records = &root;
        if (root.is_object() && root.contains("documents")) records = &root["documents"];
        if (!records->is_array()) {
            lastError_ = "Generator data source must be a document array";
            return false;
        }

        for (size_t index = 0; index < records->size(); ++index) {
            const auto& recordJson = (*records)[index];
            GeneratorDocument document;
            document.id = "document_" + std::to_string(index);
            if (recordJson.is_string()) {
                document.text = recordJson.get<std::string>();
            } else if (recordJson.is_object()) {
                document.id = recordJson.value("id", document.id);
                document.title = recordJson.value("title", std::string{});
                document.text = recordJson.value("text",
                    recordJson.value("content", recordJson.value("document_text", std::string{})));
                if (recordJson.contains("metadata")) {
                    document.metadataJson = recordJson["metadata"].dump();
                }
            }
            if (!document.text.empty()) documents_.push_back(std::move(document));
        }
        return true;
    } catch (const std::exception& error) {
        lastError_ = "Invalid generator data source: " + std::string(error.what());
        documents_.clear();
        return false;
    }
}

bool ConceptGenerator::validateBindings(const GeneratorDocument& document,
                                        const GeneratorFrameDefinition& frame,
                                        GeneratorBindingSet& bindings) const {
    bindings.valid = false;
    bindings.error.clear();
    auto fail = [&](const std::string& error) {
        bindings.error = error;
        return false;
    };

    for (const auto& slot : frame.slots) {
        size_t count = 0;
        if (slot.id == "question") count = bindings.question.empty() ? 0 : 1;
        else if (slot.id == "target_state") count = bindings.targetState.empty() ? 0 : 1;
        else if (slot.id == "reasoning_steps") count = bindings.reasoningSteps.size();
        else if (slot.id == "answer") count = bindings.answer.empty() ? 0 : 1;
        else if (slot.id == "evidence") count = bindings.evidence.size();
        else if (slot.id == "atoms") count = bindings.atoms.size();

        if (slot.required && count == 0) return fail("Required slot '" + slot.id + "' is empty");
        if (slot.minItems > 0 && count < slot.minItems) {
            return fail("Slot '" + slot.id + "' needs at least "
                        + std::to_string(slot.minItems) + " items");
        }
        if (slot.maxItems > 0 && count > slot.maxItems) {
            return fail("Slot '" + slot.id + "' exceeds "
                        + std::to_string(slot.maxItems) + " items");
        }
    }

    for (size_t index = 0; index < bindings.evidence.size(); ++index) {
        const auto& evidence = bindings.evidence[index];
        if (evidence.criterion.empty()) {
            return fail("Evidence " + std::to_string(index + 1) + " has no criterion");
        }
        if (evidence.quote.empty() || document.text.find(evidence.quote) == std::string::npos) {
            return fail("Evidence " + std::to_string(index + 1)
                        + " is not an exact quote from the document");
        }
    }

    std::unordered_set<std::string> atomIds;
    for (size_t index = 0; index < bindings.atoms.size(); ++index) {
        const auto& atom = bindings.atoms[index];
        if (atom.id.empty()) return fail("Atom " + std::to_string(index + 1) + " has no id");
        if (!atomIds.insert(atom.id).second) return fail("Duplicate atom id '" + atom.id + "'");
        if (atom.value.empty()) return fail("Atom " + std::to_string(index + 1) + " has no value");
        if (atom.type != GeneratorValueType::Integer
            && atom.type != GeneratorValueType::Float
            && atom.type != GeneratorValueType::Boolean
            && atom.type != GeneratorValueType::String) {
            return fail("Atom '" + atom.id + "' does not use a supported primitive type");
        }
        if (atom.type == GeneratorValueType::Integer && !isInteger(atom.value)) {
            return fail("Atom '" + atom.id + "' is not a valid integer");
        }
        if (atom.type == GeneratorValueType::Float && !isFloat(atom.value)) {
            return fail("Atom '" + atom.id + "' is not a valid float");
        }
        if (atom.type == GeneratorValueType::Boolean
            && atom.value != "true" && atom.value != "false") {
            return fail("Atom '" + atom.id + "' is not a valid boolean");
        }
        if (!atom.derived) {
            if (atom.sourceQuote.empty()) {
                return fail("Atom '" + atom.id + "' needs a source quote");
            }
            if (document.text.find(atom.sourceQuote) == std::string::npos) {
                return fail("Atom '" + atom.id + "' quote is not present in the document");
            }
        }
    }

    auto validateAtomReferences = [&](const std::string& text,
                                      const std::string& field) -> bool {
        size_t position = 0;
        constexpr const char* prefix = "{{atom:";
        constexpr size_t prefixLength = 7;
        while ((position = text.find(prefix, position)) != std::string::npos) {
            const size_t end = text.find("}}", position + prefixLength);
            if (end == std::string::npos) return fail("Malformed atom reference in " + field);
            const std::string id = text.substr(position + prefixLength,
                                               end - position - prefixLength);
            if (!atomIds.count(id)) {
                return fail("Unknown atom reference '" + id + "' in " + field);
            }
            position = end + 2;
        }
        return true;
    };
    if (!validateAtomReferences(bindings.question, "question")
        || !validateAtomReferences(bindings.targetState, "target_state")
        || !validateAtomReferences(bindings.answer, "answer")) return false;
    for (size_t index = 0; index < bindings.reasoningSteps.size(); ++index) {
        if (!validateAtomReferences(bindings.reasoningSteps[index],
                                    "reasoning step " + std::to_string(index + 1))) return false;
    }
    for (size_t index = 0; index < bindings.evidence.size(); ++index) {
        if (!validateAtomReferences(bindings.evidence[index].criterion,
                                    "evidence criterion " + std::to_string(index + 1))) return false;
    }

    bindings.valid = true;
    return true;
}

const GeneratorLexiconGroup* ConceptGenerator::findGroup(const std::string& id) const {
    const auto it = std::find_if(groups_.begin(), groups_.end(),
        [&](const GeneratorLexiconGroup& group) { return group.id == id; });
    return it == groups_.end() ? nullptr : &*it;
}

const GeneratorLexiconEntry* ConceptGenerator::selectEntry(
    const GeneratorLexiconGroup& group,
    uint64_t baseSeed,
    uint64_t iteration,
    size_t groupOrdinal,
    GeneratorTraversal traversal,
    size_t& selectedIndex) const {
    if (group.entries.empty()) return nullptr;
    if (traversal == GeneratorTraversal::RoundRobin) {
        selectedIndex = static_cast<size_t>((baseSeed + iteration + groupOrdinal)
                                            % group.entries.size());
    } else {
        const uint64_t mixed = splitmix64(baseSeed ^ fnv1a(group.id)
                                          ^ splitmix64(iteration + groupOrdinal));
        selectedIndex = static_cast<size_t>(mixed % group.entries.size());
    }
    return &group.entries[selectedIndex];
}

std::string ConceptGenerator::renderField(
    const GeneratorFrameDefinition& frame,
    const std::string& target,
    const GeneratorBindingSet& bindings,
    uint64_t baseSeed,
    uint64_t iteration,
    GeneratorTraversal traversal,
    std::vector<GeneratorSelection>& selections) const {
    const auto fieldIt = std::find_if(frame.fields.begin(), frame.fields.end(),
        [&](const GeneratorFieldTemplate& field) { return field.target == target; });
    if (fieldIt == frame.fields.end()) return {};

    size_t groupOrdinal = selections.size();
    std::ostringstream output;
    for (const auto& segment : fieldIt->segments) {
        std::string value;
        if (segment.kind == GeneratorSegmentKind::Literal) {
            value = segment.text;
        } else if (segment.kind == GeneratorSegmentKind::Binding) {
            value = bindingText(bindings, segment.key);
            if (value.empty()) value = segment.fallback;
        } else if (segment.kind == GeneratorSegmentKind::Lexicon) {
            const auto* group = findGroup(segment.key);
            if (group) {
                size_t selectedIndex = 0;
                const auto* entry = selectEntry(*group, baseSeed, iteration,
                                                groupOrdinal++, traversal, selectedIndex);
                if (entry) {
                    value = entry->text;
                    selections.push_back({group->id, selectedIndex, entry->type, entry->text});
                }
            }
        }
        if (value.empty() && segment.optional) continue;
        output << segment.prefix << value << segment.suffix;
    }
    return sentenceCase(output.str());
}

GeneratedConceptFrame ConceptGenerator::assemble(
    size_t frameIndex,
    const GeneratorBindingSet& bindings,
    uint64_t baseSeed,
    uint64_t iteration,
    GeneratorTraversal traversal) const {
    GeneratedConceptFrame generated;
    generated.seed = baseSeed;
    generated.iteration = iteration;
    generated.sourceBound = bindings.valid;
    if (frameIndex >= frames_.size() || !bindings.valid) return generated;

    const auto& frame = frames_[frameIndex];
    auto& block = generated.conceptBlock;
    block.name = frame.label + " — " + bindings.documentId;
    block.format_type = frame.formatType;
    block.source_sequence_id = bindings.documentId;

    block.prompt = renderField(frame, "prompt", bindings, baseSeed, iteration,
                               traversal, generated.selections);
    if (block.prompt.empty()) block.prompt = bindings.question;
    block.prompt = applyAtomBindings(block.prompt, bindings.atoms);

    block.intermediates.reserve(bindings.reasoningSteps.size());
    for (const auto& step : bindings.reasoningSteps) {
        block.intermediates.push_back(applyAtomBindings(step, bindings.atoms));
    }

    block.answer = renderField(frame, "answer", bindings, baseSeed, iteration,
                               traversal, generated.selections);
    if (block.answer.empty()) block.answer = bindings.answer;
    block.answer = applyAtomBindings(block.answer, bindings.atoms);

    if (!bindings.targetState.empty() || !bindings.evidence.empty()) {
        GRIM::ConceptBlockGoal goal;
        goal.target_state = applyAtomBindings(bindings.targetState, bindings.atoms);
        for (const auto& evidence : bindings.evidence) {
            goal.success_criteria.push_back({
                applyAtomBindings(evidence.criterion, bindings.atoms),
                evidence.quote
            });
        }
        block.goal = std::move(goal);
    }

    const std::string identity = frame.id + "|" + bindings.documentId + "|"
        + std::to_string(baseSeed) + "|" + std::to_string(iteration) + "|"
        + block.prompt + "|" + block.answer;
    const uint64_t idHigh = fnv1a(identity);
    const uint64_t idLow = fnv1a("generator|" + identity);
    std::ostringstream id;
    id << std::hex << std::setfill('0') << std::setw(16) << idHigh
       << std::setw(16) << idLow;
    block.id = id.str();
    block.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    block.recomputeDerived();
    return generated;
}

} // namespace GRIM::DataCollection
